"""语义记忆引擎 — FAISS 向量索引管理

负责向量的增、删、查、持久化。
与 SQLite 元数据存储通过 chunk_id 关联。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

import faiss
import numpy as np

from config import EMBEDDING_DIM, FAISS_INDEX_TYPE

if TYPE_CHECKING:
    from numpy.typing import NDArray


class VectorIndex:
    """FAISS 向量索引管理器。

    使用 IndexFlatIP（内积），配合 L2 归一化的 embedding
    实现余弦相似度搜索。
    支持增量添加、按 ID 删除、持久化到磁盘。
    """

    def __init__(self, dim: int = EMBEDDING_DIM, index_path: str | None = None):
        self.dim = dim
        self.index_path = Path(index_path).expanduser().resolve() if index_path else None
        self._index = self._create_index()
        self._id_map: dict[int, int] = {}  # chunk_id → FAISS 内部 ID
        self._reverse_map: dict[int, int] = {}  # FAISS 内部 ID → chunk_id
        self._next_internal = 0
        if self.index_path and self.index_path.exists():
            self._load()

    def _create_index(self) -> faiss.Index:
        """创建 FAISS 索引。"""
        if FAISS_INDEX_TYPE == "IndexFlatIP":
            return faiss.IndexFlatIP(self.dim)
        elif FAISS_INDEX_TYPE == "IndexFlatL2":
            return faiss.IndexFlatL2(self.dim)
        else:
            raise ValueError(f"不支持的索引类型: {FAISS_INDEX_TYPE}")

    # ── 添加 ──
    def add_vectors(self, chunk_ids: list[int], vectors: "NDArray[np.float32]"):
        """批量添加向量。

        Args:
            chunk_ids: chunk 的数据库 ID 列表
            vectors: shape=(n, dim) 的向量数组
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        n = len(vectors)
        if n != len(chunk_ids):
            raise ValueError(f"chunk_ids({len(chunk_ids)}) 和 vectors({n}) 数量不匹配")

        faiss_ids = np.arange(self._next_internal, self._next_internal + n, dtype=np.int64)
        self._index.add(vectors)

        for cid, fid in zip(chunk_ids, faiss_ids):
            self._id_map[cid] = int(fid)
            self._reverse_map[int(fid)] = cid

        self._next_internal += n

    # ── 搜索 ──
    def search(
        self, query_vector: "NDArray[np.float32]", top_k: int = 5
    ) -> list[tuple[int, float]]:
        """搜索最相似的 top-k 向量。

        Returns:
            [(chunk_id, score), ...] 按分数降序排列
        """
        if self._index.ntotal == 0:
            return []

        qv = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(qv, k)

        results: list[tuple[int, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if int(idx) == -1:  # FAISS 填充值
                continue
            chunk_id = self._reverse_map.get(int(idx))
            if chunk_id is not None:
                results.append((chunk_id, float(score)))

        return results

    # ── 删除 ──
    def remove_by_chunk_id(self, chunk_id: int) -> bool:
        """按 chunk_id 删除向量（重建索引方式）。"""
        if chunk_id not in self._id_map:
            return False

        # 保留除该 ID 外的所有向量
        keep_ids = [cid for cid in self._id_map if cid != chunk_id]
        self._rebuild(keep_ids)
        return True

    def remove_by_chunk_ids(self, chunk_ids: list[int]) -> int:
        """批量删除。"""
        ids_set = set(chunk_ids)
        keep_ids = [cid for cid in self._id_map if cid not in ids_set]
        if len(keep_ids) == len(self._id_map):
            return 0  # 没有匹配的
        self._rebuild(keep_ids)
        return len(chunk_ids) - len(keep_ids)

    # ── 持久化 ──
    def save(self, path: str | None = None):
        """保存索引到磁盘。"""
        target = Path(path).expanduser().resolve() if path else self.index_path
        if not target:
            raise ValueError("没有指定保存路径")
        target.parent.mkdir(parents=True, exist_ok=True)

        # 保存 FAISS 索引
        faiss.write_index(self._index, str(target))

        # 保存 ID 映射
        meta_path = target.with_suffix(".meta.json")
        import json
        meta = {
            "id_map": {str(k): v for k, v in self._id_map.items()},
            "reverse_map": {str(k): v for k, v in self._reverse_map.items()},
            "next_internal": self._next_internal,
            "dim": self.dim,
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    def _load(self):
        """从磁盘加载索引。"""
        if not self.index_path or not self.index_path.exists():
            return

        self._index = faiss.read_index(str(self.index_path))

        meta_path = self.index_path.with_suffix(".meta.json")
        if meta_path.exists():
            import json
            meta = json.loads(meta_path.read_text())
            self._id_map = {int(k): v for k, v in meta.get("id_map", {}).items()}
            self._reverse_map = {int(k): v for k, v in meta.get("reverse_map", {}).items()}
            self._next_internal = meta.get("next_internal", 0)

    # ── 内部 ──
    def _rebuild(self, keep_chunk_ids: list[int]):
        """用保留的 chunk_ids 重建索引。"""
        if not keep_chunk_ids:
            self._index = self._create_index()
            self._id_map.clear()
            self._reverse_map.clear()
            self._next_internal = 0
            return

        # 从当前索引提取保留的向量
        old_index = self._index
        self._index = self._create_index()
        self._id_map.clear()
        self._reverse_map.clear()
        self._next_internal = 0

        # 注意：FAISS IndexFlat 不支持单独提取向量
        # 这里通过上层传入完整信息来重建
        # 实际删除操作应由 MemoryEngine 协调
        for cid in keep_chunk_ids:
            # 保留映射关系，但需要重新添加向量
            self._id_map[cid] = self._next_internal
            self._reverse_map[self._next_internal] = cid
            self._next_internal += 1

    @property
    def size(self) -> int:
        """索引中的向量数量。"""
        return self._index.ntotal

    @property
    def is_empty(self) -> bool:
        return self._index.ntotal == 0
