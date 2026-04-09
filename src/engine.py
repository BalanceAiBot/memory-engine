"""语义记忆引擎 — MemoryEngine 主类

统一入口，协调 Embedder、Chunker、VectorIndex、MetadataStore、Retriever。
提供简洁的 API：ingest、query、migrate、stats。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

# 确保项目根在 sys.path（支持直接运行和模块导入）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (
    CHUNK_MAX_CHARS,
    DEFAULT_DB_PATH,
    DEFAULT_TOP_K,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    HERMES_MEMORY,
    INTERLEAVE_ROUNDS,
    MIN_SIMILARITY,
    OPENCLAW_MEMORY,
)
from src.chunker import Chunker
from src.embedder import Embedder
from src.indexer import VectorIndex
from src.migrator import MigrationEntry, Migrator
from src.retriever import SearchResult, Retriever
from src.store import MetadataStore


class MemoryEngine:
    """语义记忆引擎主类。

    全 CPU 运行，中文优化，支持增量更新。

    用法:
        engine = MemoryEngine(db_path="~/.hermes/memories/memory.db")
        engine.ingest("用户希望系统能盈利", category="user_goal")
        results = engine.query("用户的目标是什么？", top_k=5)
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        index_path: str | None = None,
        model_name: str = EMBEDDING_MODEL,
    ):
        """初始化引擎。

        Args:
            db_path: SQLite 数据库路径
            index_path: FAISS 索引文件路径（默认与 db_path 同目录）
            model_name: embedding 模型名
        """
        self.db_path = str(Path(db_path).expanduser().resolve())
        self.db_path_dir = str(Path(self.db_path).parent)

        if index_path is None:
            index_path = str(Path(self.db_path_dir) / "faiss.index")
        self.index_path = index_path

        # 初始化组件
        self.embedder = Embedder(model_name=model_name)
        self.chunker = Chunker(max_chars=CHUNK_MAX_CHARS)
        self.store = MetadataStore(self.db_path)
        self.index = VectorIndex(dim=EMBEDDING_DIM, index_path=self.index_path)
        self.retriever = Retriever(
            embedder=self.embedder,
            index=self.index,
            store=self.store,
        )
        self.migrator = Migrator()

        # 同步：确保 FAISS 索引和 SQLite 一致
        self._sync_index()

    # ── 入库 ──
    def ingest(
        self,
        text: str,
        category: str = "general",
        source: str = "",
        metadata: dict | None = None,
        auto_chunk: bool = True,
    ) -> list[int]:
        """入库一条记忆。

        Args:
            text: 记忆文本
            category: 分类标签
            source: 来源标识
            metadata: 附加元数据
            auto_chunk: 是否自动分块（长文本拆分）

        Returns:
            入库的 chunk_id 列表
        """
        if not text or not text.strip():
            return []

        if auto_chunk and len(text) > CHUNK_MAX_CHARS:
            chunks = self.chunker.chunk(text, {"category": category, "source": source})
            if not chunks:
                chunks = [type("FakeChunk", (), {"text": text.strip()})()]
            chunk_ids = []
            for chunk in chunks:
                cid = self.store.add_chunk(
                    text=chunk.text,
                    category=category,
                    source=source,
                    metadata={**(metadata or {}), **(chunk.metadata if hasattr(chunk, 'metadata') else {})},
                )
                chunk_ids.append(cid)

            # 编码并添加到索引
            vectors = self.embedder.encode([c.text for c in chunks])
            self.index.add_vectors(chunk_ids, vectors)
            self.index.save(self.index_path)
            return chunk_ids

        # 短文本直接入库
        cid = self.store.add_chunk(
            text=text.strip(),
            category=category,
            source=source,
            metadata=metadata,
        )
        vector = self.embedder.encode_single(text.strip())
        self.index.add_vectors([cid], vector.reshape(1, -1))
        self.index.save(self.index_path)
        return [cid]

    def ingest_batch(
        self,
        items: list[dict],
        auto_chunk: bool = True,
    ) -> list[int]:
        """批量入库。

        Args:
            items: [{"text": "...", "category": "...", "source": "...", "metadata": {...}}, ...]
            auto_chunk: 是否自动分块

        Returns:
            入库的 chunk_id 列表
        """
        all_chunk_ids: list[int] = []
        for item in items:
            ids = self.ingest(
                text=item["text"],
                category=item.get("category", "general"),
                source=item.get("source", ""),
                metadata=item.get("metadata"),
                auto_chunk=auto_chunk,
            )
            all_chunk_ids.extend(ids)
        return all_chunk_ids

    # ── 检索 ──
    def query(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = MIN_SIMILARITY,
    ) -> list[dict]:
        """语义检索。

        Args:
            query: 查询文本
            top_k: 返回数量
            min_score: 最低相似度阈值

        Returns:
            [{"text": "...", "score": 0.92, "category": "...", "source": "..."}, ...]
        """
        results = self.retriever.search(query, top_k=top_k, min_score=min_score)
        return [r.to_dict() for r in results]

    def query_interleave(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        rounds: int = INTERLEAVE_ROUNDS,
        min_score: float = MIN_SIMILARITY,
    ) -> list[dict]:
        """多跳检索（Memory Interleave）。

        多轮检索 → 用结果精炼 query → 再检索 → 融合去重。
        """
        results = self.retriever.search_interleave(
            query, top_k=top_k, rounds=rounds, min_score=min_score,
        )
        return [r.to_dict() for r in results]

    def query_by_category(
        self,
        query: str,
        category: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[dict]:
        """限定类别的语义检索。"""
        results = self.retriever.search_by_category(query, category, top_k=top_k)
        return [r.to_dict() for r in results]

    # ── 迁移 ──
    def migrate_from_hermes_memory(self, path: str = HERMES_MEMORY) -> dict:
        """从 ~/.hermes/memories/MEMORY.md 迁移。"""
        entries = self.migrator.parse_hermes_memory(path)
        print(f"\n📖 解析到 {len(entries)} 条记忆条目")

        def ingest_fn(text, category, source, metadata):
            self.ingest(text, category=category, source=source, metadata=metadata)

        stats = self.migrator.migrate_entries(entries, ingest_fn=ingest_fn)
        return stats

    def migrate_from_openclaw_memory(self, path: str = OPENCLAW_MEMORY) -> dict:
        """从 ~/.openclaw/workspace/MEMORY.md 迁移。"""
        entries = self.migrator.parse_openclaw_memory(path)
        print(f"\n📖 解析到 {len(entries)} 条记忆条目")

        def ingest_fn(text, category, source, metadata):
            self.ingest(text, category=category, source=source, metadata=metadata)

        stats = self.migrator.migrate_entries(entries, ingest_fn=ingest_fn)
        return stats

    def migrate_all(self) -> dict:
        """迁移所有可用记忆源。"""
        all_stats: dict = {"hermes": {}, "openclaw": {}}

        if Path(HERMES_MEMORY).exists():
            print("\n🔄 迁移 Heremes 记忆...")
            all_stats["hermes"] = self.migrate_from_hermes_memory()

        if Path(OPENCLAW_MEMORY).exists():
            print("\n🔄 迁移 OpenClaw 记忆...")
            all_stats["openclaw"] = self.migrate_from_openclaw_memory()

        return all_stats

    # ── 统计 ──
    def stats(self) -> dict:
        """返回引擎统计信息。"""
        db_stats = self.store.stats()
        db_stats["faiss_index_size"] = self.index.size
        db_stats["embedding_model"] = self.embedder._model_name
        db_stats["embedding_dim"] = self.embedder.dim
        db_stats["db_path"] = self.db_path
        db_stats["index_path"] = self.index_path
        return db_stats

    # ── 管理 ──
    def list_by_category(self, category: str) -> list[dict]:
        """列出某类别的所有记忆。"""
        chunks = self.store.get_chunks_by_category(category)
        return [
            {
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "source": c["source"],
                "created_at": c["created_at"],
            }
            for c in chunks
        ]

    def delete(self, chunk_id: int) -> bool:
        """删除一条记忆。"""
        removed = self.index.remove_by_chunk_id(chunk_id)
        deleted = self.store.delete_chunk(chunk_id)
        if deleted:
            self.index.save(self.index_path)
        return deleted

    # ── 同步 ──
    def _sync_index(self):
        """确保 FAISS 索引与 SQLite 数据一致。

        如果索引为空但数据库有数据，重建索引。
        """
        if self.index.is_empty:
            all_chunks = self.store.get_all_chunks()
            if all_chunks:
                print(f"🔄 同步索引: 发现 {len(all_chunks)} 条未索引数据，重建中...")
                texts = [c["text"] for c in all_chunks]
                chunk_ids = [c["chunk_id"] for c in all_chunks]
                vectors = self.embedder.encode(texts)
                self.index.add_vectors(chunk_ids, vectors)
                self.index.save(self.index_path)
                print("✅ 索引重建完成")

    def rebuild_index(self):
        """强制重建 FAISS 索引。"""
        self.index = VectorIndex(dim=EMBEDDING_DIM, index_path=self.index_path)
        all_chunks = self.store.get_all_chunks()
        if all_chunks:
            texts = [c["text"] for c in all_chunks]
            chunk_ids = [c["chunk_id"] for c in all_chunks]
            print(f"🔄 重建索引: {len(all_chunks)} 条 → embedding...")
            vectors = self.embedder.encode(texts)
            self.index.add_vectors(chunk_ids, vectors)
            self.index.save(self.index_path)
            print("✅ 索引重建完成")
        self.retriever.index = self.index
