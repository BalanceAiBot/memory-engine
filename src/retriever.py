"""语义记忆引擎 — 检索器（含多跳推理）

支持：
- 单轮语义检索
- 多跳 Memory Interleave（检索→精炼→再检索）
- 结果去重与融合
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List

from config import DEFAULT_TOP_K, INTERLEAVE_ROUNDS, MIN_SIMILARITY

if TYPE_CHECKING:
    from src.embedder import Embedder
    from src.indexer import VectorIndex
    from src.store import MetadataStore


class SearchResult:
    """单条检索结果"""
    __slots__ = ("chunk_id", "text", "score", "category", "source", "metadata", "round_num")

    def __init__(
        self,
        chunk_id: int,
        text: str,
        score: float,
        category: str,
        source: str,
        metadata: dict,
        round_num: int = 1,
    ):
        self.chunk_id = chunk_id
        self.text = text
        self.score = round(score, 4)
        self.category = category
        self.source = source
        self.metadata = metadata
        self.round_num = round_num

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": self.score,
            "category": self.category,
            "source": self.source,
            "metadata": self.metadata,
            "round_num": self.round_num,
        }

    def __repr__(self):
        return f"SearchResult(id={self.chunk_id}, score={self.score}, cat={self.category})"


class Retriever:
    """语义检索器。

    封装 FAISS 搜索 + SQLite 元数据回填，
    支持单轮和多跳检索。
    """

    def __init__(
        self,
        embedder: "Embedder",
        index: "VectorIndex",
        store: "MetadataStore",
    ):
        self.embedder = embedder
        self.index = index
        self.store = store

    # ── 单轮检索 ──
    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = MIN_SIMILARITY,
    ) -> list[SearchResult]:
        """单轮语义检索。

        Args:
            query: 查询文本
            top_k: 返回数量
            min_score: 最低相似度阈值

        Returns:
            SearchResult 列表，按分数降序
        """
        if self.index.is_empty:
            return []

        query_vec = self.embedder.encode_single(query)
        hits = self.index.search(query_vec, top_k)

        if not hits:
            return []

        # 过滤低分
        hits = [(cid, score) for cid, score in hits if score >= min_score]
        if not hits:
            return []

        chunk_ids = [cid for cid, _ in hits]
        chunks = self.store.get_chunks_by_ids(chunk_ids)
        chunk_map = {c["chunk_id"]: c for c in chunks}

        results: list[SearchResult] = []
        for cid, score in hits:
            meta = chunk_map.get(cid)
            if meta is None:
                continue
            results.append(SearchResult(
                chunk_id=cid,
                text=meta["text"],
                score=score,
                category=meta["category"],
                source=meta["source"],
                metadata=meta["metadata"],
                round_num=1,
            ))

        return results

    # ── 多跳检索 (Memory Interleave) ──
    def search_interleave(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        rounds: int = INTERLEAVE_ROUNDS,
        min_score: float = MIN_SIMILARITY,
    ) -> list[SearchResult]:
        """多轮检索精炼。

        策略：
        1. 第一轮用原始 query 检索
        2. 将 top-1 结果作为上下文，扩展 query
        3. 后续轮次用扩展后的 query 检索
        4. 合并所有轮次结果，去重后按分数排序
        """
        all_results: dict[int, SearchResult] = {}  # chunk_id → result
        refined_query = query

        for round_num in range(1, rounds + 1):
            hits = self.search(refined_query, top_k=top_k, min_score=min_score)

            if not hits:
                break

            # 收集本轮结果
            for result in hits:
                result.round_num = round_num
                if result.chunk_id not in all_results:
                    all_results[result.chunk_id] = result
                else:
                    # 保留最高分
                    if result.score > all_results[result.chunk_id].score:
                        all_results[result.chunk_id] = result

            # 精炼 query：取本轮 top-1 结果拼接
            if round_num < rounds and hits:
                top_result = hits[0]
                refined_query = f"{query} 相关上下文: {top_result.text[:200]}"

        # 按分数降序
        sorted_results = sorted(
            all_results.values(), key=lambda r: r.score, reverse=True
        )
        return sorted_results

    # ── 按类别检索 ──
    def search_by_category(
        self,
        query: str,
        category: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[SearchResult]:
        """限定类别的语义检索。"""
        results = self.search(query, top_k=top_k * 3)  # 扩大候选集
        return [r for r in results if r.category == category][:top_k]
