"""Retriever Module - Vector Search & Hybrid Search."""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING, List

from config import DEFAULT_TOP_K, INTERLEAVE_ROUNDS, MIN_SIMILARITY
from src.bm25 import SimpleBM25

if TYPE_CHECKING:
    from src.embedder import Embedder
    from src.store import MetadataStore
    from src.indexer import VectorIndex


class SearchResult:
    """单个检索结果的数据结构。"""
    def __init__(self, chunk_id: int, text: str, score: float, category: str, source: str, round_num: int = 1):
        self.chunk_id = chunk_id
        self.text = text
        self.score = score
        self.category = category
        self.source = source
        self.round_num = round_num

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": self.score,
            "category": self.category,
            "source": self.source,
            "round_num": self.round_num,
        }


class Retriever:
    """语义检索器。"""

    def __init__(
        self,
        embedder: "Embedder",
        index: "VectorIndex",
        store: "MetadataStore",
    ):
        self.embedder = embedder
        self.index = index
        self.store = store
        
        # BM25 Index
        self.bm25 = None
        self.bm25_corpus_tokens = []

    def update_bm25_index(self, chunks: list[dict]):
        """使用当前数据库中的 chunks 重建 BM25 索引。"""
        print(f"[BM25] Rebuilding index with {len(chunks)} chunks.")
        if not chunks:
            self.bm25 = None
            return

        # 预处理语料
        self.bm25_corpus_tokens = []
        for c in chunks:
            # Use SimpleBM25 internal tokenizer
            self.bm25_corpus_tokens.append(c['text'])

        self.bm25 = SimpleBM25(self.bm25_corpus_tokens)
        print(f"[BM25] Index built.")

    def _expand_query(self, query: str) -> str:
        """简单的查询扩展/纠错，提升口语化查询的召回率。"""
        replacements = {
            "修了啥": "修复 问题",
            "修了": "修复",
            "啥": "什么",
            "咋回事": "原因 错误",
            "咋了": "原因 错误",
            "出了啥": "发生 问题",
            "上次": "最近", # "Last time" often implies "recent"
        }
        for k, v in replacements.items():
            if k in query:
                query = query.replace(k, v)
        return query

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = MIN_SIMILARITY,
        hybrid: bool = True,
    ) -> list[SearchResult]:
        """
        执行检索。
        """
        # 1. Query Expansion
        expanded_query = self._expand_query(query)
        if expanded_query != query:
            # print(f"[Hybrid] Expanded query: '{query}' -> '{expanded_query}'")
            pass

        # 使用扩展后的查询进行搜索 (如果 hybrid=True)
        search_query = expanded_query if hybrid else query
        if self.index.is_empty:
            return []

        query_vec = self.embedder.encode_single(search_query)
        
        # 1. 向量检索 (召回更多候选)
        import sys
        print(f"[DEBUG] Query: '{query}', Expanded: '{search_query}'", file=sys.stderr)
        # 混合模式下多召回一些，因为 BM25 会过滤掉一些
        vec_top_k = top_k * 3 if hybrid else top_k
        vec_hits = self.index.search(query_vec, vec_top_k)
        print(f"[DEBUG] Vec Hits: {len(vec_hits)}", file=sys.stderr)
        if vec_hits:
            print(f"[DEBUG] Top Vec Hit: {vec_hits[0]}", file=sys.stderr)
        
        if not vec_hits:
            return []

        # 2. BM25 检索 (关键词兜底)
        bm25_scores = {}
        if hybrid and self.bm25:
            # ... (existing code)
            # DEBUG
            # print(f"[DEBUG] Query: {query}, BM25 Top: {bm25_sorted[:3]}")
            # BM25 返回的是 (index, score) 列表
            # 注意: SimpleBM25.search 需要 query 和 corpus_tokens
            # 我们的 corpus_tokens 就是 chunks 的 text
            # 为了对齐 ID，我们需要把 chunks 的 chunk_id 映射到 BM25 的 index
            
            all_chunks = self.store.get_all_chunks()
            if len(all_chunks) != len(self.bm25_corpus_tokens):
                # 如果索引和数据库不一致，刷新 BM25
                self.update_bm25_index(all_chunks)
            
            # 计算 BM25 分数
            bm25_ranked = self.bm25.search(query, self.bm25_corpus_tokens, top_k * 3)
            # bm25_ranked is list of (index_in_corpus, score)
            
            # Map corpus index back to chunk_id
            for idx, score in bm25_ranked:
                if idx < len(all_chunks):
                    chunk_id = all_chunks[idx]['chunk_id']
                    bm25_scores[chunk_id] = score

        # 3. 融合结果 (RRF - Reciprocal Rank Fusion)
        # Score = (alpha * vec_rank_score) + ((1-alpha) * bm25_rank_score)
        # 简化版: 直接使用 RRF
        
        rrf_results = {}
        k = 60  # RRF 常数

        # 处理向量结果
        vec_chunk_ids = [cid for cid, _ in vec_hits]
        vec_chunks = self.store.get_chunks_by_ids(vec_chunk_ids)
        vec_map = {c['chunk_id']: c for c in vec_chunks}

        for rank, (cid, score) in enumerate(vec_hits):
            if score < min_score and not hybrid: # Pure vector mode threshold
                continue
            
            # RRF 贡献: 1 / (k + rank)
            # 同时保留原始分数用于加权 (可选)
            rrf_results[cid] = rrf_results.get(cid, 0) + 1.0 / (k + rank)

        # 处理 BM25 结果
        if hybrid and self.bm25:
            # 重新计算 BM25 排序用于 RRF
            bm25_sorted = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
            for rank, (cid, _) in enumerate(bm25_sorted):
                rrf_results[cid] = rrf_results.get(cid, 0) + 1.0 / (k + rank)

        # 4. 排序并回填元数据
        final_chunk_ids = sorted(rrf_results, key=lambda x: rrf_results[x], reverse=True)[:top_k]
        final_chunks = self.store.get_chunks_by_ids(final_chunk_ids)
        chunk_map = {c['chunk_id']: c for c in final_chunks}

        results = []
        for cid in final_chunk_ids:
            meta = chunk_map.get(cid)
            if meta is None:
                continue
            
            # 如果是混合模式，使用 RRF 分数，否则使用原始向量分数
            if hybrid:
                final_score = rrf_results[cid]
            else:
                # Find original vector score
                for v_cid, v_score in vec_hits:
                    if v_cid == cid:
                        final_score = v_score
                        break
                else:
                    final_score = 0.0

            results.append(SearchResult(
                chunk_id=cid,
                text=meta['text'],
                score=round(final_score, 4),
                category=meta.get('category', ''),
                source=meta.get('source', ''),
            ))

        return results
