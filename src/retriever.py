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
            self.bm25_corpus_tokens = []
            self.chunk_id_to_index = {}
            return

        # 建立 chunk_id -> index 的映射
        self.chunk_id_to_index = {}
        self.bm25_corpus_tokens = []
        
        for i, c in enumerate(chunks):
            cid = c['chunk_id']
            self.chunk_id_to_index[cid] = i
            self.bm25_corpus_tokens.append(c['text'])

        self.bm25 = SimpleBM25(self.bm25_corpus_tokens)
        print(f"[BM25] Index built.")

    def _expand_query(self, query: str) -> str:
        """
        查询扩展：将口语/黑话转换为书面语。
        例如：“挂了” -> “崩溃/错误”，“慢了” -> “延迟”。
        """
        replacements = {
            "挂了": "崩溃 错误 失败",
            "崩了": "崩溃 错误 失败",
            "挂了": "崩溃 错误 失败",
            "慢了": "延迟 性能",
            "卡": "延迟 性能",
            "修了啥": "修复 问题",
            "修了": "修复",
            "啥": "什么",
            "咋回事": "原因 错误",
            "咋了": "原因 错误",
            "出了啥": "发生 问题",
            "上次": "最近",
            "查下": "查看 搜索",
        }
        for k, v in replacements.items():
            if k in query:
                query = query.replace(k, v)
        return query

    def _get_freshness_score(self, text: str, query: str = "") -> float:
        """
        计算时间衰减分数。
        针对不同类型的查询，衰减策略不同。
        """
        import re
        import datetime
        dates = re.findall(r'(\d{4})-(\d{2})-(\d{2})', text)
        if not dates:
            return 1.0

        latest = max([datetime.date(int(y), int(m), int(d)) for y, m, d in dates])
        today = datetime.date.today()
        days_diff = max(0, (today - latest).days)
        
        # 智能衰减策略
        # 1. 如果是查询“故障/修复/审计” (Tech Query)，强衰减 (用户关心最近的)
        # 2. 如果是查询“架构/工具/定义” (Fact Query)，弱衰减 (知识不会过期)
        
        tech_keywords = ["修", "错", "崩", "挂", "问题", "审计", "延迟", "慢", "bug", "死锁", "超时", "日志"]
        is_tech_query = any(kw in query for kw in tech_keywords)
        
        if is_tech_query:
            # 强衰减：1周后降为一半
            # 1.0 / (1 + 0.1 * 7) approx 0.5
            return 1.0 / (1.0 + 0.1 * days_diff)
        else:
            # 弱衰减：1个月后才明显降低
            # 1.0 / (1 + 0.01 * 30) approx 0.76
            return 1.0 / (1.0 + 0.01 * days_diff)

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
        # 1. Query Expansion (口语转书面语)
        expanded_query = self._expand_query(query)
        search_query = expanded_query if hybrid else query
        
        if self.index.is_empty:
            return []

        query_vec = self.embedder.encode_single(search_query)
        
        # 2. 向量检索 (召回更多候选)
        vec_top_k = top_k * 3 if hybrid else top_k
        vec_hits = self.index.search(query_vec, vec_top_k)
        
        if not vec_hits:
            return []

        # 3. BM25 检索 (关键词兜底 + 停用词过滤)
        bm25_scores = {}
        if hybrid and self.bm25:
            bm25_ranked = self.bm25.search(query, top_k * 3)
            index_to_chunk_id = {v: k for k, v in self.chunk_id_to_index.items()}
            for idx, score in bm25_ranked:
                if idx in index_to_chunk_id:
                    chunk_id = index_to_chunk_id[idx]
                    if score > 0: 
                        bm25_scores[chunk_id] = score

        # 4. 融合结果 (RRF) + 类别优先权
        rrf_results = {}
        k = 60

        # 检测查询意图：如果是技术/问题类查询，提升 maintenance_log 权重
        tech_keywords = ["修", "错", "崩", "挂", "问题", "审计", "延迟", "慢", "bug", "死锁", "超时"]
        is_tech_query = any(kw in query for kw in tech_keywords) or any(kw in expanded_query for kw in tech_keywords)

        vec_chunk_ids = [cid for cid, _ in vec_hits]
        vec_chunks = self.store.get_chunks_by_ids(vec_chunk_ids)
        vec_map = {c['chunk_id']: c for c in vec_chunks}

        # 处理向量结果
        for rank, (cid, score) in enumerate(vec_hits):
            meta = vec_map.get(cid, {})
            category = meta.get('category', '')
            
            # 基础 RRF 分数
            score_rrf = 1.0 / (k + rank)
            
            # 类别加成 (Category Boost)
            category_boost = 1.0
            if is_tech_query:
                if category == 'maintenance_log': category_boost = 3.0  # 大幅提高维护日志权重
                elif category == 'dev_tool': category_boost = 2.0
                elif category == 'user_profile': category_boost = 0.5  # 降低用户画像权重
                elif category == 'learning_note': category_boost = 0.7 # 降低学习笔记权重 (非紧急故障)
            
            rrf_results[cid] = rrf_results.get(cid, 0) + score_rrf * category_boost

        # 处理 BM25 结果 (技术查询中，关键词匹配权重更高)
        if hybrid and self.bm25:
            bm25_sorted = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
            for rank, (cid, _) in enumerate(bm25_sorted):
                meta = self.store.get_chunk(cid)
                category = meta.get('category', '') if meta else ''
                
                bm25_weight = 3.0
                if is_tech_query:
                    if category == 'maintenance_log': bm25_weight = 6.0
                    elif category == 'user_profile': bm25_weight = 0.5

                rrf_results[cid] = rrf_results.get(cid, 0) + bm25_weight / (k + rank)

        # 5. 排序并回填元数据
        final_chunk_ids = sorted(rrf_results, key=lambda x: rrf_results[x], reverse=True)[:top_k]
        final_chunks = self.store.get_chunks_by_ids(final_chunk_ids)
        chunk_map = {c['chunk_id']: c for c in final_chunks}

        results = []
        for cid in final_chunk_ids:
            meta = chunk_map.get(cid)
            if meta is None:
                continue
            
            if hybrid:
                final_score = rrf_results[cid]
            else:
                for v_cid, v_score in vec_hits:
                    if v_cid == cid:
                        final_score = v_score
                        break
                else:
                    final_score = 0.0

            # 应用时间衰减 (新记忆优先)
            time_weight = self._get_freshness_score(meta['text'])
            final_score *= time_weight

            results.append(SearchResult(
                chunk_id=cid,
                text=meta['text'],
                score=round(final_score, 4),
                category=meta.get('category', ''),
                source=meta.get('source', ''),
            ))

        return results
