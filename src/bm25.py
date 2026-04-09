"""轻量级 BM25 检索实现 (纯 Python，零依赖)"""
import math
import re
import os
import sys

# 尝试导入 jieba，如果不存在则使用备用分词
try:
    import jieba
    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False

class SimpleBM25:
    """
    一个极简的 BM25 实现，用于混合检索。
    不需要 jieba 或 rank_bm25，使用字符级 N-gram 分词。
    """
    # 停用词表：过滤掉无意义的通用词汇，提高技术词汇的权重
    STOP_WORDS = {
        "关于", "用户", "系统", "笔记", "核心", "知识点", "安装", "命令", 
        "记录", "成功", "失败", "完成", "什么", "怎么", "如何",
        "是", "的", "了", "在", "和", "与", "或", "我", "你", "他",
        "主要", "功能", "支持", "使用", "这个", "那个"
        # 注意：移除了 "问题"，因为在调试语境下它是重要关键词
    }

    def __init__(self, corpus, k1=1.5, b=0.75):
        """
        初始化 BM25 索引。
        :param corpus: 字符串列表，每个字符串代表一个文档。
        """
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lengths = []
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.tf = [] # Store TF matrix

        self._initialize(corpus)

    def _tokenize(self, text):
        """分词器：中文至少 2 字一组，英文按词，并过滤停用词。"""
        if HAS_JIEBA:
            return [t for t in jieba.cut(text.lower()) if t not in self.STOP_WORDS]
        
        # 备用方案
        en_tokens = re.findall(r'[a-z0-9]+', text.lower())
        cn_tokens = re.findall(r'[\u4e00-\u9fa5]{2,}', text)
        
        # 过滤停用词
        return [t for t in (en_tokens + cn_tokens) if t not in self.STOP_WORDS]

    def _initialize(self, corpus):
        """构建索引。"""
        nd = {} 
        self.tf = []

        for i, text in enumerate(corpus):
            tokens = self._tokenize(text)
            self.doc_len.append(len(tokens))
            f = {}
            for token in tokens:
                f[token] = f.get(token, 0) + 1
                nd[token] = nd.get(token, 0) + 1
            self.tf.append(f)

        self.doc_freqs = nd
        self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size > 0 else 0

        # 计算 IDF
        for token, freq in nd.items():
            self.idf[token] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1)

    def get_scores(self, query):
        """计算 query 与整个 corpus 的相关性分数列表。"""
        query_tokens = self._tokenize(query)
        scores = [0.0] * self.corpus_size
        
        if not query_tokens:
            return scores

        # 预先计算 query 权重
        query_weights = {}
        for token in query_tokens:
            idf = self.idf.get(token, 0)
            if idf > 0: 
                query_weights[token] = idf

        if not query_weights:
            return scores

        # 遍历文档计算分数
        for i in range(self.corpus_size):
            doc_len = self.doc_len[i]
            doc_tf = self.tf[i]

            for token, idf_weight in query_weights.items():
                freq = doc_tf.get(token, 0)
                if freq == 0:
                    continue
                
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                scores[i] += idf_weight * (numerator / denominator)

        return scores

    def search(self, query, top_k=5):
        """检索 top-k 结果。"""
        scores = self.get_scores(query)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        # 过滤掉分数为 0 的结果
        return [(i, scores[i]) for i in ranked if scores[i] > 0][:top_k]
