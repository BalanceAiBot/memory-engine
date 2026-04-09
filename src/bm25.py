"""轻量级 BM25 检索实现 (纯 Python，零依赖)"""
import math
import re
import os

# 尝试导入 jieba，如果不存在则使用备用分词
try:
    import jieba
    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False
    # print("⚠️ jieba not found, using fallback regex tokenizer.")

class SimpleBM25:
    """
    一个极简的 BM25 实现，用于混合检索。
    不需要 jieba 或 rank_bm25，使用字符级 N-gram 分词。
    """
    def __init__(self, corpus, k1=1.5, b=0.75):
        """
        初始化 BM25 索引。
        :param corpus: 字符串列表，每个字符串代表一个文档。
        :param k1: BM25 超参数，控制词频饱和度。
        :param b: BM25 超参数，控制文档长度归一化。
        """
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lengths = []
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []

        self._initialize(corpus)

    def _tokenize(self, text):
        """分词器：中文至少 2 字一组，英文按词。"""
        if HAS_JIEBA:
            return list(jieba.cut(text.lower()))
        
        # 备用方案：
        # 1. 提取连续的英文/数字
        en_tokens = re.findall(r'[a-z0-9]+', text.lower())
        # 2. 提取连续的中文字符串 (至少 2 个字，过滤掉单字噪音)
        cn_tokens = re.findall(r'[\u4e00-\u9fa5]{2,}', text)
        
        # 对于中文字符串，为了召回率，也可以拆分 bigram (2-gram)
        # 比如 "安全问题" -> ["安全", "全问", "问题"]
        # 但这会增加索引大小。暂时先用词组匹配。
        
        return en_tokens + cn_tokens

    def _initialize(self, corpus):
        """构建索引。"""
        nd = {}  # 词 -> 包含该词的文档数
        tf = []  # 词 -> 该词在文档中的词频

        for i, text in enumerate(corpus):
            tokens = self._tokenize(text)
            f = {}
            self.doc_len.append(len(tokens))

            for token in tokens:
                f[token] = f.get(token, 0) + 1
                nd[token] = nd.get(token, 0) + 1

            tf.append(f)

        self.doc_freqs = nd
        self.avgdl = sum(self.doc_len) / self.corpus_size

        # 计算 IDF
        for token, freq in nd.items():
            # BM25 IDF 公式: log((N - n + 0.5) / (n + 0.5) + 1)
            self.idf[token] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1)

    def get_score(self, query, index):
        """计算 query 与 index 文档的相关性分数。"""
        score = 0.0
        tokens = self._tokenize(query)
        
        if not tokens:
            return 0.0

        doc_len = self.doc_len[index]
        tf = self.doc_freqs  # 这里实际上应该是每篇文档的词频，这里简化处理
        # 修正：我们需要每篇文档的词频，而不仅仅是全局 df
        # 为了节省内存，我们在初始化时没存全量 TF 矩阵，这里重新遍历太慢。
        # 优化方案：初始化时存好 TF 矩阵
        pass 
    
    def _get_tf(self, text_tokens):
        """计算单个文档的词频。"""
        f = {}
        for token in text_tokens:
            f[token] = f.get(token, 0) + 1
        return f

    def get_scores(self, query, corpus_tokens):
        """计算 query 与整个 corpus 的相关性分数列表。"""
        query_tokens = self._tokenize(query)
        scores = []
        
        if not query_tokens:
            return [0.0] * self.corpus_size

        # 预先计算 query 中每个词的 IDF 权重
        query_weights = {}
        for token in query_tokens:
            idf = self.idf.get(token, 0)
            if idf > 0: 
                query_weights[token] = idf

        # DEBUG: 打印查询权重
        # print(f"[BM25 DEBUG] Query: {query}, Weights: {query_weights}", file=sys.stderr)

        if not query_weights:
            return [0.0] * self.corpus_size

        for i, doc_tokens in enumerate(corpus_tokens):
            score = 0.0
            doc_len = len(doc_tokens)
            doc_tf = self._get_tf(doc_tokens)

            # DEBUG: 记录匹配到的词
            matched = []

            for token, idf_weight in query_weights.items():
                freq = doc_tf.get(token, 0)
                if freq == 0:
                    continue
                
                matched.append(token)
                
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += idf_weight * (numerator / denominator)
            
            if matched: # 只要匹配到就打印
                print(f"[BM25 HIT] Doc {i} Score={score:.3f} Tokens={matched}", file=sys.stderr)
            
            scores.append(score)

        return scores

    def search(self, query, corpus_tokens, top_k=5):
        """
        检索 top-k 结果。
        :return: list of (index, score)
        """
        scores = self.get_scores(query, corpus_tokens)
        # 排序
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
