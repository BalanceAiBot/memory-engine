"""语义记忆引擎 — 测试套件"""
from __future__ import annotations

import os

# 必须在导入 torch 之前设置
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import tempfile
import unittest
from pathlib import Path

# 确保项目根在 path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import EMBEDDING_DIM
from src.chunker import Chunk, Chunker
from src.embedder import Embedder
from src.engine import MemoryEngine
from src.indexer import VectorIndex
from src.migrator import MigrationEntry, Migrator
from src.retriever import Retriever, SearchResult
from src.store import MetadataStore


class TestChunker(unittest.TestCase):
    """分块器测试"""

    def setUp(self):
        self.chunker = Chunker(max_chars=500, min_chars=10, overlap_chars=50)

    def test_short_text_no_split(self):
        """短文本不拆分"""
        text = "这是一条简短的记忆。"
        chunks = self.chunker.chunk(text)
        self.assertEqual(len(chunks), 1)
        self.assertIn("简短", chunks[0].text)

    def test_empty_text(self):
        """空文本返回空列表"""
        self.assertEqual(self.chunker.chunk(""), [])
        self.assertEqual(self.chunker.chunk("   "), [])

    def test_paragraph_split(self):
        """多段落文本分段"""
        text = "第一段内容，这是关于用户的信息。\n\n第二段内容，这是关于系统的信息。"
        chunks = self.chunker.chunk(text)
        self.assertGreaterEqual(len(chunks), 1)

    def test_long_text_split(self):
        """超长文本按句子边界拆分"""
        # 生成超过 max_chars 的文本
        text = "。".join([f"第{i}句测试内容" for i in range(100)]) + "。"
        chunks = self.chunker.chunk(text)
        self.assertGreater(len(chunks), 1)
        # 每块不超过限制
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text), self.chunker.max_chars + 50)

    def test_metadata_passthrough(self):
        """元数据透传"""
        text = "这是一段足够长的测试文本，用于验证元数据是否正确传递到分块结果中"
        chunks = self.chunker.chunk(text, {"category": "test"})
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].metadata.get("category"), "test")

    def test_chunk_fields(self):
        """Chunk 字段完整性"""
        text = "测试 chunk 字段"
        chunks = self.chunker.chunk(text)
        chunk = chunks[0]
        self.assertIsInstance(chunk, Chunk)
        self.assertTrue(len(chunk.text) > 0)
        self.assertGreaterEqual(chunk.index, 0)


class TestEmbedder(unittest.TestCase):
    """Embedding 模型测试"""

    def setUp(self):
        self.embedder = Embedder()

    def test_encode_single(self):
        """单条编码"""
        vec = self.embedder.encode_single("测试中文编码")
        self.assertEqual(vec.shape[0], EMBEDDING_DIM)
        # 归一化后模长接近 1
        import numpy as np
        norm = float(np.linalg.norm(vec))
        self.assertAlmostEqual(norm, 1.0, places=2)

    def test_encode_batch(self):
        """批量编码"""
        texts = ["第一条测试", "第二条测试", "第三条测试"]
        vectors = self.embedder.encode(texts)
        self.assertEqual(vectors.shape, (3, EMBEDDING_DIM))

    def test_semantic_similarity(self):
        """语义相似度：相关文本得分更高"""
        vec1 = self.embedder.encode_single("用户希望系统盈利")
        vec2 = self.embedder.encode_single("用户的目标是赚钱")
        vec3 = self.embedder.encode_single("今天天气很好")

        sim_similar = self.embedder.cosine_similarity(vec1, vec2)
        sim_dissimilar = self.embedder.cosine_similarity(vec1, vec3)

        # 相关文本相似度应高于不相关的
        self.assertGreater(sim_similar, sim_dissimilar)

    def test_repr(self):
        """字符串表示"""
        r = repr(self.embedder)
        self.assertIn("Embedder", r)
        self.assertIn("512", r)


class TestMetadataStore(unittest.TestCase):
    """SQLite 元数据存储测试"""

    def setUp(self):
        self.tmpfile = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmpfile.close()
        self.store = MetadataStore(self.tmpfile.name)

    def tearDown(self):
        os.unlink(self.tmpfile.name)

    def test_add_and_get(self):
        """添加和获取"""
        cid = self.store.add_chunk("测试文本", category="test", source="cli")
        chunk = self.store.get_chunk(cid)
        self.assertIsNotNone(chunk)
        self.assertEqual(chunk["text"], "测试文本")
        self.assertEqual(chunk["category"], "test")

    def test_batch_add(self):
        """批量添加"""
        items = [
            {"text": "文本一", "category": "cat1"},
            {"text": "文本二", "category": "cat2"},
        ]
        ids = self.store.add_chunks_batch(items)
        self.assertEqual(len(ids), 2)

    def test_get_by_category(self):
        """按分类查询"""
        self.store.add_chunk("安全相关", category="security")
        self.store.add_chunk("用户信息", category="user")
        results = self.store.get_chunks_by_category("security")
        self.assertEqual(len(results), 1)
        self.assertIn("安全", results[0]["text"])

    def test_stats(self):
        """统计信息"""
        self.store.add_chunk("A", category="cat1")
        self.store.add_chunk("B", category="cat1")
        self.store.add_chunk("C", category="cat2")
        stats = self.store.stats()
        self.assertEqual(stats["total_chunks"], 3)
        self.assertEqual(stats["categories"]["cat1"], 2)

    def test_delete(self):
        """删除"""
        cid = self.store.add_chunk("删除测试")
        self.assertTrue(self.store.delete_chunk(cid))
        self.assertIsNone(self.store.get_chunk(cid))

    def test_update(self):
        """更新"""
        cid = self.store.add_chunk("原始文本", category="old")
        self.store.update_chunk(cid, category="new")
        chunk = self.store.get_chunk(cid)
        self.assertEqual(chunk["category"], "new")


class TestVectorIndex(unittest.TestCase):
    """FAISS 向量索引测试"""

    def setUp(self):
        import numpy as np
        # 使用不存在的临时路径，避免 faiss 尝试加载空文件
        self.tmpdir = tempfile.mkdtemp()
        self.tmpfile = os.path.join(self.tmpdir, "test.index")
        self.index = VectorIndex(dim=EMBEDDING_DIM)  # 先不关联路径
        # 添加测试向量
        vecs = np.random.rand(5, EMBEDDING_DIM).astype(np.float32)
        # 归一化
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        self.index.add_vectors([1, 2, 3, 4, 5], vecs)

    def tearDown(self):
        import shutil
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    def test_add_and_size(self):
        """添加后索引大小"""
        self.assertEqual(self.index.size, 5)

    def test_search(self):
        """搜索"""
        import numpy as np
        query = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        query = query / np.linalg.norm(query)
        results = self.index.search(query, top_k=3)
        self.assertEqual(len(results), 3)
        # 结果是 (chunk_id, score) 元组
        self.assertIsInstance(results[0][0], int)
        self.assertIsInstance(results[0][1], float)

    def test_empty_search(self):
        """空索引搜索"""
        empty_index = VectorIndex(dim=EMBEDDING_DIM)
        import numpy as np
        results = empty_index.search(np.zeros(EMBEDDING_DIM, dtype=np.float32))
        self.assertEqual(results, [])

    def test_save_and_load(self):
        """持久化与加载"""
        self.index.save(self.tmpfile)
        new_index = VectorIndex(dim=EMBEDDING_DIM, index_path=self.tmpfile)
        self.assertEqual(new_index.size, 5)


class TestMigrator(unittest.TestCase):
    """迁移器测试"""

    def setUp(self):
        self.migrator = Migrator()

    def test_classify_identity(self):
        """身份分类"""
        cat = self.migrator._classify("关于我: 名字：小轩")
        self.assertEqual(cat, "identity")

    def test_classify_user(self):
        """用户画像分类"""
        cat = self.migrator._classify("关于用户: 工作方向：全栈开发")
        self.assertEqual(cat, "user_profile")

    def test_classify_security(self):
        """安全规则分类"""
        cat = self.migrator._classify("安全规则: 绝不外发本机私密信息")
        self.assertEqual(cat, "security_rule")

    def test_classify_dev_tool(self):
        """开发工具分类"""
        cat = self.migrator._classify("开发工具: Claude Code 2.1.88")
        self.assertEqual(cat, "dev_tool")

    def test_classify_learning(self):
        """学习笔记分类"""
        cat = self.migrator._classify("学习笔记：Day 1 核心知识点")
        self.assertEqual(cat, "learning_note")

    def test_classify_general(self):
        """无法识别的归为 general"""
        cat = self.migrator._classify("一条普通记忆")
        self.assertEqual(cat, "general")

    def test_parse_hermes_format(self):
        """解析 Heremes 格式"""
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8")
        tmp.write("关于用户: 名字是Balance\n§\n安全规则: 绝不外发信息\n§\n学习笔记：测试内容\n")
        tmp.close()

        entries = self.migrator.parse_hermes_memory(tmp.name)
        os.unlink(tmp.name)
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0].category, "user_profile")
        self.assertEqual(entries[1].category, "security_rule")


class TestMemoryEngine(unittest.TestCase):
    """MemoryEngine 集成测试"""

    @classmethod
    def setUpClass(cls):
        """创建临时数据库（只加载一次模型）"""
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.db_path = os.path.join(cls.tmpdir.name, "test.db")
        cls.index_path = os.path.join(cls.tmpdir.name, "faiss.index")
        cls.engine = MemoryEngine(
            db_path=cls.db_path,
            index_path=cls.index_path,
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_ingest_and_query(self):
        """入库和检索"""
        ids = self.engine.ingest("用户的工作方向是全栈开发和金融", category="user_profile", source="test")
        self.assertTrue(len(ids) > 0)

        results = self.engine.query("用户做什么工作？", top_k=3)
        self.assertTrue(len(results) > 0)
        # 验证返回结构
        self.assertIn("text", results[0])
        self.assertIn("score", results[0])
        self.assertIn("category", results[0])

    def test_ingest_batch(self):
        """批量入库"""
        items = [
            {"text": "安全规则：不泄露信息", "category": "security_rule", "source": "batch_test"},
            {"text": "开发工具：Claude Code", "category": "dev_tool", "source": "batch_test"},
        ]
        ids = self.engine.ingest_batch(items)
        self.assertEqual(len(ids), 2)

    def test_query_no_results(self):
        """无结果查询"""
        results = self.engine.query("zzz完全不相关的内容xyz", top_k=5)
        # 应该没有高相似度结果
        for r in results:
            self.assertLess(r["score"], 0.9)

    def test_stats(self):
        """统计"""
        stats = self.engine.stats()
        self.assertIn("total_chunks", stats)
        self.assertGreater(stats["total_chunks"], 0)
        self.assertIn("categories", stats)

    def test_list_by_category(self):
        """按类别列出"""
        items = self.engine.list_by_category("user_profile")
        self.assertTrue(len(items) > 0)
        self.assertIn("text", items[0])

    def test_delete(self):
        """删除"""
        ids = self.engine.ingest("临时记忆用于删除测试", category="test")
        cid = ids[0]
        self.assertTrue(self.engine.delete(cid))

    def test_long_text_chunking(self):
        """长文本自动分块"""
        # 用双换行分隔长段落，超过 CHUNK_MAX_CHARS 触发拆分
        long_text = "\n\n".join([f"这是关于用户的第{i}条重要信息：用户在金融领域有丰富经验，包括加密货币交易、美股投资、港股分析和A股策略。用户希望建立一个自动化的交易监控系统，能够实时分析市场数据并给出投资建议。这是一个长期项目，需要持续优化和改进。" for i in range(20)])
        ids = self.engine.ingest(long_text, category="user_profile", source="chunk_test")
        self.assertGreater(len(ids), 1)  # 应该被拆分成多块

    def test_query_interleave(self):
        """多跳检索"""
        results = self.engine.query_interleave("用户的目标", top_k=3, rounds=2)
        # 应该返回一些结果
        self.assertIsInstance(results, list)


class TestRetriever(unittest.TestCase):
    """检索器测试"""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.db_path = os.path.join(cls.tmpdir.name, "retriever_test.db")
        cls.index_path = os.path.join(cls.tmpdir.name, "retriever.index")
        cls.engine = MemoryEngine(
            db_path=cls.db_path,
            index_path=cls.index_path,
        )
        # 添加多条记忆
        cls.engine.ingest("用户名字叫Balance", category="user_profile", source="test")
        cls.engine.ingest("用户工作是全栈开发", category="user_profile", source="test")
        cls.engine.ingest("安全规则不外发信息", category="security_rule", source="test")

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_basic_search(self):
        """基本搜索"""
        results = self.engine.retriever.search("用户叫什么名字", top_k=3)
        self.assertTrue(len(results) > 0)
        # 最高分应该是关于名字的
        top_text = results[0].text
        self.assertIn("Balance", top_text)

    def test_search_by_category(self):
        """按类别搜索"""
        results = self.engine.retriever.search_by_category("信息不外发", "security_rule", top_k=3)
        self.assertTrue(len(results) > 0)
        for r in results:
            self.assertEqual(r.category, "security_rule")

    def test_search_result_fields(self):
        """搜索结果字段"""
        results = self.engine.retriever.search("开发", top_k=1)
        if results:
            r = results[0]
            self.assertIsInstance(r.to_dict(), dict)
            self.assertIn("chunk_id", r.to_dict())
            self.assertIn("score", r.to_dict())


if __name__ == "__main__":
    unittest.main(verbosity=2)
