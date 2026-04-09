"""全局配置 — 语义记忆引擎"""
import os
from pathlib import Path

# ── 模型 ──
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"  # 中文优化，~90MB
EMBEDDING_DIM = 512

# ── 分块 ──
CHUNK_MIN_CHARS = 30          # 最小块字符数（过滤噪音）
CHUNK_MAX_CHARS = 1500        # 最大块字符数（约 400-600 tokens）
CHUNK_OVERLAP_CHARS = 100     # 块重叠字符数

# ── 检索 ──
DEFAULT_TOP_K = 5
MIN_SIMILARITY = 0.3          # 最低相似度阈值
INTERLEAVE_ROUNDS = 2         # 多跳默认轮数

# ── 默认路径 ──
DEFAULT_DB_DIR = os.path.expanduser("~/.hermes/memories")
DEFAULT_DB_PATH = os.path.join(DEFAULT_DB_DIR, "memory.db")
HERMES_MEMORY = os.path.expanduser("~/.hermes/memories/MEMORY.md")
OPENCLAW_MEMORY = os.path.expanduser("~/.openclaw/workspace/MEMORY.md")

# ── 分类标签 ──
CATEGORIES = [
    "identity",          # 身份信息
    "user_profile",      # 用户画像
    "security_rule",     # 安全规则
    "dev_tool",          # 开发工具
    "project_memory",    # 项目记忆
    "maintenance_log",   # 维护记录
    "learning_note",     # 学习笔记
    "general",           # 通用
]

# ── FAISS ──
FAISS_INDEX_TYPE = "IndexFlatIP"  # 内积（embedding 已归一化，等价于余弦）
