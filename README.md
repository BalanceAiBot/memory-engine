# 🧠 语义记忆引擎 (Memory Engine)

中文优化的轻量级语义记忆引擎，基于 embedding + FAISS + SQLite。为 AI 助手提供长期语义记忆能力。

## 特性

- 🇨🇳 **中文优化** — 使用 BAAI/bge-small-zh-v1.5，~90MB 轻量模型
- 🖥️ **全 CPU 运行** — 零 CUDA 依赖，Apple Silicon 原生支持
- 🔍 **语义检索** — FAISS 向量索引，毫秒级相似度搜索
- 🧩 **智能分块** — 按段落/句子边界切分，不截断语义
- 🔗 **多跳检索** — Memory Interleave：检索→精炼→再检索
- 📦 **增量更新** — 新记忆追加入库，无需重建索引
- 💾 **持久化** — SQLite + FAISS 磁盘存储，重启可加载
- 🚚 **迁移工具** — 自动从现有 MEMORY.md 文件迁移

## 项目结构

```
memory-engine/
├── src/
│   ├── __init__.py          # 包入口
│   ├── engine.py            # MemoryEngine 主类（统一入口）
│   ├── embedder.py          # Embedding 模型封装
│   ├── chunker.py           # 智能分块器
│   ├── indexer.py           # FAISS 索引管理
│   ├── store.py             # SQLite 元数据存储
│   ├── retriever.py         # 检索器（含多跳推理）
│   └── migrator.py          # 从 MEMORY.md 迁移
├── config.py                # 全局配置
├── cli.py                   # 命令行工具
├── requirements.txt
├── README.md
└── tests/
    └── test_engine.py       # 完整测试套件
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 迁移现有记忆

```bash
python cli.py migrate
```

### 3. 检索记忆

```bash
python cli.py query "用户的工作方向是什么？" --top-k 5
```

### 4. 入库新记忆

```bash
python cli.py ingest "用户新增了一条重要规则" --category security_rule
```

## API 使用

### Python API

```python
from src.engine import MemoryEngine

# 初始化
engine = MemoryEngine(db_path="~/.hermes/memories/memory.db")

# 入库
engine.ingest("用户希望系统能盈利", category="user_goal", source="MEMORY.md")

# 批量入库
engine.ingest_batch([
    {"text": "安全规则：不外发信息", "category": "security_rule", "source": "cli"},
    {"text": "开发工具：Claude Code", "category": "dev_tool", "source": "cli"},
])

# 语义检索
results = engine.query("用户的目标是什么？", top_k=5)
for r in results:
    print(f"[{r['score']:.4f}] {r['text']}")

# 多跳检索（检索→精炼→再检索）
results = engine.query_interleave("最近有什么安全问题？", top_k=5, rounds=2)

# 限定类别检索
results = engine.query_by_category("工作", "dev_tool", top_k=3)

# 统计
stats = engine.stats()
print(stats)
# {"total_chunks": 120, "categories": {"user_profile": 7, ...}, ...}

# 按类别列出
items = engine.list_by_category("security_rule")

# 删除
engine.delete(chunk_id=1)

# 重建索引
engine.rebuild_index()

# 迁移
engine.migrate_from_hermes_memory("~/.hermes/memories/MEMORY.md")
engine.migrate_from_openclaw_memory("~/.openclaw/workspace/MEMORY.md")
engine.migrate_all()  # 迁移所有可用源
```

## CLI 命令

| 命令 | 说明 | 示例 |
|------|------|------|
| `ingest` | 入库记忆 | `python cli.py ingest "记忆内容" -c user_goal` |
| `query` | 语义检索 | `python cli.py query "用户目标" -k 5` |
| `query -i` | 多跳检索 | `python cli.py query "安全问题" -i` |
| `migrate` | 迁移记忆 | `python cli.py migrate` |
| `stats` | 查看统计 | `python cli.py stats` |
| `list` | 列出记忆 | `python cli.py list -c security` |
| `delete` | 删除记忆 | `python cli.py delete 1` |
| `rebuild` | 重建索引 | `python cli.py rebuild` |

## 分类体系

| 分类 | 说明 | 示例 |
|------|------|------|
| `identity` | 身份信息 | 名字、模型版本、上线时间 |
| `user_profile` | 用户画像 | 工作方向、时区、目标 |
| `security_rule` | 安全规则 | 不外发信息、权限控制 |
| `dev_tool` | 开发工具 | Claude Code、Codex 等 |
| `project_memory` | 项目记忆 | BrowserWing、daily-report |
| `maintenance_log` | 维护记录 | 版本更新、修复日志 |
| `learning_note` | 学习笔记 | OpenClaw 课程笔记 |
| `general` | 通用 | 未分类内容 |

## 架构说明

```
┌─────────────────────────────────────────────┐
│                MemoryEngine                  │
├─────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌───────────┐ │
│  │ Embedder │  │ Chunker  │  │  Migrator │ │
│  │ (bge-zh) │  │(智能分块)│  │(格式解析) │ │
│  └────┬─────┘  └────┬─────┘  └─────┬─────┘ │
│       │              │              │        │
│       ▼              ▼              ▼        │
│  ┌──────────────────────────────────────┐   │
│  │            VectorIndex               │   │
│  │          (FAISS CPU)                 │   │
│  └──────────────────────────────────────┘   │
│                     │                        │
│                     ▼                        │
│  ┌──────────────────────────────────────┐   │
│  │            Retriever                 │   │
│  │      (语义搜索 + 多跳推理)             │   │
│  └──────────────────────────────────────┘   │
│                     │                        │
│                     ▼                        │
│  ┌──────────────────────────────────────┐   │
│  │          MetadataStore               │   │
│  │           (SQLite)                   │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

### 工作流程

1. **入库**: 文本 → 智能分块 → Embedding → FAISS 索引 + SQLite 元数据
2. **检索**: 查询 → Embedding → FAISS Top-K 搜索 → SQLite 回填元数据
3. **多跳**: 首轮检索 → 用 top-1 结果精炼 query → 再检索 → 融合去重

## 技术选型

| 组件 | 选择 | 理由 |
|------|------|------|
| Embedding | BAAI/bge-small-zh-v1.5 | 中文优化、轻量(~90MB)、512维 |
| 向量索引 | FAISS (IndexFlatIP) | 快速、CPU 友好、持久化 |
| 元数据 | SQLite | Python 内置、零依赖、WAL 模式 |
| 分块 | 自定义规则引擎 | 按段落/句子边界，不截断语义 |
| CLI | Click + Rich | 美观、类型安全 |

## 运行测试

```bash
cd ~/Desktop/clawCoder/memory-engine
python -m pytest tests/test_engine.py -v
# 或
python tests/test_engine.py
```

## 注意事项

- 首次运行会自动下载 embedding 模型（~90MB），缓存到 `~/.cache/huggingface/`
- 数据库和索引文件默认存储在 `~/.hermes/memories/`
- 迁移不会删除原始 MEMORY.md 文件
- 增量入库时只需调用 `ingest()`，索引自动追加

## 集成到 Hermes Agent

本项目已与 Hermes Agent 的 `memory` 工具无缝集成。

### 1. 修改记忆工具代码
**目标文件**: `~/.hermes/hermes-agent/tools/memory_tool.py`

#### A. 添加语义搜索函数
在文件顶部（`MemoryStore` 类定义之后）添加 `_semantic_search` 函数：
```python
def _semantic_search(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Use Memory Engine for semantic search."""
    import subprocess, os, re, json
    
    engine_dir = os.path.expanduser("~/Desktop/clawCoder/memory-engine")
    integration = os.path.join(engine_dir, "integration.py")
    
    if not os.path.exists(integration):
        return {"success": False, "error": "Memory Engine not found."}

    try:
        # Call the integration script using Python 3.11
        result = subprocess.run(
            ["/opt/homebrew/bin/python3.11", integration, "search", query, "--top-k", str(top_k)],
            capture_output=True, text=True, timeout=30
        )
        
        # Parse output
        lines = result.stdout.strip().split('\n')
        results = []
        current_entry = None
        
        for line in lines:
            # Match header: [1] score=1.112 [maintenance_log]
            match = re.search(r'\[(\d+)\]\s+score=([\d.]+)\s+\[([^\]]+)\]', line)
            if match:
                if current_entry: results.append(current_entry)
                current_entry = {
                    "rank": int(match.group(1)),
                    "score": float(match.group(2)),
                    "category": match.group(3),
                    "text": ""
                }
            elif current_entry and line.strip() and not line.strip().startswith('['):
                current_entry["text"] += line.strip() + " "
        
        if current_entry: results.append(current_entry)
        
        return {"success": True, "query": query, "results": results}
        
    except Exception as e:
        return {"success": False, "error": str(e)}
```

#### B. 扩展 `memory_tool` 函数
找到 `memory_tool` 函数，在 `remove` 动作之后添加 `search` 处理：
```python
    elif action == "search":
        if not content:
            return tool_error("content (query) is required for 'search' action.", success=False)
        result = _semantic_search(content, top_k=5)
        return json.dumps(result, ensure_ascii=False)
```

#### C. 更新 Schema
修改 `MEMORY_SCHEMA` 定义，将 `action` 的 `enum` 从 `["add", "replace", "remove"]` 改为 `["add", "replace", "remove", "search"]`。

### 2. 更新系统指令
在 `AGENTS.md` 和 `SOUL.md` 中添加语义检索的强制指令：
> "When the user asks about past events... USE THE TOOL: `memory(action='search', content='query')`"

### 3. 生效方式
- **重启**: 修改代码后必须重启 Hermes Agent (`launchctl kickstart -k ...`)。
- **验证**: 询问 Agent "上次审计修了什么"，应能触发 `memory` 工具调用。

### 量化对比

| 指标 | 旧系统（纯文本） | 新系统（Memory Engine） | 变化 |
|------|-----------------|------------------------|------|
| MEMORY.md 行数 | 326 行 | 44 行 | ↓ **87%** |
| 文件大小 | 15.4 KB | 2.1 KB | ↓ **86%** |
| 每次会话 token | ~2521 | ~637 | ↓ **75%** |
| 可检索条目 | 1（全文） | 64 | ↑ **64x** |
| 检索精度 | 关键词匹配 | 语义理解（0.82 平均相关度） | 质的飞跃 |

### 检索延迟（Apple M4）

| 场景 | 冷启动 | 热启动 |
|------|--------|--------|
| 首次查询 | ~1667ms（模型加载） | — |
| 后续查询 | — | ~3ms |
| 平均相关度 | — | 0.817 |

### 架构差异

**旧系统**：纯文本 `§` 分隔 → 每次全文加载 → 关键词匹配

**新系统**：文本精简版（核心记忆）+ 向量索引（扩展记忆）→ 语义检索按需加载

### 为什么双写（文本 + 向量）

| 维度 | 纯文本 | 纯向量 | 双写 |
|------|--------|--------|------|
| 开机自检 | ✅ 0 延迟 | ❌ 需要查询 | ✅ 0 延迟 |
| 语义检索 | ❌ 不支持 | ✅ 支持 | ✅ 支持 |
| 可靠性 | ✅ 纯文本 | ❌ 依赖模型 | ✅ 文本兜底 |
| 维护成本 | 低 | 中 | 中 |

双写是可靠性和功能性的最佳平衡：文本层保证系统启动就有基础记忆，向量层提供按需扩展的深度检索。

## 许可证

MIT
