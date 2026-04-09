# 🧠 Memory Engine (语义记忆引擎)

**Status:** Production Ready ✅  
**Version:** 2.0 (Hybrid Search + Time Decay)  
**Platform:** Apple Silicon (M-Series) Native

A high-performance semantic memory system for AI agents. Combines **Vector Search**, **BM25 Keyword Search**, and **Time Decay** to provide precise, context-aware memory retrieval.

## ✨ 核心特性 (Features)

### 1. 混合检索 (Hybrid Search - RRF)
结合 **向量语义 (bge-base-zh)** 和 **BM25 关键词** 的优势。
- **语义理解**: 能听懂"上次修了啥" (映射到"修复/审计记录")。
- **关键词精准匹配**: 搜端口号 (8080)、版本号或具体报错代码时，精准度 100%。

### 2. 智能查询扩展 (Query Expansion)
内置口语-书面语映射，AI Agent 无需理解黑话也能搜到结果。
- "挂了/崩了" ➡️ "崩溃/错误/失败"
- "慢了/卡" ➡️ "延迟/性能"
- "上次" ➡️ "最近" (触发时间衰减机制)

### 3. 时间衰减策略 (Smart Time Decay)
根据查询意图自动调整"新鲜度"权重：
- **故障排查类查询** (如 "报错", "修了啥")：**强衰减**。优先返回最近 7 天的日志，旧日志降权。
- **事实架构类查询** (如 "架构", "工具")：**弱衰减**。确保历史文档不被过滤。

### 4. 类别优先权 (Category Boost)
针对技术类查询，自动提升 `maintenance_log` (维护日志) 和 `dev_tool` (开发工具) 的权重，过滤掉 `user_profile` (用户画像) 等无关噪音。

### 5. 零依赖运行 (Zero-Dependency Core)
核心 BM25 引擎完全手写 (纯 Python)，不依赖 `jieba` 或外部库，确保在网络波动或环境受限时依然稳定运行。

---

## 🚀 架构模式 (Architecture)

Memory Engine 采用 **C/S 架构** (Client/Server) 以消除模型加载延迟：

```
[Hermes Agent / Client]  <--HTTP (Port 54321)-->  [Memory Engine Server]
      |                                                  |
      |-- memory(action="search")                        |-- BAAI/bge-base-zh (Embedding)
      |                                                  |-- Custom BM25 (Keywords)
      |                                                  |-- RRF Fuser + Time Decay
```

---

## 🛠️ 部署与运行 (Deployment)

### 1. 启动服务端 (Server)
Memory Engine 需要作为后台服务常驻运行，以保持模型在内存中（实现毫秒级响应）。

```bash
cd ~/Desktop/clawCoder/memory-engine

# 推荐：后台启动 (使用 nohup)
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 nohup /opt/homebrew/bin/python3.11 server.py > /tmp/me.log 2>&1 &

# 检查健康状态
curl http://127.0.0.1:54321/health
# 返回: {"status": "ok"}
```

### 2. 客户端集成 (Integration)
Agent 端通过 HTTP 请求调用 API。

```python
import urllib.request
import urllib.parse

def search_memory(query: str, top_k: int = 3):
    url = f"http://127.0.0.1:54321/search?q={urllib.parse.quote(query)}&k={top_k}"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read())
            return data.get('results', [])
    except Exception as e:
        return None
```

---

## 📂 项目结构

```text
memory-engine/
├── server.py            # HTTP 服务入口 (常驻进程)
├── config.py            # 配置中心 (模型路径, 维度)
├── src/
│   ├── engine.py        # MemoryEngine 主控制器
│   ├── embedder.py      # 模型封装 (支持离线模式)
│   ├── retriever.py     # 检索器 (混合算法, 时间衰减, 类别加权)
│   ├── bm25.py          # 零依赖 BM25 实现
│   └── ...
├── scripts/
│   └── clean_memory.py  # 数据清洗工具
└── tests/               # 测试用例
```

---

## ⚙️ 高级配置

### 停用词表 (Stop Words)
在 `src/bm25.py` 中维护。过滤掉"关于"、"系统"、"用户"等高频无意义词，提高技术词汇的命中率。

### 查询扩展映射
在 `src/retriever.py` 的 `_expand_query` 方法中维护。可根据实际使用习惯添加新的口语映射。

---

## 📝 数据备份

记忆数据存储在 `~/.hermes/memories/`。
- `memory.db`: SQLite 数据库 (元数据 + 文本)
- `faiss.index`: 向量索引文件

建议定期备份这两个文件。
