# 数据库结构说明

## 1. SQLite 元数据 (`memory.db`)

### chunks 表
存储所有记忆条目的元数据。

| 字段 | 类型 | 说明 |
|------|------|------|
| `chunk_id` | INTEGER PRIMARY KEY | 自动递增 ID |
| `text` | TEXT | 记忆文本内容 |
| `category` | TEXT | 分类 (identity/user_profile/security_rule 等) |
| `source` | TEXT | 来源标识 (hermes_memory/cli/会话记录) |
| `metadata` | TEXT | JSON 格式扩展元数据 |
| `created_at` | TIMESTAMP DEFAULT CURRENT_TIMESTAMP | 创建时间 |
| `updated_at` | TIMESTAMP | 最后更新时间 |

### index_meta 表
存储索引配置元数据。

| 字段 | 说明 |
|------|------|
| `key` | 元数据键 (如 model_name, dim, created_at) |
| `value` | 元数据值 |
| `updated_at` | 更新时间 |

## 2. FAISS 索引 (`faiss.index`)

- **类型**: `IndexFlatIP` (内积相似度)
- **维度**: 512 (bge-small-zh-v1.5 模型输出)
- **归一化**: 向量已 L2 归一化，内积 = 余弦相似度
- **存储**: 内存映射文件，重启可加载

## 3. 数据文件位置

```
~/.hermes/memories/
├── memory.db          # SQLite 元数据
├── faiss.index        # FAISS 向量索引
├── MEMORY.md          # 精简版文本记忆 (44 行)
└── MEMORY.md.bak      # 原始备份 (326 行)
```
