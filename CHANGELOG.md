# 更新日志

## [1.2.0] - 2026-04-09

### 性能优化
- 新增 HTTP Server 模式 (`server.py`)，检索延迟从 ~1.5s 降至 **~5ms**
- 模型常驻内存，避免每次请求重新加载
- macOS LaunchAgent 开机自启支持

### 集成改进
- `memory_tool.py` 改用 HTTP 调用替代 subprocess，速度提升 **300 倍**
- 添加优雅降级：Server 离线时返回明确错误提示

### 文档
- 新增 `DEPLOYMENT.md`：服务管理、故障排查、备份恢复
- 更新集成指南为 HTTP 模式代码示例

## [1.1.0] - 2026-04-09

### 新功能
- 新增 `search` 动作到 `memory_tool.py`
- 类别加权检索算法 (retriever.py)
- Heartbeat 自动化维护脚本

### 数据清理
- 去重：80 条 → 64 条
- MEMORY.md 瘦身：326 行 → 44 行 (↓87%)
- 分类重平衡

## [1.0.0] - 2026-04-09

### 初始版本
- 核心引擎：Embedding + FAISS + SQLite
- 智能分块器
- MEMORY.md 迁移工具
- CLI 命令行工具
- 多跳检索支持
