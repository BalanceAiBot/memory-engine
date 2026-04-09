# 部署与运维指南

## 1. 服务启动与停止

Memory Engine 使用常驻 HTTP Server 模式提供极速检索（~5ms 延迟）。

### 1.1 检查服务状态
```bash
curl -s http://127.0.0.1:8089/health
# 返回: {"status": "ok"}
```

### 1.2 启动服务 (手动)
```bash
cd ~/Desktop/clawCoder/memory-engine
/opt/homebrew/bin/python3.11 server.py &
```

### 1.3 启动服务 (LaunchAgent - 推荐)
```bash
# 安装 LaunchAgent (开机自启)
cp ~/Desktop/clawCoder/memory-engine/ai.memory.engine.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/ai.memory.engine.plist

# 停止服务
launchctl unload ~/Library/LaunchAgents/ai.memory.engine.plist

# 重启服务
launchctl kickstart -k gui/$(id -u)/ai.memory.engine
```

### 1.4 查看服务日志
```bash
# 标准输出
tail -f /tmp/memory-engine.log

# 错误日志
tail -f /tmp/memory-engine-error.log
```

## 2. 数据备份

记忆数据存储在以下两个文件中：
- `~/.hermes/memories/memory.db` (SQLite 元数据)
- `~/.hermes/memories/faiss.index` (FAISS 向量索引)

### 2.1 手动备份
```bash
BACKUP_DIR="~/backups/memory-engine/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp ~/.hermes/memories/memory.db "$BACKUP_DIR/"
cp ~/.hermes/memories/faiss.index "$BACKUP_DIR/"
echo "✅ Backup saved to $BACKUP_DIR"
```

### 2.2 恢复数据
```bash
cp ~/backups/memory-engine/YYYYMMDD_HHMMSS/memory.db ~/.hermes/memories/
cp ~/backups/memory-engine/YYYYMMDD_HHMMSS/faiss.index ~/.hermes/memories/

# 重启服务以加载新数据
launchctl kickstart -k gui/$(id -u)/ai.memory.engine
```

## 3. 常见故障排查

### 3.1 服务无响应
```bash
# 检查进程
ps aux | grep server.py

# 强制重启
pkill -f server.py
sleep 2
/opt/homebrew/bin/python3.11 ~/Desktop/clawCoder/memory-engine/server.py &
```

### 3.2 索引不一致 (DB 条目数 ≠ FAISS 索引数)
```bash
cd ~/Desktop/clawCoder/memory-engine
/opt/homebrew/bin/python3.11 cli.py rebuild
```

### 3.3 模型加载失败
```bash
# 检查模型缓存
ls ~/.cache/huggingface/hub/ | grep bge

# 如果缓存损坏，删除后重新下载
rm -rf ~/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5
```

## 4. 性能监控

### 4.1 内存占用
```bash
ps -o pid,rss,command -p $(pgrep -f server.py)
# RSS 列即为内存占用 (KB)
# 正常范围: 400-500 MB
```

### 4.2 检索延迟测试
```bash
cd ~/Desktop/clawCoder/memory-engine
time curl -s "http://127.0.0.1:8089/search?q=test&k=1"
```

## 5. 分类体系说明

| 分类 | 用途 | 示例 |
|------|------|------|
| `identity` | AI 自身信息 | 名字、模型版本、上线时间 |
| `user_profile` | 用户画像 | 工作方向、时区、目标 |
| `security_rule` | 安全规则 | 隐私保护、权限控制 |
| `dev_tool` | 开发工具 | Claude Code/Codex/OpenCode 配置 |
| `project_memory` | 项目上下文 | BrowserWing/daily-report 架构 |
| `maintenance_log` | 维护记录 | 安全审计、版本更新 |
| `learning_note` | 学习笔记 | 课程笔记、技术总结 |
| `general` | 通用 | 未分类内容 |

## 6. 日常维护

### 6.1 定期策展 (推荐每周执行)
```bash
cd ~/Desktop/clawCoder/memory-engine
/opt/homebrew/bin/python3.11 integration.py curate
```

### 6.2 同步文件变更
```bash
/opt/homebrew/bin/python3.11 integration.py sync
```

### 6.3 清理过期记忆
```bash
# 列出所有记忆
/opt/homebrew/bin/python3.11 cli.py list

# 删除指定条目
/opt/homebrew/bin/python3.11 cli.py delete <chunk_id>
```
