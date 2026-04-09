#!/usr/bin/env python3
"""
记忆系统集成层 — 将 Memory Engine 接入现有记忆工作流

功能：
1. memory_tool_ingest(): 记忆入库 → 同时写入 MEMORY.md + Memory Engine
2. semantic_search(): 语义检索 → 替代关键词搜索
3. auto_curate(): 自动策展 → 维护精简版 MEMORY.md
4. context_build(): 上下文构建 → 按需注入相关记忆到上下文
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from src.engine import MemoryEngine

# 路径配置
HERMES_MEMORY = os.path.expanduser("~/.hermes/memories/MEMORY.md")
HERMES_USER = os.path.expanduser("~/.hermes/memories/USER.md")
OPENCLAW_MEMORY = os.path.expanduser("~/.openclaw/workspace/MEMORY.md")
MEMORY_DB = os.path.expanduser("~/.hermes/memories/memory.db")
MEMORY_INDEX = os.path.expanduser("~/.hermes/memories/faiss.index")


def get_engine() -> MemoryEngine:
    """获取或创建 Memory Engine 实例"""
    return MemoryEngine(db_path=MEMORY_DB, index_path=MEMORY_INDEX)


def memory_tool_ingest(content: str, category: str = "general",
                     source: str = "memory_tool", target: str = "both"):
    """
    替代原有 memory tool 的写入行为
    
    Args:
        content: 记忆内容
        category: 分类 (identity/user_profile/security_rule/dev_tool/
                   project_memory/maintenance_log/learning_note/general)
        source: 来源标识
        target: "both" | "engine_only" | "file_only"
    
    Returns:
        入库结果
    """
    result = {"ingested": [], "errors": []}
    
    if target in ("both", "engine_only"):
        engine = get_engine()
        chunks = engine.ingest(content, category=category, source=source)
        result["ingested"].extend(chunks)
    
    if target in ("both", "file_only"):
        # 追加到 Heremes MEMORY.md (§ 分隔格式)
        entry = f"{content}\n§\n"
        with open(HERMES_MEMORY, "a", encoding="utf-8") as f:
            f.write(entry)
        result["file_updated"] = HERMES_MEMORY
    
    return result


def semantic_search(query: str, top_k: int = 5,
                    category: str = None,
                    interleave: int = 1,
                    weighted: bool = True) -> list:
    """
    语义检索记忆 — 替代关键词搜索
    
    Args:
        query: 检索问题
        top_k: 返回结果数
        category: 按类别过滤 (可选)
        interleave: 多跳轮数 (1=单轮, 2=多跳)
        weighted: 是否使用类别加权检索（默认开启，改善模糊查询质量）
    
    Returns:
        检索结果列表 [{"text", "score", "category", "source", "id"}, ...]
    """
    engine = get_engine()
    
    if category:
        results = engine.query_by_category(query, category, top_k=top_k)
    elif weighted and interleave <= 1:
        # 使用类别加权检索
        results = engine.query_weighted(query, top_k=top_k)
    elif interleave > 1:
        results = engine.query_interleave(query, top_k=top_k, rounds=interleave)
    else:
        results = engine.query(query, top_k=top_k)
    
    # 格式化为精简输出
    formatted = []
    for r in results:
        formatted.append({
            "id": r.get("id", ""),
            "text": r["text"],
            "score": round(r["score"], 4),
            "category": r.get("category", ""),
            "source": r.get("source", ""),
        })
    
    return formatted


def auto_curate(max_entries: int = 25):
    """
    自动策展 MEMORY.md — 保留最重要的条目，其余归档到 Memory Engine
    
    策略：
    1. 读取当前 MEMORY.md
    2. 按重要度排序（身份/安全规则 > 用户画像 > 工具 > 项目 > 日志）
    3. 保留 Top N 条目写入 MEMORY.md
    4. 全部条目确保已在 Memory Engine 中
    
    Returns:
        策展结果
    """
    priority_order = {
        "identity": 1,
        "security_rule": 2,
        "user_profile": 3,
        "dev_tool": 4,
        "project_memory": 5,
        "learning_note": 6,
        "maintenance_log": 7,
        "general": 8,
    }
    
    engine = get_engine()
    stats = engine.stats()
    
    # 获取所有条目并按类别优先级排序
    all_entries = engine.store.get_all_chunks()
    all_entries.sort(key=lambda x: priority_order.get(x.get("category", "general"), 9))
    
    # 前 N 条写入 MEMORY.md
    top_entries = all_entries[:max_entries]
    
    # 构建精简版 MEMORY.md
    lines = ["# MEMORY.md - 长期记忆（精简版）\n",
             f"# 最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
             f"# 总记忆条目: {stats['total_chunks']} | 此处保留: {len(top_entries)}\n",
             f"# 完整语义检索: ~/Desktop/clawCoder/memory-engine\n\n"]
    
    current_category = None
    for entry in top_entries:
        cat = entry.get("category", "general")
        if cat != current_category:
            lines.append(f"\n## {cat}\n")
            current_category = cat
        lines.append(f"- {entry['text']}\n")
    
    lines.append(f"\n---\n*以下 {len(all_entries) - max_entries} 条记忆已归档至 Memory Engine，可通过语义检索访问*\n")
    
    # 写入
    with open(HERMES_MEMORY, "w", encoding="utf-8") as f:
        f.writelines(lines)
    
    return {
        "total_entries": len(all_entries),
        "kept_in_memory_md": len(top_entries),
        "archived_to_engine": len(all_entries) - max_entries,
        "file": HERMES_MEMORY,
    }


def context_build(query: str, max_tokens: int = 2000) -> str:
    """
    构建上下文注入 — 根据当前对话主题，检索并组装相关记忆
    
    用于在需要时动态注入记忆到上下文，替代全文加载 MEMORY.md
    
    Args:
        query: 当前对话主题/问题
        max_tokens: 最大 token 数（约 4 chars = 1 token）
    
    Returns:
        格式化后的记忆上下文字符串
    """
    max_chars = max_tokens * 4  # 粗略估算
    
    results = semantic_search(query, top_k=8, interleave=1)
    
    if not results:
        return ""
    
    # 组装上下文
    lines = [
        "=== 相关记忆检索 ===",
        f"查询: {query}",
        f"检索到 {len(results)} 条相关记忆:",
        ""
    ]
    
    total_chars = sum(len(l) for l in lines)
    for i, r in enumerate(results, 1):
        entry = f"[{i}] [{r['category']}] {r['text']}"
        if total_chars + len(entry) > max_chars:
            lines.append(f"... (剩余 {len(results) - i + 1} 条，超出 token 限制)")
            break
        lines.append(entry)
        lines.append("")
        total_chars += len(entry) + 1
    
    return "\n".join(lines)


def sync_from_files():
    """
    检查文件变更并同步到 Memory Engine
    使用 engine.migrate() 方法完成
    """
    engine = get_engine()
    # 使用已有的迁移方法
    result_h = engine.migrate_from_hermes_memory(HERMES_MEMORY)
    result_o = engine.migrate_from_openclaw_memory(OPENCLAW_MEMORY)
    new_count = result_h.get("new_chunks", 0) + result_o.get("new_chunks", 0)
    return {"new_entries": new_count}


def quick_stats() -> dict:
    """快速统计"""
    engine = get_engine()
    return engine.stats()


# ===== CLI 入口 =====
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="记忆系统集成工具")
    sub = parser.add_subparsers(dest="command")
    
    # ingest
    p_ingest = sub.add_parser("ingest", help="入库新记忆")
    p_ingest.add_argument("content", help="记忆内容")
    p_ingest.add_argument("--category", default="general")
    p_ingest.add_argument("--source", default="cli")
    
    # search
    p_search = sub.add_parser("search", help="语义检索")
    p_search.add_argument("query", help="检索问题")
    p_search.add_argument("--top-k", type=int, default=5)
    p_search.add_argument("--category", default=None)
    p_search.add_argument("--interleave", type=int, default=1)
    p_search.add_argument("--weighted", action="store_true", default=True,
                          help="使用类别加权检索（默认开启）")
    p_search.add_argument("--no-weighted", action="store_true",
                          help="关闭类别加权检索")
    
    # curate
    sub.add_parser("curate", help="自动策展 MEMORY.md")
    
    # context
    p_ctx = sub.add_parser("context", help="构建上下文注入")
    p_ctx.add_argument("query", help="当前话题")
    p_ctx.add_argument("--max-tokens", type=int, default=2000)
    
    # sync
    sub.add_parser("sync", help="从文件同步到引擎")
    
    # stats
    sub.add_parser("stats", help="快速统计")
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        result = memory_tool_ingest(args.content, args.category, args.source)
        print(f"✅ 入库 {len(result['ingested'])} 条 chunk")
    
    elif args.command == "search":
        weighted = not args.no_weighted
        results = semantic_search(
            args.query, args.top_k, args.category, args.interleave,
            weighted=weighted,
        )
        print(f"检索: {args.query}")
        print(f"加权检索: {'开启' if weighted else '关闭'}\n")
        for i, r in enumerate(results, 1):
            print(f"  [{i}] score={r['score']:.3f} [{r['category']}]")
            print(f"      {r['text'][:120]}...")
            print()
    
    elif args.command == "curate":
        result = auto_curate()
        print(f"📋 策展完成")
        print(f"   总条目: {result['total_entries']}")
        print(f"   保留: {result['kept_in_memory_md']}")
        print(f"   归档: {result['archived_to_engine']}")
    
    elif args.command == "context":
        ctx = context_build(args.query, args.max_tokens)
        print(ctx)
    
    elif args.command == "sync":
        result = sync_from_files()
        print(f"🔄 同步完成，新增 {result['new_entries']} 条")
    
    elif args.command == "stats":
        stats = quick_stats()
        print(f"📊 记忆引擎: {stats['total_chunks']} 条 | "
              f"{len(stats.get('categories', {}))} 个分类")
        for cat, count in sorted(stats.get("categories", {}).items(),
                                  key=lambda x: -x[1]):
            print(f"   {cat}: {count}")
    
    else:
        parser.print_help()
