#!/usr/bin/env python3
"""
Heartbeat Memory 集成脚本

功能：
1. 自动检测 MEMORY.md/USER.md 变更并同步
2. 自动去重（检测重复/包含关系）
3. 定期统计和健康检查
4. 短条目自动合并提醒
5. 分类重平衡提醒

用法：
  python3 heartbeat_memory.py              # 运行一次完整检查
  python3 heartbeat_memory.py --sync-only  # 仅同步
  python3 heartbeat_memory.py --dedup      # 仅去重
  python3 heartbeat_memory.py --stats      # 仅统计

可被 cron 或 heartbeat 调用：
  */30 * * * * /opt/homebrew/bin/python3.11 ~/Desktop/clawCoder/memory-engine/heartbeat_memory.py --sync-only
"""

import sys
import os
import json
import hashlib
import argparse
from datetime import datetime
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))

from src.engine import MemoryEngine
from config import DEFAULT_DB_PATH, DEFAULT_TOP_K

# 路径配置
MEMORY_DB = os.path.expanduser("~/.hermes/memories/memory.db")
MEMORY_INDEX = os.path.expanduser("~/.hermes/memories/faiss.index")
HERMES_MEMORY = os.path.expanduser("~/.hermes/memories/MEMORY.md")
HERMES_USER = os.path.expanduser("~/.hermes/memories/USER.md")
STATE_FILE = os.path.expanduser("~/.hermes/memories/.heartbeat_state.json")


def get_engine() -> MemoryEngine:
    """获取 Memory Engine 实例。"""
    return MemoryEngine(db_path=MEMORY_DB, index_path=MEMORY_INDEX)


def file_hash(path: str) -> str:
    """计算文件 MD5 哈希。"""
    p = Path(path)
    if not p.exists():
        return ""
    return hashlib.md5(p.read_bytes()).hexdigest()


def load_state() -> dict:
    """加载上次状态。"""
    p = Path(STATE_FILE)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"last_hashes": {}, "last_run": None, "last_stats": {}}


def save_state(state: dict):
    """保存状态。"""
    Path(STATE_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(STATE_FILE).write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def sync_files(engine: MemoryEngine) -> dict:
    """检测文件变更并同步到 Memory Engine。"""
    result = {"new": 0, "updated": 0, "errors": []}

    files_to_check = {
        "hermes_memory": HERMES_MEMORY,
        "hermes_user": HERMES_USER,
    }

    state = load_state()
    current_hashes = {}

    for name, path in files_to_check.items():
        h = file_hash(path)
        current_hashes[name] = h
        last_hash = state.get("last_hashes", {}).get(name, "")

        if h and h != last_hash:
            print(f"  检测到 {name} 变更，同步中...")
            try:
                if name == "hermes_memory":
                    stats = engine.migrate_from_hermes_memory(path)
                else:
                    content = Path(path).read_text(encoding="utf-8")
                    engine.ingest(content[:500], category="user_profile", source="heartbeat_sync")
                    stats = {"ingested": 1}
                result["new"] += stats.get("ingested", 0)
            except Exception as e:
                result["errors"].append(f"{name}: {e}")

    state["last_hashes"] = current_hashes
    state["last_run"] = datetime.now().isoformat()
    save_state(state)

    return result


def auto_dedup(engine: MemoryEngine) -> dict:
    """检测并报告重复条目。"""
    all_chunks = engine.store.get_all_chunks()
    result = {"duplicates": [], "containments": [], "short_entries": []}

    # 完全重复检测
    text_counts = Counter(c["text"] for c in all_chunks)
    for text, count in text_counts.items():
        if count > 1:
            ids = [c["chunk_id"] for c in all_chunks if c["text"] == text]
            result["duplicates"].append({"text": text[:80], "count": count, "ids": ids})

    # 包含关系检测
    texts = [(c["chunk_id"], c["text"]) for c in all_chunks if len(c["text"]) >= 10]
    for i, (id1, t1) in enumerate(texts):
        for j, (id2, t2) in enumerate(texts):
            if i >= j:
                continue
            if len(t1) < len(t2) and t1 in t2 and len(t1) < 50:
                result["containments"].append({
                    "short_id": id1,
                    "short_text": t1[:60],
                    "long_id": id2,
                    "long_text": t2[:60],
                })

    # 短条目检测
    for c in all_chunks:
        if 0 < len(c["text"]) < 20:
            result["short_entries"].append({
                "id": c["chunk_id"],
                "text": c["text"],
                "category": c["category"],
            })

    if result["duplicates"] or result["containments"] or result["short_entries"]:
        print(f"  去重报告:")
        if result["duplicates"]:
            print(f"    完全重复: {len(result['duplicates'])} 组")
        if result["containments"]:
            print(f"    包含关系: {len(result['containments'])} 对")
        if result["short_entries"]:
            print(f"    短条目(<20ch): {len(result['short_entries'])} 条")
    else:
        print("  无重复条目")

    return result


def health_check(engine: MemoryEngine) -> dict:
    """记忆引擎健康检查。"""
    stats = engine.stats()
    result = {
        "total_chunks": stats["total_chunks"],
        "categories": stats["categories"],
        "issues": [],
    }

    total = stats["total_chunks"]
    if total > 0:
        for cat, cnt in stats["categories"].items():
            pct = cnt / total * 100
            if pct > 35:
                result["issues"].append(
                    f"分类不均: {cat} 占比 {pct:.1f}% ({cnt}/{total})"
                )
            if pct < 3 and cnt > 0:
                result["issues"].append(
                    f"分类过少: {cat} 仅 {cnt} 条 ({pct:.1f}%)"
                )

    if stats.get("faiss_index_size", 0) != total:
        result["issues"].append(
            f"索引不一致: FAISS({stats.get('faiss_index_size', 0)}) vs DB({total})"
        )

    all_chunks = engine.store.get_all_chunks()
    short_count = sum(1 for c in all_chunks if 0 < len(c["text"]) < 20)
    if short_count > 0:
        result["issues"].append(f"短条目: {short_count} 条 (<20字符)")

    if result["issues"]:
        print(f"  健康检查: {len(result['issues'])} 个问题")
        for issue in result["issues"]:
            print(f"    - {issue}")
    else:
        print("  健康检查通过")

    result["is_healthy"] = len(result["issues"]) == 0
    state = load_state()
    state["last_stats"] = {
        "total": total,
        "categories": stats["categories"],
        "timestamp": datetime.now().isoformat(),
    }
    save_state(state)

    return result


def rebalance_categories(engine: MemoryEngine) -> dict:
    """分类重平衡。"""
    result = {"recategorized": 0, "changes": []}

    recategorize_rules = [
        ("学习笔记", "project_memory", "learning_note"),
        ("安全审计", "project_memory", "maintenance_log"),
        ("系统状态", "project_memory", "maintenance_log"),
        ("修复", "project_memory", "maintenance_log"),
        ("已知问题", "project_memory", "maintenance_log"),
        ("优化方向", "project_memory", "user_profile"),
        ("重要决定", "project_memory", "maintenance_log"),
        ("Vue 3 Dashboard", "project_memory", "dev_tool"),
        ("记忆分层", "project_memory", "general"),
    ]

    all_chunks = engine.store.get_all_chunks()
    for chunk in all_chunks:
        text = chunk["text"]
        current_cat = chunk["category"]

        for keywords, from_cat, to_cat in recategorize_rules:
            if current_cat == from_cat and keywords in text:
                engine.store.update_chunk(chunk["chunk_id"], category=to_cat)
                result["recategorized"] += 1
                result["changes"].append({
                    "id": chunk["chunk_id"],
                    "from": from_cat,
                    "to": to_cat,
                    "text": text[:60],
                })
                print(f"  重分类 #{chunk['chunk_id']}: {from_cat} -> {to_cat}")
                break

    if result["recategorized"] > 0:
        print(f"  重平衡完成: {result['recategorized']} 条已重分类")
    else:
        print("  分类已均衡，无需调整")

    return result


def print_stats(engine: MemoryEngine):
    """打印最终统计。"""
    stats = engine.stats()
    total = stats["total_chunks"]
    print(f"\n  最终统计: {total} 条 | {len(stats['categories'])} 个分类")
    for cat, cnt in sorted(stats["categories"].items(), key=lambda x: -x[1]):
        pct = cnt / total * 100 if total > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"    {cat:20s}: {cnt:3d} ({pct:5.1f}%) {bar}")


def main():
    parser = argparse.ArgumentParser(description="Heartbeat Memory 集成脚本")
    parser.add_argument("--sync-only", action="store_true", help="仅同步文件")
    parser.add_argument("--dedup", action="store_true", help="仅去重")
    parser.add_argument("--stats", action="store_true", help="仅统计")
    parser.add_argument("--rebalance", action="store_true", help="仅分类重平衡")
    parser.add_argument("--full", action="store_true", help="完整运行所有检查")
    args = parser.parse_args()

    run_all = not any([args.sync_only, args.dedup, args.stats, args.rebalance])

    print(f"\n{'='*50}")
    print(f"  Heartbeat Memory Check")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}\n")

    try:
        engine = get_engine()

        if args.sync_only or run_all or args.full:
            print("  同步检查:")
            sync_result = sync_files(engine)
            if sync_result["new"] > 0:
                print(f"    新增 {sync_result['new']} 条")

        if args.dedup or run_all or args.full:
            print("\n  去重检查:")
            auto_dedup(engine)

        if args.rebalance or run_all or args.full:
            print("\n  分类重平衡:")
            rebalance_result = rebalance_categories(engine)
            if rebalance_result["recategorized"] > 0:
                engine.rebuild_index()

        if args.stats or run_all or args.full:
            print("\n  健康检查:")
            health_check(engine)

        print_stats(engine)
        print()

    except Exception as e:
        print(f"\n  错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
