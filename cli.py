#!/usr/bin/env python3
"""语义记忆引擎 — CLI 命令行工具

用法:
    python cli.py ingest "记忆内容" --category user_goal
    python cli.py query "用户的目标是什么？" --top-k 5
    python cli.py migrate                    # 迁移现有记忆
    python cli.py stats                      # 查看统计
    python cli.py list --category security   # 按类别查看
"""
from __future__ import annotations

import sys
from pathlib import Path

# 确保项目根在 path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import click
from rich.console import Console
from rich.table import Table

from config import DEFAULT_DB_PATH, DEFAULT_TOP_K, HERMES_MEMORY, OPENCLAW_MEMORY
from src.engine import MemoryEngine

console = Console()


@click.group()
@click.option("--db", default=DEFAULT_DB_PATH, help="SQLite 数据库路径")
@click.pass_context
def cli(ctx, db):
    """🧠 语义记忆引擎 — 中文优化的语义记忆管理工具"""
    ctx.ensure_object(dict)
    ctx.obj["db"] = db


# ── Ingest ──
@cli.command()
@click.argument("text")
@click.option("--category", "-c", default="general", help="分类标签")
@click.option("--source", "-s", default="", help="来源标识")
@click.pass_context
def ingest(ctx, text, category, source):
    """入库一条记忆"""
    engine = MemoryEngine(db_path=ctx.obj["db"])
    ids = engine.ingest(text, category=category, source=source)
    if ids:
        console.print(f"[green]✅ 已入库 {len(ids)} 条 chunk，ID: {ids}[/green]")
        console.print(f"   分类: {category} | 来源: {source or '默认'}")
    else:
        console.print("[yellow]⚠️  文本为空，跳过[/yellow]")


# ── Query ──
@cli.command()
@click.argument("query")
@click.option("--top-k", "-k", default=DEFAULT_TOP_K, help="返回数量")
@click.option("--min-score", default=0.3, help="最低相似度阈值")
@click.option("--interleave", "-i", is_flag=True, help="启用多跳检索")
@click.pass_context
def query(ctx, query, top_k, min_score, interleave):
    """语义检索记忆"""
    engine = MemoryEngine(db_path=ctx.obj["db"])

    if interleave:
        results = engine.query_interleave(query, top_k=top_k, min_score=min_score)
        mode = "多跳检索"
    else:
        results = engine.query(query, top_k=top_k, min_score=min_score)
        mode = "语义检索"

    if not results:
        console.print("[yellow]未找到相关记忆[/yellow]")
        return

    console.print(f"\n🔍 {mode} — 查询: [bold]{query}[/bold]\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("分数", justify="right", width=7)
    table.add_column("分类", width=15)
    table.add_column("内容", min_width=40)

    for i, r in enumerate(results, 1):
        table.add_row(
            str(i),
            f"{r['score']:.4f}",
            r.get("category", ""),
            r["text"][:120] + ("..." if len(r["text"]) > 120 else ""),
        )

    console.print(table)


# ── Migrate ──
@cli.command()
@click.option("--hermes", is_flag=True, help="仅迁移 Heremes 记忆")
@click.option("--openclaw", is_flag=True, help="仅迁移 OpenClaw 记忆")
@click.pass_context
def migrate(ctx, hermes, openclaw):
    """迁移现有 MEMORY.md 文件"""
    engine = MemoryEngine(db_path=ctx.obj["db"])

    if hermes and not openclaw:
        console.print(f"\n🔄 迁移 Heremes 记忆: {HERMES_MEMORY}")
        stats = engine.migrate_from_hermes_memory()
    elif openclaw and not hermes:
        console.print(f"\n🔄 迁移 OpenClaw 记忆: {OPENCLAW_MEMORY}")
        stats = engine.migrate_from_openclaw_memory()
    else:
        console.print("\n🔄 迁移所有可用记忆源...")
        stats = engine.migrate_all()

    console.print("\n📊 迁移统计:")
    for key, val in stats.items():
        if isinstance(val, dict):
            console.print(f"  {key}: {val}")
        else:
            console.print(f"  {key}: {val}")


# ── Stats ──
@cli.command()
@click.pass_context
def stats(ctx):
    """查看记忆统计"""
    engine = MemoryEngine(db_path=ctx.obj["db"])
    s = engine.stats()

    console.print("\n📊 记忆引擎统计\n")

    console.print(f"  数据库:     {s.get('db_path', '')}")
    console.print(f"  索引文件:   {s.get('index_path', '')}")
    console.print(f"  模型:       {s.get('embedding_model', '')}")
    console.print(f"  向量维度:   {s.get('embedding_dim', '')}")
    console.print(f"  总块数:     {s.get('total_chunks', 0)}")
    console.print(f"  索引大小:   {s.get('faiss_index_size', 0)} 向量")

    if s.get("categories"):
        console.print("\n  按分类:")
        for cat, cnt in s["categories"].items():
            bar = "█" * min(cnt, 30)
            console.print(f"    {cat:20s} {bar} ({cnt})")

    if s.get("sources"):
        console.print("\n  按来源:")
        for src, cnt in s["sources"].items():
            console.print(f"    {src:30s} ({cnt})")


# ── List ──
@cli.command()
@click.option("--category", "-c", default=None, help="过滤分类")
@click.option("--limit", "-l", default=20, help="显示条数")
@click.pass_context
def list_cmd(ctx, category, limit):
    """列出记忆条目"""
    engine = MemoryEngine(db_path=ctx.obj["db"])

    if category:
        items = engine.list_by_category(category)
        console.print(f"\n📋 分类: [bold]{category}[/bold] ({len(items)} 条)\n")
    else:
        s = engine.stats()
        items = []
        for cat in s.get("categories", {}):
            items.extend(engine.list_by_category(cat))
        console.print(f"\n📋 全部记忆 ({len(items)} 条)\n")

    for i, item in enumerate(items[:limit], 1):
        console.print(f"  [{i}] {item['text'][:100]}")
        console.print(f"      来源: {item.get('source', '')} | 时间: {item.get('created_at', '')[:10]}")
        console.print()

    if len(items) > limit:
        console.print(f"  ... 还有 {len(items) - limit} 条 (使用 --limit 增加)")


# ── Delete ──
@cli.command()
@click.argument("chunk_id", type=int)
@click.pass_context
def delete(ctx, chunk_id):
    """删除一条记忆"""
    engine = MemoryEngine(db_path=ctx.obj["db"])
    chunk = engine.store.get_chunk(chunk_id)
    if chunk:
        console.print(f"  删除: {chunk['text'][:80]}...")
        engine.delete(chunk_id)
        console.print("[green]✅ 已删除[/green]")
    else:
        console.print(f"[yellow]⚠️  未找到 chunk_id={chunk_id}[/yellow]")


# ── Rebuild ──
@cli.command()
@click.pass_context
def rebuild(ctx):
    """重建 FAISS 索引"""
    engine = MemoryEngine(db_path=ctx.obj["db"])
    engine.rebuild_index()
    console.print("[green]✅ 索引重建完成[/green]")


if __name__ == "__main__":
    cli(obj={})
