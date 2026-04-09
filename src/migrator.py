"""语义记忆引擎 — 迁移脚本

从现有 MEMORY.md 文件解析并迁移到语义记忆引擎。
支持两种格式：
1. ~/.hermes/memories/MEMORY.md — § 分隔的向量记忆格式
2. ~/.openclaw/workspace/MEMORY.md — 结构化 Markdown
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from src.chunker import Chunk, Chunker


# ── 分类映射规则 ──
_CATEGORY_KEYWORDS = {
    "identity": ["关于我", "名字：小轩", "首次上线", "模型：", "关于我:"],
    "user_profile": ["关于用户", "工作方向", "名字：Balance", "时区", "目标：", "希望我", "授权了"],
    "security_rule": ["安全规则", "绝不外发", "工作目录", "私密信息"],
    "dev_tool": ["开发工具", "Claude Code", "Codex", "OpenCode", "Trae", "coding-agent"],
    "project_memory": ["项目路径", "架构", "项目", "系统", "Agent"],
    "maintenance_log": ["维护", "修复", "更新", "版本", "部署", "日志"],
    "learning_note": ["学习笔记", "核心知识点", "课程", "Day "],
}

# BrowserWing 专用
_BROWSERWING_KEYWORDS = ["BrowserWing", "executor API", "浏览器自动化", "AI Agent API"]


@dataclass
class MigrationEntry:
    """从 MEMORY.md 解析出的一条记忆"""
    text: str
    category: str = "general"
    source: str = ""
    metadata: dict = field(default_factory=dict)


class Migrator:
    """记忆迁移器。

    解析旧格式 MEMORY.md，自动分类后交给 MemoryEngine 入库。
    """

    def __init__(self):
        self.chunker = Chunker()

    # ── 赫密斯格式迁移 ──
    def parse_hermes_memory(self, path: str) -> list[MigrationEntry]:
        """解析 ~/.hermes/memories/MEMORY.md。

        格式：§ 分隔，每行是 "类别: 内容" 或 "类别 > 子类别: 内容"
        """
        p = Path(path).expanduser()
        if not p.exists():
            return []

        text = p.read_text(encoding="utf-8")
        entries: list[MigrationEntry] = []

        # 按 § 分割
        sections = [s.strip() for s in text.split("§") if s.strip()]

        for section in sections:
            # 清理 Markdown
            section = re.sub(r'\*\*(.+?)\*\*', r'\1', section)
            section = section.strip()
            if not section:
                continue

            # 尝试提取类别（格式："类别: 内容" 或 "类别 > 子类别: 内容"）
            category = "general"
            clean_text = section

            # 提取前缀
            prefix_match = re.match(r'^([^:：>]+?)\s*[>：:]\s*(.+)$', section, re.DOTALL)
            if prefix_match:
                prefix = prefix_match.group(1).strip()
                content = prefix_match.group(2).strip()
                category = self._classify(prefix + " " + content)
                clean_text = f"{prefix}: {content}"
            else:
                category = self._classify(section)
                clean_text = section

            if len(clean_text) < 10:
                continue

            entries.append(MigrationEntry(
                text=clean_text,
                category=category,
                source="hermes_memory",
                metadata={"original_path": str(p)},
            ))

        return entries

    # ── OpenClaw 格式迁移 ──
    def parse_openclaw_memory(self, path: str) -> list[MigrationEntry]:
        """解析 ~/.openclaw/workspace/MEMORY.md。

        Markdown 结构：
        ## 一级标题（类别）
        ### 二级标题
        - 列表项（具体内容）
        """
        p = Path(path).expanduser()
        if not p.exists():
            return []

        text = p.read_text(encoding="utf-8")
        entries: list[MigrationEntry] = []

        # 解析 Markdown 结构
        lines = text.split("\n")
        current_section = ""
        current_subsection = ""
        section_items: list[str] = []

        def flush_section():
            if not current_section or not section_items:
                return
            combined = " ".join(section_items)
            if len(combined) > 5:
                category = self._classify(current_section + " " + combined)
                entries.append(MigrationEntry(
                    text=combined.strip(),
                    category=category,
                    source="openclaw_memory",
                    metadata={
                        "original_path": str(p),
                        "section": current_section,
                        "subsection": current_subsection,
                    },
                ))

        for line in lines:
            stripped = line.strip()

            # 跳过空行和分隔线
            if not stripped or stripped.startswith("---"):
                continue

            # 跳过文件头
            if stripped.startswith("# MEMORY.md"):
                continue

            # 一级标题 → 新类别
            h2_match = re.match(r'^##\s+(.+)$', stripped)
            if h2_match:
                flush_section()
                current_section = h2_match.group(1).strip()
                current_subsection = ""
                section_items = []
                continue

            # 二级标题
            h3_match = re.match(r'^###\s+(.+)$', stripped)
            if h3_match:
                flush_section()
                current_subsection = h3_match.group(1).strip()
                section_items = []
                continue

            # 列表项
            item_match = re.match(r'^[-*]\s+(.+)$', stripped)
            if item_match:
                item = item_match.group(1).strip()
                # 清理 Markdown
                item = re.sub(r'\*\*(.+?)\*\*', r'\1', item)
                item = re.sub(r'`(.+?)`', r'\1', item)

                prefix = f"{current_section}"
                if current_subsection:
                    prefix += f" > {current_subsection}"

                section_items.append(f"{prefix}: {item}")
                continue

            # 非列表文本（段落）
            if stripped and not stripped.startswith("#"):
                prefix = current_section
                if current_subsection:
                    prefix += f" > {current_subsection}"
                clean = re.sub(r'\*\*(.+?)\*\*', r'\1', stripped)
                section_items.append(f"{prefix}: {clean}")

        flush_section()
        return entries

    # ── 分类 ──
    def _classify(self, text: str) -> str:
        """根据关键词自动分类。"""
        text_lower = text.lower()

        # BrowserWing 优先
        for kw in _BROWSERWING_KEYWORDS:
            if kw.lower() in text_lower:
                return "project_memory"

        for category, keywords in _CATEGORY_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in text_lower:
                    return category

        return "general"

    # ── 辅助 ──
    def migrate_entries(
        self,
        entries: list[MigrationEntry],
        ingest_fn,  # 回调: (text, category, source, metadata) -> None
        verbose: bool = True,
    ) -> dict:
        """批量迁移条目。

        Args:
            entries: MigrationEntry 列表
            ingest_fn: 入库回调函数
            verbose: 是否打印进度

        Returns:
            迁移统计
        """
        stats = {"total": 0, "ingested": 0, "skipped": 0, "by_category": {}}

        for entry in entries:
            stats["total"] += 1

            if not entry.text or len(entry.text.strip()) < 5:
                stats["skipped"] += 1
                continue

            ingest_fn(
                text=entry.text.strip(),
                category=entry.category,
                source=entry.source,
                metadata=entry.metadata,
            )

            stats["ingested"] += 1
            cat = entry.category
            stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1

            if verbose:
                print(f"  ✓ [{cat}] {entry.text[:60]}...")

        return stats
