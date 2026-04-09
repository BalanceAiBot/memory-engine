"""语义记忆引擎 — 智能分块器

按段落/句子边界切分文本，避免截断语义。
支持中英文混合，优先在自然边界处分割。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from config import CHUNK_MAX_CHARS, CHUNK_MIN_CHARS, CHUNK_OVERLAP_CHARS


@dataclass
class Chunk:
    """分块结果"""
    text: str
    start: int = 0       # 原文起始偏移
    end: int = 0         # 原文结束偏移
    index: int = 0       # 块序号
    metadata: dict = field(default_factory=dict)


class Chunker:
    """智能分块器。

    策略：
    1. 先按双换行分段（段落级）
    2. 过长段落按句子边界（。！？.\n）拆分
    3. 仍然过长的块强制截断
    4. 相邻块保留重叠，防止上下文丢失
    """

    # 中英文句子结束符
    _SENTENCE_BOUNDARY = re.compile(
        r'(?<=[。！？\.\!\?\n\r])'
    )

    def __init__(
        self,
        max_chars: int = CHUNK_MAX_CHARS,
        min_chars: int = CHUNK_MIN_CHARS,
        overlap_chars: int = CHUNK_OVERLAP_CHARS,
    ):
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.overlap_chars = overlap_chars

    def chunk(self, text: str, metadata: dict | None = None) -> List[Chunk]:
        """将文本切分为智能块。

        Args:
            text: 待分块的原始文本
            metadata: 附加元数据（透传到每个 Chunk）

        Returns:
            Chunk 列表
        """
        if not text or not text.strip():
            return []

        meta = metadata or {}
        paragraphs = self._split_paragraphs(text)
        raw_chunks: list[tuple[str, int, int]] = []  # (text, start, end)

        for para_text, para_start, para_end in paragraphs:
            if len(para_text) <= self.max_chars:
                raw_chunks.append((para_text, para_start, para_end))
            else:
                # 长段落 → 按句子拆分
                raw_chunks.extend(
                    self._split_long_paragraph(para_text, para_start)
                )

        # 合并小块（保留最小长度）
        merged = self._merge_small_chunks(raw_chunks)

        # 构造 Chunk 对象
        chunks: List[Chunk] = []
        for i, (t, s, e) in enumerate(merged):
            if len(t.strip()) < self.min_chars:
                continue
            chunks.append(
                Chunk(
                    text=t.strip(),
                    start=s,
                    end=e,
                    index=i,
                    metadata=dict(meta),
                )
            )
        return chunks

    # ── 内部方法 ──
    def _split_paragraphs(self, text: str) -> list[tuple[str, int, int]]:
        """按双换行拆段落。"""
        parts: list[tuple[str, int, int]] = []
        for para in re.split(r'\n\s*\n', text):
            para = para.strip()
            if not para:
                continue
            start = text.find(para)
            parts.append((para, start, start + len(para)))
        return parts

    def _split_long_paragraph(
        self, text: str, base_offset: int
    ) -> list[tuple[str, int, int]]:
        """拆分超长段落。"""
        results: list[tuple[str, int, int]] = []
        sentences = self._SENTENCE_BOUNDARY.split(text)
        sentences = [s for s in sentences if s.strip()]

        current = ""
        current_start = base_offset

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if not current:
                current_start = base_offset + text.find(sent, text.find(current) if current else 0)

            if len(current) + len(sent) + 1 <= self.max_chars:
                current += " " + sent if current else sent
            else:
                if current.strip():
                    results.append((current.strip(), current_start, current_start + len(current)))
                # 重叠：从当前块尾部取 overlap
                if self.overlap_chars > 0 and current:
                    overlap = current[-self.overlap_chars:]
                    current = overlap + " " + sent
                else:
                    current = sent
                # 重新计算 start
                current_start = text.rfind(sent[:30], 0, text.find(sent) + len(sent))
                if current_start < 0:
                    current_start = base_offset

        if current.strip():
            results.append((current.strip(), current_start, current_start + len(current)))

        # 如果仍然没有拆成功（单句超长），强制截断
        if not results:
            for i in range(0, len(text), self.max_chars):
                segment = text[i:i + self.max_chars].strip()
                if len(segment) >= self.min_chars:
                    results.append((segment, base_offset + i, base_offset + i + len(segment)))

        return results

    def _merge_small_chunks(
        self, chunks: list[tuple[str, int, int]]
    ) -> list[tuple[str, int, int]]:
        """合并过短的相邻块。"""
        if not chunks:
            return chunks

        merged: list[tuple[str, int, int]] = []
        current_text, current_start, current_end = chunks[0]

        for text, start, end in chunks[1:]:
            if len(current_text) + len(text) + 1 <= self.max_chars and len(current_text) < self.min_chars:
                current_text += " " + text
                current_end = end
            else:
                merged.append((current_text, current_start, current_end))
                current_text, current_start, current_end = text, start, end

        merged.append((current_text, current_start, current_end))
        return merged
