"""语义记忆引擎 — SQLite 元数据存储

存储 chunk 文本、分类、来源、时间戳等元数据，
与 FAISS 向量索引通过 chunk_id 关联。
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class MetadataStore:
    """SQLite 元数据存储层。

    表结构：
    - chunks: 存储分块文本和元数据
    - index_meta: 存储 FAISS 索引元信息
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id    INTEGER PRIMARY KEY AUTOINCREMENT,
        text        TEXT NOT NULL,
        category    TEXT DEFAULT 'general',
        source      TEXT DEFAULT '',
        metadata    TEXT DEFAULT '{}',
        created_at  TEXT NOT NULL,
        updated_at  TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_category ON chunks(category);
    CREATE INDEX IF NOT EXISTS idx_source ON chunks(source);
    CREATE INDEX IF NOT EXISTS idx_created ON chunks(created_at);

    CREATE TABLE IF NOT EXISTS index_meta (
        key     TEXT PRIMARY KEY,
        value   TEXT NOT NULL
    );
    """

    def __init__(self, db_path: str):
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(self._SCHEMA)

    # ── 写入 ──
    def add_chunk(
        self,
        text: str,
        category: str = "general",
        source: str = "",
        metadata: dict | None = None,
    ) -> int:
        """添加一个 chunk，返回 chunk_id。"""
        now = datetime.now(timezone.utc).isoformat()
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        with self._conn() as conn:
            cursor = conn.execute(
                "INSERT INTO chunks (text, category, source, metadata, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (text, category, source, meta_json, now, now),
            )
            return cursor.lastrowid

    def add_chunks_batch(
        self,
        items: list[dict],
    ) -> list[int]:
        """批量添加 chunks，返回 chunk_id 列表。"""
        now = datetime.now(timezone.utc).isoformat()
        ids: list[int] = []
        with self._conn() as conn:
            for item in items:
                meta_json = json.dumps(item.get("metadata", {}), ensure_ascii=False)
                cursor = conn.execute(
                    "INSERT INTO chunks (text, category, source, metadata, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        item["text"],
                        item.get("category", "general"),
                        item.get("source", ""),
                        meta_json,
                        now,
                        now,
                    ),
                )
                ids.append(cursor.lastrowid)
        return ids

    # ── 查询 ──
    def get_chunk(self, chunk_id: int) -> dict | None:
        """根据 ID 获取单个 chunk。"""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()
            return self._row_to_dict(row) if row else None

    def get_chunks_by_ids(self, chunk_ids: list[int]) -> list[dict]:
        """批量获取 chunks（用于 FAISS 检索后回填元数据）。"""
        if not chunk_ids:
            return []
        placeholders = ",".join("?" * len(chunk_ids))
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_all_chunks(self) -> list[dict]:
        """获取全部 chunks（用于重建索引）。"""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks ORDER BY chunk_id"
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_chunks_by_category(self, category: str) -> list[dict]:
        """按分类获取 chunks。"""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE category = ? ORDER BY chunk_id",
                (category,),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def search_chunks(self, query: str, category: str | None = None) -> list[dict]:
        """关键词搜索（辅助功能，主要用于调试）。"""
        like = f"%{query}%"
        sql = "SELECT * FROM chunks WHERE text LIKE ?"
        params: list[Any] = [like]
        if category:
            sql += " AND category = ?"
            params.append(category)
        sql += " ORDER BY chunk_id"
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [self._row_to_dict(r) for r in rows]

    # ── 更新 / 删除 ──
    def update_chunk(self, chunk_id: int, **kwargs) -> bool:
        """更新 chunk 字段。"""
        now = datetime.now(timezone.utc).isoformat()
        fields = []
        values: list[Any] = []
        for k, v in kwargs.items():
            if k == "metadata":
                v = json.dumps(v, ensure_ascii=False)
            fields.append(f"{k} = ?")
            values.append(v)
        fields.append("updated_at = ?")
        values.append(now)
        values.append(chunk_id)

        with self._conn() as conn:
            cursor = conn.execute(
                f"UPDATE chunks SET {', '.join(fields)} WHERE chunk_id = ?",
                values,
            )
            return cursor.rowcount > 0

    def delete_chunk(self, chunk_id: int) -> bool:
        """删除一个 chunk。"""
        with self._conn() as conn:
            cursor = conn.execute(
                "DELETE FROM chunks WHERE chunk_id = ?", (chunk_id,)
            )
            return cursor.rowcount > 0

    # ── 索引元信息 ──
    def set_index_meta(self, key: str, value: str):
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO index_meta (key, value) VALUES (?, ?)",
                (key, value),
            )

    def get_index_meta(self, key: str) -> str | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM index_meta WHERE key = ?", (key,)
            ).fetchone()
            return row["value"] if row else None

    # ── 统计 ──
    def stats(self) -> dict:
        """返回数据库统计信息。"""
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) as cnt FROM chunks").fetchone()["cnt"]
            rows = conn.execute(
                "SELECT category, COUNT(*) as cnt FROM chunks GROUP BY category ORDER BY cnt DESC"
            ).fetchall()
            categories = {r["category"]: r["cnt"] for r in rows}

            source_rows = conn.execute(
                "SELECT source, COUNT(*) as cnt FROM chunks GROUP BY source ORDER BY cnt DESC"
            ).fetchall()
            sources = {r["source"]: r["cnt"] for r in source_rows}

            return {
                "total_chunks": total,
                "categories": categories,
                "sources": sources,
            }

    # ── 工具 ──
    @staticmethod
    def _row_to_dict(row) -> dict:
        return {
            "chunk_id": row["chunk_id"],
            "text": row["text"],
            "category": row["category"],
            "source": row["source"],
            "metadata": json.loads(row["metadata"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
