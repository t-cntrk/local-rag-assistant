"""SQLite yardımcıları.

Şema:
    chunks(id INTEGER PRIMARY KEY, source TEXT, title TEXT,
           content TEXT, embedding TEXT)

embedding alanı, vektörün JSON string hâlidir (ör. "[0.12, -0.03, ...]").
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    """Verilen yola bir SQLite bağlantısı açar."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path) -> None:
    """chunks tablosunu (varsa silip) yeniden oluşturur."""
    conn = get_connection(db_path)
    try:
        conn.execute("DROP TABLE IF EXISTS chunks")
        conn.execute(
            """
            CREATE TABLE chunks (
                id        INTEGER PRIMARY KEY,
                source    TEXT,
                title     TEXT,
                content   TEXT,
                embedding TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def insert_chunk(
    db_path: str | Path,
    source: str,
    title: str,
    content: str,
    embedding: list[float],
) -> int:
    """Tek bir chunk'ı ekler. embedding JSON string olarak saklanır.

    Eklenen satırın id'sini döndürür.
    """
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            "INSERT INTO chunks (source, title, content, embedding) "
            "VALUES (?, ?, ?, ?)",
            (source, title, content, json.dumps(embedding)),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def get_all_chunks(db_path: str | Path) -> list[dict[str, Any]]:
    """Tüm chunk'ları döndürür. embedding, list[float] olarak çözülür."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT id, source, title, content, embedding FROM chunks ORDER BY id"
        ).fetchall()
    finally:
        conn.close()

    result: list[dict[str, Any]] = []
    for row in rows:
        result.append(
            {
                "id": row["id"],
                "source": row["source"],
                "title": row["title"],
                "content": row["content"],
                "embedding": json.loads(row["embedding"]) if row["embedding"] else [],
            }
        )
    return result
