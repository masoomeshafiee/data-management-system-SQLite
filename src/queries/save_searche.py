from __future__ import annotations

import json
import sqlite3
from typing import Any

import pandas as pd


def list_saved_searches(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    target_table: str | None = None,
    include_shared: bool = True,
) -> pd.DataFrame:
    where = ["(created_by_user_id = :user_id"]
    params: dict[str, Any] = {"user_id": user_id}

    if include_shared:
        where[0] += " OR is_shared = 1)"
    else:
        where[0] += ")"

    if target_table:
        where.append("target_table = :target_table")
        params["target_table"] = target_table

        sql = f"""
    SELECT
        ss.id,
        ss.name,
        ss.target_table,
        ss.created_by_user_id,
        ss.is_shared,
        ss.created_at,
        ss.updated_at,
        u.user_name
    FROM SavedSearch ss
    JOIN User u ON u.id = ss.created_by_user_id
    WHERE {" AND ".join(where)}
    ORDER BY ss.is_shared DESC, ss.name COLLATE NOCASE
    """
    return pd.read_sql_query(sql, conn, params=params)


def get_saved_search_by_id(conn: sqlite3.Connection, saved_search_id: int) -> dict[str, Any] | None:
    sql = """
    SELECT *
    FROM SavedSearch
    WHERE id = :saved_search_id
    """
    df = pd.read_sql_query(sql, conn, params={"saved_search_id": saved_search_id})
    if df.empty:
        return None

    row = df.iloc[0].to_dict()
    return {
        "id": int(row["id"]),
        "name": row["name"],
        "target_table": row["target_table"],
        "filters": json.loads(row["filters_json"] or "{}"),
        "selected_filters": json.loads(row["selected_filters_json"] or "[]"),
        "selected_columns": json.loads(row["selected_columns_json"] or "[]"),
        "limit": int(row["result_limit"]),
        "created_by_user_id": int(row["created_by_user_id"]),
        "is_shared": bool(row["is_shared"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def create_saved_search(
    conn: sqlite3.Connection,
    *,
    name: str,
    target_table: str,
    filters: dict[str, Any],
    selected_filters: list[str],
    selected_columns: list[str],
    result_limit: int,
    created_by_user_id: int,
    is_shared: bool = False,
) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO SavedSearch (
            name,
            target_table,
            filters_json,
            selected_filters_json,
            selected_columns_json,
            result_limit,
            created_by_user_id,
            is_shared
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            name.strip(),
            target_table,
            json.dumps(filters, ensure_ascii=False, sort_keys=True),
            json.dumps(selected_filters, ensure_ascii=False, sort_keys=True),
            json.dumps(selected_columns, ensure_ascii=False, sort_keys=True),
            int(result_limit),
            int(created_by_user_id),
            1 if is_shared else 0,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def update_saved_search(
    conn: sqlite3.Connection,
    *,
    saved_search_id: int,
    name: str,
    target_table: str,
    filters: dict[str, Any],
    selected_filters: list[str],
    selected_columns: list[str],
    result_limit: int,
    is_shared: bool,
) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE SavedSearch
        SET
            name = ?,
            target_table = ?,
            filters_json = ?,
            selected_filters_json = ?,
            selected_columns_json = ?,
            result_limit = ?,
            is_shared = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (
            name.strip(),
            target_table,
            json.dumps(filters, ensure_ascii=False, sort_keys=True),
            json.dumps(selected_filters, ensure_ascii=False, sort_keys=True),
            json.dumps(selected_columns, ensure_ascii=False, sort_keys=True),
            int(result_limit),
            1 if is_shared else 0,
            int(saved_search_id),
        ),
    )
    conn.commit()


def delete_saved_search(conn: sqlite3.Connection, saved_search_id: int) -> None:
    cur = conn.cursor()
    cur.execute("DELETE FROM SavedSearch WHERE id = ?", (int(saved_search_id),))
    conn.commit()