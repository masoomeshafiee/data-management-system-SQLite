import sqlite3
from typing import Any, Dict, List, Tuple, Optional, Iterable
from collections import deque
import pandas as pd
from config import TABLE_RELATIONSHIPS, FIELD_REGISTRY, FieldDef

import logging 

logger = logging.getLogger(__name__)


# ------------------------------------------------
# DB schema utils
# ------------------------------------------------

def expand_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cur.fetchall()]

# =========================================================
# Field registry helpers
# =========================================================

def get_fields_for_target(target_table: str) -> list[FieldDef]:
    return [
        f for f in FIELD_REGISTRY.values()
        if target_table in f.applies_to
    ]


def get_default_columns_for_target(target_table: str) -> list[str]:
    return [
        f.alias for f in FIELD_REGISTRY.values()
        if target_table in f.applies_to and f.default_visible
    ]


def get_filterable_fields_for_target(target_table: str) -> list[FieldDef]:
    return [
        f for f in FIELD_REGISTRY.values()
        if target_table in f.applies_to and f.filterable
    ]
def get_selectable_fields_for_target(target_table: str) -> list[FieldDef]:
    return [
        f for f in FIELD_REGISTRY.values()
        if target_table in f.applies_to and f.selectable
    ]


def group_fields_by_section(fields: list[FieldDef]) -> dict[str, list[FieldDef]]:
    grouped: dict[str, list[FieldDef]] = {}
    for f in fields:
        grouped.setdefault(f.section, []).append(f)

    for section in grouped:
        grouped[section] = sorted(grouped[section], key=lambda x: x.output_label.lower())

    return dict(sorted(grouped.items(), key=lambda kv: kv[0]))

def get_field(alias: str) -> FieldDef:
    if alias not in FIELD_REGISTRY:
        raise ValueError(f"Unknown field alias: {alias}")
    return FIELD_REGISTRY[alias]


# ----------------------------------------------------
# Join inference utilities
# ----------------------------------------------------
def build_join_graph() -> dict[str, list[tuple[str, str]]]:
    graph: dict[str, list[tuple[str, str]]] = {}

    def add_edge(frm: str, to: str, sql: str) -> None:
        graph.setdefault(frm, []).append((to, sql))

    for frm, rels in TABLE_RELATIONSHIPS.items():
        for fk_col, to in rels.items():
            add_edge(frm, to, f"JOIN {to} ON {frm}.{fk_col} = {to}.id")
            add_edge(to, frm, f"JOIN {frm} ON {frm}.{fk_col} = {to}.id")

    return graph


def bfs_path_joins(main_table: str, target_table: str) -> list[str] | None:
    if main_table == target_table:
        return []

    q = deque([main_table])
    prev = {main_table: None}
    via_join = {main_table: None}

    while q:
        cur = q.popleft()
        for nxt, join_sql in build_join_graph().get(cur, []):
            if nxt in prev:
                continue
            prev[nxt] = cur
            via_join[nxt] = join_sql

            if nxt == target_table:
                path = []
                node = nxt
                while node != main_table:
                    path.append(via_join[node])
                    node = prev[node]
                path.reverse()
                return path

            q.append(nxt)

    return None

def infer_joins_bfs(
    requested_tables: Iterable[str],
    main_table: str,
    base_tables: Optional[set[str]] = None,
) -> list[str]:
    base_tables = set(base_tables or set())
    requested = set(requested_tables or set())
    requested.discard(main_table)

    def joined_table(join_sql: str) -> str:
        return join_sql.split()[1]

    seen = set()
    joins: list[str] = []

    for target in sorted(requested):
        path = bfs_path_joins(main_table, target)
        if path is None:
            raise ValueError(f"No join path from {main_table} to {target}")
        for join_sql in path:
            jt = joined_table(join_sql)
            if jt in base_tables:
                continue
            if join_sql not in seen:
                seen.add(join_sql)
                joins.append(join_sql)

    return joins


# =======================================================
# Querie builder utils
# =======================================================


def build_where(unique_fields: Dict[str, Any]) -> Tuple[str, List[Any]]:
    """
    Builds WHERE clause that correctly handles NULL comparisons.
    """
    parts = []
    values = []
    for k, v in unique_fields.items():
        if v is None:
            parts.append(f"{k} IS NULL")
        else:
            parts.append(f"{k} = ?")
            values.append(v)
    return " AND ".join(parts), values

def get_where_clause_for_filters(filters: Optional[Dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    filters = filters or {}
    where_clauses: list[str] = []
    params: dict[str, Any] = {}

    for key, value in filters.items():
        field = get_field(key)

        if not field.filterable:
            raise ValueError(f"Field '{key}' is not filterable.")

        if value is None or value == "":
            continue

        if field.data_type == "text":
            where_clauses.append(f"{field.sql} = :{key} COLLATE NOCASE")
        else:
            where_clauses.append(f"{field.sql} = :{key}")

        params[key] = value

    return where_clauses, params


def build_query_context(
    *,
    main_table: str,
    filters: Optional[Dict[str, Any]] = None,
    extra_where: Optional[list[str]] = None,
    extra_params: Optional[Dict[str, Any]] = None,
    base_tables: Optional[set[str]] = None,
    required_tables_extra: Optional[set[str]] = None,
) -> tuple[list[str], dict[str, Any], list[str]]:
    where_clauses: list[str] = []
    params: dict[str, Any] = {}
    required_tables = set(required_tables_extra or set())

    filter_clauses, filter_params = get_where_clause_for_filters(filters)
    where_clauses.extend(filter_clauses)
    params.update(filter_params)

    for key in (filters or {}).keys():
        tbl = get_field(key).table
        if not base_tables or tbl not in base_tables:
            required_tables.add(tbl)

    joins = infer_joins_bfs(
        requested_tables=required_tables,
        main_table=main_table,
        base_tables=base_tables,
    )

    if extra_where:
        where_clauses.extend(extra_where)
    if extra_params:
        params.update(extra_params)

    return where_clauses, params, joins

# -----------------------------------------------
# Execute query
# -----------------------------------------------
def execute_query(
    conn: sqlite3.Connection,
    query: str,
    params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    try:
        return pd.read_sql_query(query, conn, params=params or {})
    except Exception:
        logger.exception("Query failed")
        raise
