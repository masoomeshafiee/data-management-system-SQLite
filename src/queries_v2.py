from __future__ import annotations

import logging
import sqlite3
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, FrozenSet

import pandas as pd

from lab_db_app.config import FIELD_REGISTRY, FieldDef, BASE_EXPERIMENT_QUERY

logger = logging.getLogger(__name__)


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


# =========================================================
# Relationships / join graph
# =========================================================

TABLE_RELATIONSHIPS = {
    "Experiment": {
        "organism_id": "Organism",
        "protein_id": "Protein",
        "strain_id": "StrainOrCellLine",
        "condition_id": "Condition",
        "user_id": "User",
        "capture_setting_id": "CaptureSetting",
    },
    "RawFiles": {"experiment_id": "Experiment"},
    "TrackingFiles": {"experiment_id": "Experiment"},
    "Masks": {"experiment_id": "Experiment"},
    "Experiment_Analysis_Files_Link": {
        "experiment_id": "Experiment",
        "analysis_file_id": "AnalysisFiles",
    },
    "Results_Analysis_Files_Link": {
        "result_id": "Results",
        "analysis_file_id": "AnalysisFiles",
    },
}

BASE_EXPERIMENT_TABLES = {
    "Experiment",
    "Organism",
    "Protein",
    "StrainOrCellLine",
    "Condition",
    "CaptureSetting",
    "User",
}


def build_join_graph() -> dict[str, list[tuple[str, str]]]:
    graph: dict[str, list[tuple[str, str]]] = {}

    def add_edge(frm: str, to: str, sql: str) -> None:
        graph.setdefault(frm, []).append((to, sql))

    for frm, rels in TABLE_RELATIONSHIPS.items():
        for fk_col, to in rels.items():
            add_edge(frm, to, f"JOIN {to} ON {frm}.{fk_col} = {to}.id")
            add_edge(to, frm, f"JOIN {frm} ON {frm}.{fk_col} = {to}.id")

    return graph


_JOIN_GRAPH = build_join_graph()


def bfs_path_joins(main_table: str, target_table: str) -> list[str] | None:
    if main_table == target_table:
        return []

    q = deque([main_table])
    prev = {main_table: None}
    via_join = {main_table: None}

    while q:
        cur = q.popleft()
        for nxt, join_sql in _JOIN_GRAPH.get(cur, []):
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


# =========================================================
# SQL helpers
# =========================================================


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


def expand_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cur.fetchall()]


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


# =========================================================
# App-facing queries
# =========================================================

def list_users(conn: sqlite3.Connection) -> pd.DataFrame:
    sql = """
    SELECT id, user_name, last_name, email
    FROM User
    ORDER BY user_name COLLATE NOCASE;
    """
    return execute_query(conn, sql)


def get_experiment_metadata(conn: sqlite3.Connection, experiment_id: int) -> pd.DataFrame:
    sql = BASE_EXPERIMENT_QUERY + " WHERE Experiment.id = :experiment_id;"
    return execute_query(conn, sql, {"experiment_id": experiment_id})


def list_experiments(
    conn: sqlite3.Connection,
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 100,
) -> pd.DataFrame:
    where_clauses, params, joins = build_query_context(
        main_table="Experiment",
        filters=filters,
        base_tables=BASE_EXPERIMENT_TABLES,
    )

    query = BASE_EXPERIMENT_QUERY
    if joins:
        query += " " + " ".join(joins)
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    query += " ORDER BY Experiment.date DESC, Experiment.id DESC"
    if limit:
        query += " LIMIT :limit"
        params["limit"] = int(limit)

    return execute_query(conn, query, params)


def search_experiments(
    conn: sqlite3.Connection,
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 200,
) -> pd.DataFrame:
    return list_experiments(conn, filters=filters, limit=limit)


def list_entity(
    conn: sqlite3.Connection,
    requested_columns: list[str],
    main_table: str = "Experiment",
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 100,
) -> pd.DataFrame:
    filters = filters or {}
    sql_columns: list[str] = []
    requested_tables = {main_table}

    if requested_columns == ["*"]:
        all_cols = expand_columns(conn, main_table)
        sql_columns = [f"{main_table}.{c}" for c in all_cols]
    else:
        for alias in requested_columns:
            field = get_field(alias)
            if not field.selectable:
                raise ValueError(f"Field '{alias}' is not selectable.")
            sql_columns.append(f"{field.sql} AS {field.output_label}")
            requested_tables.add(field.table)

    for alias in filters.keys():
        requested_tables.add(get_field(alias).table)

    joins = infer_joins_bfs(
        requested_tables=requested_tables,
        main_table=main_table,
        base_tables={main_table},
    )

    where_clauses, params = get_where_clause_for_filters(filters)

    query = f"SELECT DISTINCT {', '.join(sql_columns)} FROM {main_table} " + " ".join(joins)
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    if limit:
        query += " LIMIT :limit"
        params["limit"] = int(limit)

    return execute_query(conn, query, params)

def search_table(
    conn: sqlite3.Connection,
    *,
    main_table: str,
    filters: Optional[Dict[str, Any]] = None,
    requested_columns: Optional[list[str]] = None,
    limit: int = 200,
) -> pd.DataFrame:
    filters = filters or {}

    if not requested_columns:
        requested_columns = get_default_columns_for_target(main_table)

    return list_entity(
        conn=conn,
        requested_columns=requested_columns,
        main_table=main_table,
        filters=filters,
        limit=limit,
    )


def list_experiments_between_dates(
    conn: sqlite3.Connection,
    start_date: str,
    end_date: str,
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 100,
) -> pd.DataFrame:
    extra_where = ["Experiment.date BETWEEN :start_date AND :end_date"]
    extra_params = {"start_date": start_date, "end_date": end_date}

    where_clauses, params, joins = build_query_context(
        main_table="Experiment",
        filters=filters,
        extra_where=extra_where,
        extra_params=extra_params,
        base_tables=BASE_EXPERIMENT_TABLES,
    )

    query = BASE_EXPERIMENT_QUERY
    if joins:
        query += " " + " ".join(joins)
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    query += " ORDER BY Experiment.date ASC"
    if limit:
        query += " LIMIT :limit"
        params["limit"] = int(limit)

    return execute_query(conn, query, params)


def find_most_recent_experiment(
    conn: sqlite3.Connection,
    filters: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    where_clauses, params, joins = build_query_context(
        main_table="Experiment",
        filters=filters,
        base_tables=BASE_EXPERIMENT_TABLES,
    )

    query = BASE_EXPERIMENT_QUERY
    if joins:
        query += " " + " ".join(joins)
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    query += " ORDER BY Experiment.date DESC, Experiment.id DESC LIMIT 1"

    return execute_query(conn, query, params)


def find_earliest_experiment(
    conn: sqlite3.Connection,
    filters: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    where_clauses, params, joins = build_query_context(
        main_table="Experiment",
        filters=filters,
        base_tables=BASE_EXPERIMENT_TABLES,
    )

    query = BASE_EXPERIMENT_QUERY
    if joins:
        query += " " + " ".join(joins)
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    query += " ORDER BY Experiment.date ASC, Experiment.id ASC LIMIT 1"

    return execute_query(conn, query, params)


def count_experiments_by_period(
    conn: sqlite3.Connection,
    period: str = "year",
    filters: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    if period == "year":
        date_expr = "substr(Experiment.date, 1, 4)"
    elif period == "month":
        date_expr = "substr(Experiment.date, 1, 7)"
    else:
        raise ValueError("period must be 'year' or 'month'")

    where_clauses, params, joins = build_query_context(
        main_table="Experiment",
        filters=filters,
        base_tables=BASE_EXPERIMENT_TABLES,
    )

    query = f"SELECT {date_expr} AS period, COUNT(DISTINCT Experiment.id) AS experiment_count FROM Experiment"
    if joins:
        query += " " + " ".join(joins)
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    query += " GROUP BY period ORDER BY period ASC"

    return execute_query(conn, query, params)


def find_experiments_missing_files(
    conn: sqlite3.Connection,
    file_types: list[str] | tuple[str, ...] = ("raw", "tracking", "mask", "analysis"),
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 50,
) -> pd.DataFrame:
    where_clauses, params, joins = build_query_context(
        main_table="Experiment",
        filters=filters,
        base_tables=BASE_EXPERIMENT_TABLES,
    )

    file_joins = []
    if "raw" in file_types:
        file_joins.append("LEFT JOIN RawFiles ON RawFiles.experiment_id = Experiment.id")
        where_clauses.append("RawFiles.id IS NULL")
    if "tracking" in file_types:
        file_joins.append("LEFT JOIN TrackingFiles ON TrackingFiles.experiment_id = Experiment.id")
        where_clauses.append("TrackingFiles.id IS NULL")
    if "mask" in file_types:
        file_joins.append("LEFT JOIN Masks ON Masks.experiment_id = Experiment.id")
        where_clauses.append("Masks.id IS NULL")
    if "analysis" in file_types:
        file_joins.append("LEFT JOIN ExperimentAnalysisFiles ON ExperimentAnalysisFiles.experiment_id = Experiment.id")
        file_joins.append("LEFT JOIN AnalysisFiles ON AnalysisFiles.id = ExperimentAnalysisFiles.analysis_file_id")
        where_clauses.append("AnalysisFiles.id IS NULL")

    query = BASE_EXPERIMENT_QUERY
    if joins:
        query += " " + " ".join(joins)
    if file_joins:
        query += " " + " ".join(file_joins)
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    query += " ORDER BY Experiment.date DESC, Experiment.id DESC"
    if limit:
        query += " LIMIT :limit"
        params["limit"] = int(limit)

    return execute_query(conn, query, params)



# =========================================================
# files for the experiment queries
# =========================================================

def get_raw_files_for_experiment(conn: sqlite3.Connection, experiment_id: int) -> pd.DataFrame:
    sql = """
    SELECT id, file_name, file_type, file_path
    FROM RawFiles
    WHERE experiment_id = :experiment_id
    ORDER BY id
    """
    return execute_query(conn, sql, {"experiment_id": experiment_id})


def get_tracking_files_for_experiment(conn: sqlite3.Connection, experiment_id: int) -> pd.DataFrame:
    sql = """
    SELECT id, file_name, file_type, file_path, threshold, linking_distance, gap_closing_distance, max_frame_gap, trackmate_settings_json
    FROM TrackingFiles
    WHERE experiment_id = :experiment_id
    ORDER BY id
    """
    return execute_query(conn, sql, {"experiment_id": experiment_id})


def get_masks_for_experiment(conn: sqlite3.Connection, experiment_id: int) -> pd.DataFrame:
    sql = """
    SELECT id, file_name, mask_type, file_type, file_path, segmentation_method, segmentation_parameters
    FROM Masks
    WHERE experiment_id = :experiment_id
    ORDER BY id
    """
    return execute_query(conn, sql, {"experiment_id": experiment_id})


def get_analysis_files_for_experiment(conn: sqlite3.Connection, experiment_id: int) -> pd.DataFrame:
    sql = """
    SELECT af.id, af.file_name, af.file_type, af.file_path
    FROM Experiment_Analysis_Files_link eaf
    JOIN AnalysisFiles af ON af.id = eaf.analysis_file_id
    WHERE eaf.experiment_id = :experiment_id
    ORDER BY af.id
    """
    return execute_query(conn, sql, {"experiment_id": experiment_id})

def get_results_for_experiment(conn: sqlite3.Connection, experiment_id: int) -> pd.DataFrame:
    sql = """
    SELECT DISTINCT
        r.id,
        r.result_type,
        r.result_value,
        r.sample_size,
        r.standard_error,
        r.analysis_method,
        r.analysis_parameters_json
    FROM Experiment_Analysis_Files_Link eaf
    JOIN AnalysisFiles af
        ON af.id = eaf.analysis_file_id
    JOIN Result_Analysis_Files_Link raf
        ON raf.analysis_file_id = af.id
    JOIN Results r
        ON r.id = raf.result_id
    WHERE eaf.experiment_id = :experiment_id
    ORDER BY r.id
    """
    return execute_query(conn, sql, {"experiment_id": experiment_id})

