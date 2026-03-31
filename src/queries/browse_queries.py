from __future__ import annotations

import logging
import sqlite3
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, FrozenSet

import pandas as pd

from config import BASE_EXPERIMENT_QUERY, BASE_EXPERIMENT_TABLES
from queries.queries_utils import expand_columns,get_field, get_default_columns_for_target, infer_joins_bfs,get_where_clause_for_filters, build_query_context, execute_query

logger = logging.getLogger(__name__)



# =========================================================
# App-facing queries
# =========================================================
def get_distinct_values(conn: sqlite3.Connection, table: str, column: str) -> list[str]:
    query = f"""
        SELECT DISTINCT {column}
        FROM {table}
        WHERE {column} IS NOT NULL AND TRIM(CAST({column} AS TEXT)) <> ''
        ORDER BY {column} COLLATE NOCASE
    """
    cur = conn.execute(query)
    return [row[0] for row in cur.fetchall()]

def list_users(conn: sqlite3.Connection) -> pd.DataFrame:
    sql = """
    SELECT id, user_name, last_name, email
    FROM User
    ORDER BY user_name COLLATE NOCASE;
    """
    return execute_query(conn, sql)



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


def search_experiments(
    conn: sqlite3.Connection,
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 200,
) -> pd.DataFrame:
    return list_experiments(conn, filters=filters, limit=limit)


def get_experiment_metadata(conn: sqlite3.Connection, experiment_id: int) -> pd.DataFrame:
    sql = BASE_EXPERIMENT_QUERY + " WHERE Experiment.id = :experiment_id;"
    return execute_query(conn, sql, {"experiment_id": experiment_id})

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

