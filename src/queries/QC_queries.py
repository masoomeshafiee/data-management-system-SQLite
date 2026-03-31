import sqlite3
import pandas as pd
from typing import Dict, Any, Optional

from config import BASE_EXPERIMENT_QUERY, BASE_EXPERIMENT_TABLES
from queries.queries_utils import build_query_context, execute_query, get_field


# =========================================================
# Shared constants / helpers
# =========================================================


def _coalesce_text(expr: str) -> str:
    return f"COALESCE(NULLIF(TRIM({expr}), ''), '[missing]')"


def _limit_clause(limit: Optional[int], params: Dict[str, Any]) -> str:
    if limit is None:
        return ""
    params["limit"] = int(limit)
    return " LIMIT :limit"


# =========================================================
# 0) Invalid experiments
# =========================================================

def find_invalid_experiments(
    conn: sqlite3.Connection,
    filters: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = 100,
) -> pd.DataFrame:
    where_clauses, params, joins = build_query_context(
        main_table="Experiment",
        filters=filters,
        extra_where=["Experiment.is_valid = 0"],
        base_tables=BASE_EXPERIMENT_TABLES,
    )

    query = BASE_EXPERIMENT_QUERY
    if joins:
        query += " " + " ".join(joins)
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)

    query += " ORDER BY Experiment.date DESC, Experiment.id DESC"
    query += _limit_clause(limit, params)

    return execute_query(conn, query, params)


# =========================================================
# 1) Experiments missing expected file types
# =========================================================

def find_experiments_missing_files(
    conn: sqlite3.Connection,
    file_types: list[str] | tuple[str, ...] = ("raw", "tracking", "mask", "analysis"),
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 500,
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
        file_joins.append("LEFT JOIN Experiment_Analysis_Files_Link eaf ON eaf.experiment_id = Experiment.id")
        file_joins.append("LEFT JOIN AnalysisFiles ON AnalysisFiles.id = eaf.analysis_file_id")
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
# 2) Experiments missing selected metadata fields
# =========================================================

def find_experiments_missing_metadata(
    conn: sqlite3.Connection,
    required_fields: list[str],
    filters: Optional[Dict[str, Any]] = None,
    mode: str = "any",
    limit: int = 500,
) -> pd.DataFrame:
    if not required_fields:
        raise ValueError("required_fields cannot be empty")

    required_tables = set()
    missing_conditions = []

    for alias in required_fields:
        field = get_field(alias)
        required_tables.add(field.table)

        if field.data_type in {"text", "date"}:
            missing_conditions.append(f"({field.sql} IS NULL OR TRIM(CAST({field.sql} AS TEXT)) = '')")
        else:
            missing_conditions.append(f"{field.sql} IS NULL")

    where_clauses, params, joins = build_query_context(
        main_table="Experiment",
        filters=filters,
        base_tables=BASE_EXPERIMENT_TABLES,
        required_tables_extra=required_tables,
    )

    if mode == "any":
        where_clauses.append("(" + " OR ".join(missing_conditions) + ")")
    elif mode == "all":
        where_clauses.append("(" + " AND ".join(missing_conditions) + ")")
    else:
        raise ValueError("mode must be 'any' or 'all'")

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


# =========================================================
# 3) Generic missing values finder
# =========================================================

def find_missing_values(
    conn: sqlite3.Connection,
    *,
    main_table: str,
    requested_columns: list[str],
    missing_columns: list[str] | str,
    filters: Optional[Dict[str, Any]] = None,
    mode: str = "any",
    limit: int = 500,
) -> pd.DataFrame:
    filters = filters or {}

    if isinstance(missing_columns, str):
        missing_columns = [missing_columns]

    if not requested_columns:
        raise ValueError("requested_columns cannot be empty")
    if not missing_columns:
        raise ValueError("missing_columns cannot be empty")

    sql_requested_columns = []
    required_tables = {main_table}

    for alias in requested_columns:
        field = get_field(alias)
        sql_requested_columns.append(f"{field.sql} AS {field.output_label}")
        required_tables.add(field.table)

    missing_conditions = []
    for alias in missing_columns:
        field = get_field(alias)
        required_tables.add(field.table)

        if field.data_type in {"text", "date"}:
            missing_conditions.append(f"({field.sql} IS NULL OR TRIM(CAST({field.sql} AS TEXT)) = '')")
        else:
            missing_conditions.append(f"{field.sql} IS NULL")

    where_clauses, params, joins = build_query_context(
        main_table=main_table,
        filters=filters,
        required_tables_extra=required_tables,
        base_tables={main_table},
    )

    if mode == "any":
        where_clauses.append("(" + " OR ".join(missing_conditions) + ")")
    elif mode == "all":
        where_clauses.append("(" + " AND ".join(missing_conditions) + ")")
    else:
        raise ValueError("mode must be 'any' or 'all'")

    query = f"SELECT DISTINCT {', '.join(sql_requested_columns)} FROM {main_table}"
    if joins:
        query += " " + " ".join(joins)
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    query += " LIMIT :limit"
    params["limit"] = int(limit)

    return execute_query(conn, query, params)


# =========================================================
# 4) Logical duplicate experiments
# =========================================================

def find_duplicate_experiments(
    conn: sqlite3.Connection,
    filters: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    where_clauses, params, joins = build_query_context(
        main_table="Experiment",
        filters=filters,
        base_tables=BASE_EXPERIMENT_TABLES,
    )

    query = """
    SELECT
        Organism.organism_name AS organism,
        Protein.protein_name AS protein,
        StrainOrCellLine.strain_name AS strain,
        Condition.condition_name AS condition,
        Condition.concentration_value AS concentration_value,
        Condition.concentration_unit AS concentration_unit,
        CaptureSetting.capture_type AS capture_type,
        CaptureSetting.exposure_time AS exposure_time,
        CaptureSetting.time_interval AS time_interval,
        User.user_name AS user_name,
        Experiment.date AS date,
        Experiment.replicate AS replicate,
        COUNT(*) AS duplicate_count,
        GROUP_CONCAT(Experiment.id) AS experiment_ids
    FROM Experiment
    JOIN Organism ON Experiment.organism_id = Organism.id
    JOIN Protein ON Experiment.protein_id = Protein.id
    JOIN StrainOrCellLine ON Experiment.strain_id = StrainOrCellLine.id
    JOIN Condition ON Experiment.condition_id = Condition.id
    JOIN CaptureSetting ON Experiment.capture_setting_id = CaptureSetting.id
    LEFT JOIN User ON Experiment.user_id = User.id
    """

    if joins:
        query += " " + " ".join(joins)

    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)

    query += """
    GROUP BY
        Organism.organism_name,
        Protein.protein_name,
        StrainOrCellLine.strain_name,
        Condition.condition_name,
        Condition.concentration_value,
        Condition.concentration_unit,
        CaptureSetting.capture_type,
        CaptureSetting.exposure_time,
        CaptureSetting.time_interval,
        User.user_name,
        Experiment.date,
        Experiment.replicate
    HAVING COUNT(*) > 1
    ORDER BY duplicate_count DESC, date DESC
    """

    return execute_query(conn, query, params)


# =========================================================
# 5) Coverage summary by experiment
# =========================================================

def count_experiment_file_coverage(
    conn: sqlite3.Connection,
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 1000,
) -> pd.DataFrame:
    where_clauses, params, joins = build_query_context(
        main_table="Experiment",
        filters=filters,
        base_tables=BASE_EXPERIMENT_TABLES,
    )

    query = """
    SELECT
        Experiment.id AS experiment_id,
        Experiment.date,
        Experiment.replicate,
        Organism.organism_name AS organism,
        Protein.protein_name AS protein,
        Condition.condition_name AS condition,
        CaptureSetting.capture_type AS capture_type,
        COUNT(DISTINCT RawFiles.id) AS raw_count,
        COUNT(DISTINCT TrackingFiles.id) AS tracking_count,
        COUNT(DISTINCT Masks.id) AS mask_count,
        COUNT(DISTINCT AnalysisFiles.id) AS analysis_count,
        COUNT(DISTINCT Results.id) AS result_count
    FROM Experiment
    JOIN Organism ON Experiment.organism_id = Organism.id
    JOIN Protein ON Experiment.protein_id = Protein.id
    JOIN Condition ON Experiment.condition_id = Condition.id
    JOIN CaptureSetting ON Experiment.capture_setting_id = CaptureSetting.id
    LEFT JOIN RawFiles ON RawFiles.experiment_id = Experiment.id
    LEFT JOIN TrackingFiles ON TrackingFiles.experiment_id = Experiment.id
    LEFT JOIN Masks ON Masks.experiment_id = Experiment.id
    LEFT JOIN Experiment_Analysis_Files_Link eaf ON eaf.experiment_id = Experiment.id
    LEFT JOIN AnalysisFiles ON AnalysisFiles.id = eaf.analysis_file_id
    LEFT JOIN Result_Analysis_Files_Link raf ON raf.analysis_file_id = AnalysisFiles.id
    LEFT JOIN Results ON Results.id = raf.result_id
    """

    if joins:
        query += " " + " ".join(joins)

    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)

    query += """
    GROUP BY
        Experiment.id,
        Experiment.date,
        Experiment.replicate,
        Organism.organism_name,
        Protein.protein_name,
        Condition.condition_name,
        CaptureSetting.capture_type
    ORDER BY Experiment.date DESC, Experiment.id DESC
    """

    if limit:
        query += " LIMIT :limit"
        params["limit"] = int(limit)

    return execute_query(conn, query, params)


# =========================================================
# 6) Incomplete linked-entity checks
# =========================================================

def find_incomplete_linked_entities(
    conn: sqlite3.Connection,
    *,
    base_table: str = "Experiment",
    present_entity: tuple[str, str] = ("RawFiles", "experiment_id"),
    missing_entity: tuple[str, str] = ("TrackingFiles", "experiment_id"),
    present_bridge: Optional[tuple[str, str, str]] = None,
    missing_bridge: Optional[tuple[str, str, str]] = None,
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 500,
) -> pd.DataFrame:
    filters = filters or {}

    present_table, present_fk = present_entity
    missing_table, missing_fk = missing_entity

    where_clauses, params, joins = build_query_context(
        main_table=base_table,
        filters=filters,
        base_tables={base_table, present_table, missing_table},
    )

    if present_bridge:
        pb_table, pb_fk, pb_target = present_bridge
        present_join = f"""
            INNER JOIN {pb_table} ON {pb_table}.{pb_fk} = {base_table}.id
            INNER JOIN {present_table} ON {present_table}.{present_fk} = {pb_table}.{pb_target}
        """
    else:
        present_join = f"INNER JOIN {present_table} ON {present_table}.{present_fk} = {base_table}.id"

    if missing_bridge:
        mb_table, mb_fk, mb_target = missing_bridge
        missing_join = f"""
            LEFT JOIN {mb_table} ON {mb_table}.{mb_fk} = {base_table}.id
            LEFT JOIN {missing_table} ON {missing_table}.{missing_fk} = {mb_table}.{mb_target}
        """
        missing_null_cond = f"{missing_table}.{missing_fk} IS NULL"
    else:
        missing_join = f"LEFT JOIN {missing_table} ON {missing_table}.{missing_fk} = {base_table}.id"
        missing_null_cond = f"{missing_table}.{missing_fk} IS NULL"

    query = f"""
    SELECT DISTINCT
        {base_table}.id AS {base_table.lower()}_id
    FROM {base_table}
    {present_join}
    {missing_join}
    {' '.join(joins) if joins else ''}
    WHERE {missing_null_cond}
    """

    if where_clauses:
        query += " AND " + " AND ".join(where_clauses)

    query += f" ORDER BY {base_table}.id ASC LIMIT :limit"
    params["limit"] = int(limit)

    return execute_query(conn, query, params)


# =========================================================
# 7) Experiments with analysis files but no linked results
# =========================================================

def find_experiments_with_analysis_but_no_results(
    conn: sqlite3.Connection,
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 500,
) -> pd.DataFrame:
    where_clauses, params, joins = build_query_context(
        main_table="Experiment",
        filters=filters,
        base_tables={"Experiment","Experiment_Analysis_Files_Link", "AnalysisFiles", "Result_Analysis_Files_Link"},
    )

    query = """
    SELECT DISTINCT
        Experiment.id AS experiment_id,
        Experiment.date,
        Experiment.replicate
    FROM Experiment
    INNER JOIN Experiment_Analysis_Files_Link eaf
        ON eaf.experiment_id = Experiment.id
    INNER JOIN AnalysisFiles af
        ON af.id = eaf.analysis_file_id
    LEFT JOIN Result_Analysis_Files_Link raf
        ON raf.analysis_file_id = af.id
    """

    if joins:
        query += " " + " ".join(joins)

    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses) + " AND raf.analysis_file_id IS NULL"
    else:
        query += " WHERE raf.analysis_file_id IS NULL"

    query += " ORDER BY Experiment.id ASC LIMIT :limit"
    params["limit"] = int(limit)

    return execute_query(conn, query, params)


# =========================================================
# 8) Analysis files with no linked results
# =========================================================

def find_analysis_files_without_results(
    conn: sqlite3.Connection,
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 500,
) -> pd.DataFrame:
    where_clauses, params, joins = build_query_context(
        main_table="Experiment",
        filters=filters,
        base_tables={"Experiment", "Experiment_Analysis_Files_Link", "AnalysisFiles"},
    )

    query = """
    SELECT
        af.id AS analysis_file_id,
        af.file_name,
        af.file_type,
        af.file_path
    FROM AnalysisFiles af
    JOIN Experiment_Analysis_Files_Link eaf ON eaf.analysis_file_id = af.id
    JOIN Experiment ON Experiment.id = eaf.experiment_id
    LEFT JOIN Result_Analysis_Files_Link raf ON raf.analysis_file_id = af.id
    LEFT JOIN Results r ON r.id = raf.result_id
    """

    if joins:
        query += " " + " ".join(joins)

    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses) + " AND r.id IS NULL"
    else:
        query += " WHERE r.id IS NULL"

    query += " ORDER BY af.id ASC LIMIT :limit"
    params["limit"] = int(limit)

    return execute_query(conn, query, params)


# =========================================================
# 9) Results with no linked analysis files
# =========================================================

def find_results_without_analysis_files(
    conn: sqlite3.Connection,
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 500,
) -> pd.DataFrame:
    where_clauses, params, joins = build_query_context(
        main_table="Experiment",
        filters=filters,
        base_tables={"Experiment", "Experiment_Analysis_Files_Link", "AnalysisFiles", "Result_Analysis_Files_Link", "Results"},
    )

    query = """
    SELECT
        r.id AS result_id,
        r.result_type,
        r.result_value,
        r.sample_size,
        r.standard_error
    FROM Results r
    LEFT JOIN Result_Analysis_Files_Link raf ON raf.result_id = r.id
    LEFT JOIN AnalysisFiles af ON af.id = raf.analysis_file_id
    """

    if joins:
        query += " " + " ".join(joins)

    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses) + " AND af.id IS NULL"
    else:
        query += " WHERE af.id IS NULL"

    query += " ORDER BY r.id ASC LIMIT :limit"
    params["limit"] = int(limit)

    return execute_query(conn, query, params)
