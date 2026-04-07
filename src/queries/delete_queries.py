import sqlite3
from typing import Any, Iterable, Optional

import pandas as pd

from config import REFERENCE_PARENT_CONFIG
from queries.queries_utils import build_query_context, execute_query


# =========================================================
# Helpers
# =========================================================

def _normalize_ids(ids: Iterable[int | str]) -> list[int]:
    unique_ids = []
    seen = set()
    for x in ids:
        x = int(x)
        if x not in seen:
            seen.add(x)
            unique_ids.append(x)
    return unique_ids


def _make_in_clause(ids: list[int], prefix: str = "id") -> tuple[str, dict[str, Any]]:
    if not ids:
        raise ValueError("ids cannot be empty")

    params = {f"{prefix}_{i}": v for i, v in enumerate(ids)}
    placeholders = ", ".join(f":{prefix}_{i}" for i in range(len(ids)))
    return f"({placeholders})", params


def _execute_write(
    conn: sqlite3.Connection,
    query: str,
    params: Optional[dict[str, Any]] = None,
) -> int:
    cur = conn.execute(query, params or {})
    return cur.rowcount if cur.rowcount is not None else 0


def preview_delete_rows_by_ids(
    conn: sqlite3.Connection,
    *,
    table: str,
    ids: Iterable[int | str],
    id_column: str = "id",
) -> pd.DataFrame:
    ids = _normalize_ids(ids)
    if not ids:
        return pd.DataFrame()

    in_clause, params = _make_in_clause(ids)
    query = f"""
    SELECT *
    FROM {table}
    WHERE {id_column} IN {in_clause}
    ORDER BY {id_column}
    """
    return execute_query(conn, query, params)


# =========================================================
# Browse / filter based experiment selection
# =========================================================

def preview_experiments_by_filters(
    conn: sqlite3.Connection,
    *,
    filters: Optional[dict[str, Any]] = None,
    limit: int = 500,
) -> pd.DataFrame:
    where_clauses, params, joins = build_query_context(
        main_table="Experiment",
        filters=filters,
        base_tables={"Experiment"},
    )

    query = """
    SELECT DISTINCT
        Experiment.id AS experiment_id,
        Experiment.date,
        Experiment.replicate
    FROM Experiment
    """

    if joins:
        query += " " + " ".join(joins)
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)

    query += " ORDER BY Experiment.date DESC, Experiment.id DESC LIMIT :limit"
    params["limit"] = int(limit)

    return execute_query(conn, query, params)



# =========================================================
# M2M-aware experiment delete preview
# =========================================================

def preview_experiment_delete_summary(
    conn: sqlite3.Connection,
    experiment_ids: Iterable[int | str],
) -> pd.DataFrame:
    experiment_ids = _normalize_ids(experiment_ids)
    if not experiment_ids:
        return pd.DataFrame([{
            "experiment_count": 0,
            "raw_file_count": 0,
            "tracking_file_count": 0,
            "mask_count": 0,
            "experiment_analysis_links_removed": 0,
            "linked_analysis_file_count": 0,
            "analysis_files_deleted_count": 0,
            "analysis_files_preserved_count": 0,
            "result_analysis_links_removed": 0,
            "linked_result_count": 0,
            "results_deleted_count": 0,
            "results_preserved_count": 0,
        }])

    exp_in, exp_params = _make_in_clause(experiment_ids, prefix="exp")

    query = f"""
    WITH selected_experiments AS (
        SELECT id
        FROM Experiment
        WHERE id IN {exp_in}
    ),
    selected_analysis_files AS (
        SELECT DISTINCT eaf.analysis_file_id AS id
        FROM Experiment_Analysis_Files_Link eaf
        JOIN selected_experiments se ON se.id = eaf.experiment_id
    ),
    analysis_files_to_delete AS (
        SELECT saf.id
        FROM selected_analysis_files saf
        WHERE NOT EXISTS (
            SELECT 1
            FROM Experiment_Analysis_Files_Link eaf
            WHERE eaf.analysis_file_id = saf.id
              AND eaf.experiment_id NOT IN (SELECT id FROM selected_experiments)
        )
    ),
    analysis_files_preserved AS (
        SELECT saf.id
        FROM selected_analysis_files saf
        WHERE EXISTS (
            SELECT 1
            FROM Experiment_Analysis_Files_Link eaf
            WHERE eaf.analysis_file_id = saf.id
              AND eaf.experiment_id NOT IN (SELECT id FROM selected_experiments)
        )
    ),
    linked_results_from_deleted_analysis AS (
        SELECT DISTINCT raf.result_id AS id
        FROM Result_Analysis_Files_Link raf
        JOIN analysis_files_to_delete afd ON afd.id = raf.analysis_file_id
    ),
    results_to_delete AS (
        SELECT lr.id
        FROM linked_results_from_deleted_analysis lr
        WHERE NOT EXISTS (
            SELECT 1
            FROM Result_Analysis_Files_Link raf
            WHERE raf.result_id = lr.id
              AND raf.analysis_file_id NOT IN (SELECT id FROM analysis_files_to_delete)
        )
    ),
    results_preserved AS (
        SELECT lr.id
        FROM linked_results_from_deleted_analysis lr
        WHERE EXISTS (
            SELECT 1
            FROM Result_Analysis_Files_Link raf
            WHERE raf.result_id = lr.id
              AND raf.analysis_file_id NOT IN (SELECT id FROM analysis_files_to_delete)
        )
    )
    SELECT
        (SELECT COUNT(*) FROM selected_experiments) AS experiment_count,
        (SELECT COUNT(*) FROM RawFiles rf
            JOIN selected_experiments se ON se.id = rf.experiment_id) AS raw_file_count,
        (SELECT COUNT(*) FROM TrackingFiles tf
            JOIN selected_experiments se ON se.id = tf.experiment_id) AS tracking_file_count,
        (SELECT COUNT(*) FROM Masks m
            JOIN selected_experiments se ON se.id = m.experiment_id) AS mask_count,
        (SELECT COUNT(*) FROM Experiment_Analysis_Files_Link eaf
            JOIN selected_experiments se ON se.id = eaf.experiment_id) AS experiment_analysis_links_removed,
        (SELECT COUNT(*) FROM selected_analysis_files) AS linked_analysis_file_count,
        (SELECT COUNT(*) FROM analysis_files_to_delete) AS analysis_files_deleted_count,
        (SELECT COUNT(*) FROM analysis_files_preserved) AS analysis_files_preserved_count,
        (SELECT COUNT(*) FROM Result_Analysis_Files_Link raf
            JOIN analysis_files_to_delete afd ON afd.id = raf.analysis_file_id) AS result_analysis_links_removed,
        (SELECT COUNT(*) FROM linked_results_from_deleted_analysis) AS linked_result_count,
        (SELECT COUNT(*) FROM results_to_delete) AS results_deleted_count,
        (SELECT COUNT(*) FROM results_preserved) AS results_preserved_count
    """
    return execute_query(conn, query, exp_params)


def preview_experiment_delete_details(
    conn: sqlite3.Connection,
    experiment_ids: Iterable[int | str],
) -> dict[str, pd.DataFrame]:
    experiment_ids = _normalize_ids(experiment_ids)
    if not experiment_ids:
        return {
            "experiments": pd.DataFrame(),
            "raw_files": pd.DataFrame(),
            "tracking_files": pd.DataFrame(),
            "masks": pd.DataFrame(),
            "experiment_analysis_links": pd.DataFrame(),
            "analysis_files_to_delete": pd.DataFrame(),
            "analysis_files_preserved": pd.DataFrame(),
            "result_analysis_links_to_remove": pd.DataFrame(),
            "results_to_delete": pd.DataFrame(),
            "results_preserved": pd.DataFrame(),
        }

    exp_in, exp_params = _make_in_clause(experiment_ids, prefix="exp")

    experiments = execute_query(
        conn,
        f"SELECT * FROM Experiment WHERE id IN {exp_in} ORDER BY id",
        exp_params,
    )
    raw_files = execute_query(
        conn,
        f"SELECT * FROM RawFiles WHERE experiment_id IN {exp_in} ORDER BY id",
        exp_params,
    )
    tracking_files = execute_query(
        conn,
        f"SELECT * FROM TrackingFiles WHERE experiment_id IN {exp_in} ORDER BY id",
        exp_params,
    )
    masks = execute_query(
        conn,
        f"SELECT * FROM Masks WHERE experiment_id IN {exp_in} ORDER BY id",
        exp_params,
    )
    experiment_analysis_links = execute_query(
        conn,
        f"""
        SELECT *
        FROM Experiment_Analysis_Files_Link
        WHERE experiment_id IN {exp_in}
        ORDER BY experiment_id, analysis_file_id
        """,
        exp_params,
    )

    analysis_files_to_delete = execute_query(
        conn,
        f"""
        WITH selected_experiments AS (
            SELECT id FROM Experiment WHERE id IN {exp_in}
        ),
        selected_analysis_files AS (
            SELECT DISTINCT eaf.analysis_file_id AS id
            FROM Experiment_Analysis_Files_Link eaf
            JOIN selected_experiments se ON se.id = eaf.experiment_id
        )
        SELECT af.*
        FROM AnalysisFiles af
        JOIN selected_analysis_files saf ON saf.id = af.id
        WHERE NOT EXISTS (
            SELECT 1
            FROM Experiment_Analysis_Files_Link eaf
            WHERE eaf.analysis_file_id = af.id
              AND eaf.experiment_id NOT IN (SELECT id FROM selected_experiments)
        )
        ORDER BY af.id
        """,
        exp_params,
    )

    analysis_files_preserved = execute_query(
        conn,
        f"""
        WITH selected_experiments AS (
            SELECT id FROM Experiment WHERE id IN {exp_in}
        ),
        selected_analysis_files AS (
            SELECT DISTINCT eaf.analysis_file_id AS id
            FROM Experiment_Analysis_Files_Link eaf
            JOIN selected_experiments se ON se.id = eaf.experiment_id
        )
        SELECT af.*
        FROM AnalysisFiles af
        JOIN selected_analysis_files saf ON saf.id = af.id
        WHERE EXISTS (
            SELECT 1
            FROM Experiment_Analysis_Files_Link eaf
            WHERE eaf.analysis_file_id = af.id
              AND eaf.experiment_id NOT IN (SELECT id FROM selected_experiments)
        )
        ORDER BY af.id
        """,
        exp_params,
    )

    result_analysis_links_to_remove = execute_query(
        conn,
        f"""
        WITH selected_experiments AS (
            SELECT id FROM Experiment WHERE id IN {exp_in}
        ),
        selected_analysis_files AS (
            SELECT DISTINCT eaf.analysis_file_id AS id
            FROM Experiment_Analysis_Files_Link eaf
            JOIN selected_experiments se ON se.id = eaf.experiment_id
        ),
        analysis_files_to_delete AS (
            SELECT saf.id
            FROM selected_analysis_files saf
            WHERE NOT EXISTS (
                SELECT 1
                FROM Experiment_Analysis_Files_Link eaf
                WHERE eaf.analysis_file_id = saf.id
                  AND eaf.experiment_id NOT IN (SELECT id FROM selected_experiments)
            )
        )
        SELECT raf.*
        FROM Result_Analysis_Files_Link raf
        JOIN analysis_files_to_delete afd ON afd.id = raf.analysis_file_id
        ORDER BY raf.result_id, raf.analysis_file_id
        """,
        exp_params,
    )

    results_to_delete = execute_query(
        conn,
        f"""
        WITH selected_experiments AS (
            SELECT id FROM Experiment WHERE id IN {exp_in}
        ),
        selected_analysis_files AS (
            SELECT DISTINCT eaf.analysis_file_id AS id
            FROM Experiment_Analysis_Files_Link eaf
            JOIN selected_experiments se ON se.id = eaf.experiment_id
        ),
        analysis_files_to_delete AS (
            SELECT saf.id
            FROM selected_analysis_files saf
            WHERE NOT EXISTS (
                SELECT 1
                FROM Experiment_Analysis_Files_Link eaf
                WHERE eaf.analysis_file_id = saf.id
                  AND eaf.experiment_id NOT IN (SELECT id FROM selected_experiments)
            )
        ),
        linked_results AS (
            SELECT DISTINCT raf.result_id AS id
            FROM Result_Analysis_Files_Link raf
            JOIN analysis_files_to_delete afd ON afd.id = raf.analysis_file_id
        )
        SELECT r.*
        FROM Results r
        JOIN linked_results lr ON lr.id = r.id
        WHERE NOT EXISTS (
            SELECT 1
            FROM Result_Analysis_Files_Link raf
            WHERE raf.result_id = r.id
              AND raf.analysis_file_id NOT IN (SELECT id FROM analysis_files_to_delete)
        )
        ORDER BY r.id
        """,
        exp_params,
    )

    results_preserved = execute_query(
        conn,
        f"""
        WITH selected_experiments AS (
            SELECT id FROM Experiment WHERE id IN {exp_in}
        ),
        selected_analysis_files AS (
            SELECT DISTINCT eaf.analysis_file_id AS id
            FROM Experiment_Analysis_Files_Link eaf
            JOIN selected_experiments se ON se.id = eaf.experiment_id
        ),
        analysis_files_to_delete AS (
            SELECT saf.id
            FROM selected_analysis_files saf
            WHERE NOT EXISTS (
                SELECT 1
                FROM Experiment_Analysis_Files_Link eaf
                WHERE eaf.analysis_file_id = saf.id
                  AND eaf.experiment_id NOT IN (SELECT id FROM selected_experiments)
            )
        ),
        linked_results AS (
            SELECT DISTINCT raf.result_id AS id
            FROM Result_Analysis_Files_Link raf
            JOIN analysis_files_to_delete afd ON afd.id = raf.analysis_file_id
        )
        SELECT r.*
        FROM Results r
        JOIN linked_results lr ON lr.id = r.id
        WHERE EXISTS (
            SELECT 1
            FROM Result_Analysis_Files_Link raf
            WHERE raf.result_id = r.id
              AND raf.analysis_file_id NOT IN (SELECT id FROM analysis_files_to_delete)
        )
        ORDER BY r.id
        """,
        exp_params,
    )

    return {
        "experiments": experiments,
        "raw_files": raw_files,
        "tracking_files": tracking_files,
        "masks": masks,
        "experiment_analysis_links": experiment_analysis_links,
        "analysis_files_to_delete": analysis_files_to_delete,
        "analysis_files_preserved": analysis_files_preserved,
        "result_analysis_links_to_remove": result_analysis_links_to_remove,
        "results_to_delete": results_to_delete,
        "results_preserved": results_preserved,
    }


# =========================================================
# M2M-aware analysis file delete preview
# =========================================================

def preview_analysis_file_delete_summary(
    conn: sqlite3.Connection,
    analysis_file_ids: Iterable[int | str],
) -> pd.DataFrame:
    analysis_file_ids = _normalize_ids(analysis_file_ids)
    if not analysis_file_ids:
        return pd.DataFrame([{
            "analysis_file_count": 0,
            "experiment_analysis_links_removed": 0,
            "result_analysis_links_removed": 0,
            "linked_result_count": 0,
            "results_deleted_count": 0,
            "results_preserved_count": 0,
        }])

    af_in, af_params = _make_in_clause(analysis_file_ids, prefix="af")

    query = f"""
    WITH selected_analysis_files AS (
        SELECT id
        FROM AnalysisFiles
        WHERE id IN {af_in}
    ),
    linked_results AS (
        SELECT DISTINCT raf.result_id AS id
        FROM Result_Analysis_Files_Link raf
        JOIN selected_analysis_files saf ON saf.id = raf.analysis_file_id
    ),
    results_to_delete AS (
        SELECT lr.id
        FROM linked_results lr
        WHERE NOT EXISTS (
            SELECT 1
            FROM Result_Analysis_Files_Link raf
            WHERE raf.result_id = lr.id
              AND raf.analysis_file_id NOT IN (SELECT id FROM selected_analysis_files)
        )
    ),
    results_preserved AS (
        SELECT lr.id
        FROM linked_results lr
        WHERE EXISTS (
            SELECT 1
            FROM Result_Analysis_Files_Link raf
            WHERE raf.result_id = lr.id
              AND raf.analysis_file_id NOT IN (SELECT id FROM selected_analysis_files)
        )
    )
    SELECT
        (SELECT COUNT(*) FROM selected_analysis_files) AS analysis_file_count,
        (SELECT COUNT(*) FROM Experiment_Analysis_Files_Link eaf
            JOIN selected_analysis_files saf ON saf.id = eaf.analysis_file_id) AS experiment_analysis_links_removed,
        (SELECT COUNT(*) FROM Result_Analysis_Files_Link raf
            JOIN selected_analysis_files saf ON saf.id = raf.analysis_file_id) AS result_analysis_links_removed,
        (SELECT COUNT(*) FROM linked_results) AS linked_result_count,
        (SELECT COUNT(*) FROM results_to_delete) AS results_deleted_count,
        (SELECT COUNT(*) FROM results_preserved) AS results_preserved_count
    """
    return execute_query(conn, query, af_params)

# =========================================================
# Leaf / near-leaf delete previews
# =========================================================

def preview_leaf_delete_summary(
    conn: sqlite3.Connection,
    *,
    table: str,
    ids: Iterable[int | str],
) -> pd.DataFrame:
    """
    Generic one-row summary for true leaf tables that have no downstream links
    in the current delete policy.
    """
    normalized_ids = _normalize_ids(ids)
    if not normalized_ids:
        return pd.DataFrame([{
            "selected_count": 0,
        }])

    return pd.DataFrame([{
        "selected_count": len(normalized_ids),
    }])


def preview_result_delete_summary(
    conn: sqlite3.Connection,
    result_ids: Iterable[int | str],
) -> pd.DataFrame:
    """
    Preview summary for deleting results.
    Since deleting results also deletes rows from Result_Analysis_Files_Link,
    include both counts.
    """
    normalized_ids = _normalize_ids(result_ids)
    if not normalized_ids:
        return pd.DataFrame([{
            "result_count": 0,
            "result_analysis_link_count": 0,
        }])

    result_in, result_params = _make_in_clause(normalized_ids, prefix="res")

    query = f"""
    SELECT
        (SELECT COUNT(*) FROM Results WHERE id IN {result_in}) AS result_count,
        (
            SELECT COUNT(*)
            FROM Result_Analysis_Files_Link
            WHERE result_id IN {result_in}
        ) AS result_analysis_link_count
    """
    return execute_query(conn, query, result_params)


def preview_result_delete_details(
    conn: sqlite3.Connection,
    result_ids: Iterable[int | str],
) -> dict[str, pd.DataFrame]:
    normalized_ids = _normalize_ids(result_ids)
    if not normalized_ids:
        return {
            "results": pd.DataFrame(),
            "result_analysis_links": pd.DataFrame(),
        }

    result_in, result_params = _make_in_clause(normalized_ids, prefix="res")

    results_df = execute_query(
        conn,
        f"""
        SELECT *
        FROM Results
        WHERE id IN {result_in}
        ORDER BY id
        """,
        result_params,
    )

    result_links_df = execute_query(
        conn,
        f"""
        SELECT *
        FROM Result_Analysis_Files_Link
        WHERE result_id IN {result_in}
        ORDER BY result_id, analysis_file_id
        """,
        result_params,
    )

    return {
        "results": results_df,
        "result_analysis_links": result_links_df,
    }
# =========================================================
# Leaf deletions
# =========================================================

def delete_raw_files_by_ids(conn: sqlite3.Connection, raw_file_ids: Iterable[int | str]) -> int:
    ids = _normalize_ids(raw_file_ids)
    if not ids:
        return 0
    in_clause, params = _make_in_clause(ids)
    return _execute_write(conn, f"DELETE FROM RawFiles WHERE id IN {in_clause}", params)


def delete_tracking_files_by_ids(conn: sqlite3.Connection, tracking_file_ids: Iterable[int | str]) -> int:
    ids = _normalize_ids(tracking_file_ids)
    if not ids:
        return 0
    in_clause, params = _make_in_clause(ids)
    return _execute_write(conn, f"DELETE FROM TrackingFiles WHERE id IN {in_clause}", params)


def delete_masks_by_ids(conn: sqlite3.Connection, mask_ids: Iterable[int | str]) -> int:
    ids = _normalize_ids(mask_ids)
    if not ids:
        return 0
    in_clause, params = _make_in_clause(ids)
    return _execute_write(conn, f"DELETE FROM Masks WHERE id IN {in_clause}", params)


def delete_results_by_ids(conn: sqlite3.Connection, result_ids: Iterable[int | str]) -> dict[str, int]:
    ids = _normalize_ids(result_ids)
    if not ids:
        return {"result_analysis_links_deleted": 0, "results_deleted": 0}

    in_clause, params = _make_in_clause(ids)
    link_count = _execute_write(conn, f"DELETE FROM Result_Analysis_Files_Link WHERE result_id IN {in_clause}", params)
    result_count = _execute_write(conn, f"DELETE FROM Results WHERE id IN {in_clause}", params)

    return {
        "result_analysis_links_deleted": link_count,
        "results_deleted": result_count,
    }


# =========================================================
# M2M-aware cascade delete: analysis files
# =========================================================

def delete_analysis_files_cascade(
    conn: sqlite3.Connection,
    analysis_file_ids: Iterable[int | str],
) -> dict[str, int]:
    analysis_file_ids = _normalize_ids(analysis_file_ids)
    if not analysis_file_ids:
        return {
            "experiment_analysis_links_deleted": 0,
            "result_analysis_links_deleted": 0,
            "results_deleted": 0,
            "analysis_files_deleted": 0,
        }

    af_in, af_params = _make_in_clause(analysis_file_ids, prefix="af")

    results_to_delete_df = execute_query(
        conn,
        f"""
        WITH selected_analysis_files AS (
            SELECT id
            FROM AnalysisFiles
            WHERE id IN {af_in}
        ),
        linked_results AS (
            SELECT DISTINCT raf.result_id AS id
            FROM Result_Analysis_Files_Link raf
            JOIN selected_analysis_files saf ON saf.id = raf.analysis_file_id
        )
        SELECT lr.id
        FROM linked_results lr
        WHERE NOT EXISTS (
            SELECT 1
            FROM Result_Analysis_Files_Link raf
            WHERE raf.result_id = lr.id
              AND raf.analysis_file_id NOT IN (SELECT id FROM selected_analysis_files)
        )
        """,
        af_params,
    )
    results_to_delete = _normalize_ids(results_to_delete_df["id"].tolist()) if not results_to_delete_df.empty else []

    result_link_count = _execute_write(
        conn,
        f"DELETE FROM Result_Analysis_Files_Link WHERE analysis_file_id IN {af_in}",
        af_params,
    )
    experiment_link_count = _execute_write(
        conn,
        f"DELETE FROM Experiment_Analysis_Files_Link WHERE analysis_file_id IN {af_in}",
        af_params,
    )

    results_deleted = 0
    if results_to_delete:
        r_in, r_params = _make_in_clause(results_to_delete, prefix="res")
        results_deleted = _execute_write(conn, f"DELETE FROM Results WHERE id IN {r_in}", r_params)

    analysis_deleted = _execute_write(
        conn,
        f"DELETE FROM AnalysisFiles WHERE id IN {af_in}",
        af_params,
    )

    return {
        "experiment_analysis_links_deleted": experiment_link_count,
        "result_analysis_links_deleted": result_link_count,
        "results_deleted": results_deleted,
        "analysis_files_deleted": analysis_deleted,
    }


# =========================================================
# M2M-aware cascade delete: experiments
# =========================================================

def delete_experiments_cascade(
    conn: sqlite3.Connection,
    experiment_ids: Iterable[int | str],
) -> dict[str, int]:
    experiment_ids = _normalize_ids(experiment_ids)
    if not experiment_ids:
        return {
            "experiment_analysis_links_deleted": 0,
            "result_analysis_links_deleted": 0,
            "raw_files_deleted": 0,
            "tracking_files_deleted": 0,
            "masks_deleted": 0,
            "results_deleted": 0,
            "analysis_files_deleted": 0,
            "experiments_deleted": 0,
        }

    exp_in, exp_params = _make_in_clause(experiment_ids, prefix="exp")

    analysis_files_to_delete_df = execute_query(
        conn,
        f"""
        WITH selected_experiments AS (
            SELECT id FROM Experiment WHERE id IN {exp_in}
        ),
        selected_analysis_files AS (
            SELECT DISTINCT eaf.analysis_file_id AS id
            FROM Experiment_Analysis_Files_Link eaf
            JOIN selected_experiments se ON se.id = eaf.experiment_id
        )
        SELECT saf.id
        FROM selected_analysis_files saf
        WHERE NOT EXISTS (
            SELECT 1
            FROM Experiment_Analysis_Files_Link eaf
            WHERE eaf.analysis_file_id = saf.id
              AND eaf.experiment_id NOT IN (SELECT id FROM selected_experiments)
        )
        """,
        exp_params,
    )
    analysis_files_to_delete = (
        _normalize_ids(analysis_files_to_delete_df["id"].tolist())
        if not analysis_files_to_delete_df.empty else []
    )

    results_to_delete: list[int] = []
    result_link_count = 0

    if analysis_files_to_delete:
        af_in, af_params = _make_in_clause(analysis_files_to_delete, prefix="af")

        results_to_delete_df = execute_query(
            conn,
            f"""
            WITH selected_analysis_files AS (
                SELECT id FROM AnalysisFiles WHERE id IN {af_in}
            ),
            linked_results AS (
                SELECT DISTINCT raf.result_id AS id
                FROM Result_Analysis_Files_Link raf
                JOIN selected_analysis_files saf ON saf.id = raf.analysis_file_id
            )
            SELECT lr.id
            FROM linked_results lr
            WHERE NOT EXISTS (
                SELECT 1
                FROM Result_Analysis_Files_Link raf
                WHERE raf.result_id = lr.id
                  AND raf.analysis_file_id NOT IN (SELECT id FROM selected_analysis_files)
            )
            """,
            af_params,
        )
        results_to_delete = (
            _normalize_ids(results_to_delete_df["id"].tolist())
            if not results_to_delete_df.empty else []
        )

        result_link_count = _execute_write(
            conn,
            f"DELETE FROM Result_Analysis_Files_Link WHERE analysis_file_id IN {af_in}",
            af_params,
        )

    experiment_link_count = _execute_write(
        conn,
        f"DELETE FROM Experiment_Analysis_Files_Link WHERE experiment_id IN {exp_in}",
        exp_params,
    )

    raw_count = _execute_write(conn, f"DELETE FROM RawFiles WHERE experiment_id IN {exp_in}", exp_params)
    tracking_count = _execute_write(conn, f"DELETE FROM TrackingFiles WHERE experiment_id IN {exp_in}", exp_params)
    mask_count = _execute_write(conn, f"DELETE FROM Masks WHERE experiment_id IN {exp_in}", exp_params)

    results_deleted = 0
    if results_to_delete:
        r_in, r_params = _make_in_clause(results_to_delete, prefix="res")
        results_deleted = _execute_write(conn, f"DELETE FROM Results WHERE id IN {r_in}", r_params)

    analysis_deleted = 0
    if analysis_files_to_delete:
        af_in, af_params = _make_in_clause(analysis_files_to_delete, prefix="af")
        analysis_deleted = _execute_write(conn, f"DELETE FROM AnalysisFiles WHERE id IN {af_in}", af_params)

    experiment_count = _execute_write(conn, f"DELETE FROM Experiment WHERE id IN {exp_in}", exp_params)

    return {
        "experiment_analysis_links_deleted": experiment_link_count,
        "result_analysis_links_deleted": result_link_count,
        "raw_files_deleted": raw_count,
        "tracking_files_deleted": tracking_count,
        "masks_deleted": mask_count,
        "results_deleted": results_deleted,
        "analysis_files_deleted": analysis_deleted,
        "experiments_deleted": experiment_count,
    }


# =========================================================
# Restrictive delete for parent/reference tables
# =========================================================

def preview_reference_entity_delete(
    conn: sqlite3.Connection,
    *,
    parent_table: str,
    parent_ids: Iterable[int | str],
) -> dict[str, pd.DataFrame]:
    if parent_table not in REFERENCE_PARENT_CONFIG:
        raise ValueError(f"Unsupported reference table: {parent_table}")

    parent_ids = _normalize_ids(parent_ids)
    if not parent_ids:
        return {
            "parents": pd.DataFrame(),
            "referencing_experiments": pd.DataFrame(),
            "summary": pd.DataFrame([{
                "selected_parent_count": 0,
                "referencing_experiment_count": 0,
                "deletable_parent_count": 0,
                "blocked_parent_count": 0,
            }]),
        }

    fk_col = REFERENCE_PARENT_CONFIG[parent_table]
    p_in, p_params = _make_in_clause(parent_ids, prefix="parent")

    parents = execute_query(
        conn,
        f"SELECT * FROM {parent_table} WHERE id IN {p_in} ORDER BY id",
        p_params,
    )

    referencing_experiments = execute_query(
        conn,
        f"""
        SELECT Experiment.*
        FROM Experiment
        WHERE Experiment.{fk_col} IN {p_in}
        ORDER BY Experiment.id
        """,
        p_params,
    )

    summary = execute_query(
        conn,
        f"""
        WITH selected_parents AS (
            SELECT id
            FROM {parent_table}
            WHERE id IN {p_in}
        ),
        blocked_parents AS (
            SELECT DISTINCT sp.id
            FROM selected_parents sp
            JOIN Experiment e ON e.{fk_col} = sp.id
        )
        SELECT
            (SELECT COUNT(*) FROM selected_parents) AS selected_parent_count,
            (SELECT COUNT(*) FROM Experiment e WHERE e.{fk_col} IN {p_in}) AS referencing_experiment_count,
            (
                SELECT COUNT(*)
                FROM selected_parents sp
                WHERE sp.id NOT IN (SELECT id FROM blocked_parents)
            ) AS deletable_parent_count,
            (SELECT COUNT(*) FROM blocked_parents) AS blocked_parent_count
        """,
        p_params,
    )

    return {
        "parents": parents,
        "referencing_experiments": referencing_experiments,
        "summary": summary,
    }


def delete_reference_entities_restrict(
    conn: sqlite3.Connection,
    *,
    parent_table: str,
    parent_ids: Iterable[int | str],
) -> dict[str, int]:
    if parent_table not in REFERENCE_PARENT_CONFIG:
        raise ValueError(f"Unsupported reference table: {parent_table}")

    parent_ids = _normalize_ids(parent_ids)
    if not parent_ids:
        return {"deleted": 0, "blocked": 0}

    fk_col = REFERENCE_PARENT_CONFIG[parent_table]
    p_in, p_params = _make_in_clause(parent_ids, prefix="parent")

    deletable_df = execute_query(
        conn,
        f"""
        SELECT p.id
        FROM {parent_table} p
        WHERE p.id IN {p_in}
          AND NOT EXISTS (
              SELECT 1
              FROM Experiment e
              WHERE e.{fk_col} = p.id
          )
        """,
        p_params,
    )
    deletable_ids = _normalize_ids(deletable_df["id"].tolist()) if not deletable_df.empty else []

    if not deletable_ids:
        return {"deleted": 0, "blocked": len(parent_ids)}

    d_in, d_params = _make_in_clause(deletable_ids, prefix="delp")
    deleted = _execute_write(conn, f"DELETE FROM {parent_table} WHERE id IN {d_in}", d_params)

    return {
        "deleted": deleted,
        "blocked": len(parent_ids) - deleted,
    }


# =========================================================
# Convenience
# =========================================================

def select_experiment_ids_from_dataframe(df: pd.DataFrame, id_column: str = "experiment_id") -> list[int]:
    if df.empty:
        return []
    if id_column not in df.columns:
        raise ValueError(f"Column '{id_column}' not found in DataFrame.")
    return _normalize_ids(df[id_column].tolist())


def select_ids_from_dataframe(df: pd.DataFrame, id_column: str = "id") -> list[int]:
    if df.empty:
        return []
    if id_column not in df.columns:
        raise ValueError(f"Column '{id_column}' not found in DataFrame.")
    return _normalize_ids(df[id_column].tolist())