"""
Centralized module for deleting records from the database.
"""
import sqlite3
import logging
import pandas as pd
from queries import COLUMN_MAP, build_query_context, find_invalid_foreign_keys, find_near_duplicates_by_columns, find_experiments_missing_files, find_incomplete_linked_entities_generalized,find_invalid_categorical_values
from queries import logger as queries_logger  # import the queries logger instance
from datetime import datetime
from pathlib import Path



OUTPUT_DIR = "../data/deletion_previews"
# ---------------------------------------------------------------------
# Module logger for deletions
# ---------------------------------------------------------------------

logger = logging.getLogger("delete_records")
if not logger.handlers:
    handler = logging.FileHandler("../data/db_deletion.log")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    handler.setLevel(logging.DEBUG)      # ensures all messages pass 
    logger.propagate = False

    # 🔗 Share handler with queries logger so both log to the same file
    queries_logger.addHandler(handler)
    queries_logger.setLevel(logging.DEBUG)
    queries_logger.propagate = False
'''
# set up logging
logging.basicConfig(
    filename="../data/db_deletion.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
'''

# ---------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------
def execute_delete(db_path, query, params=None, dry_run=True, output_dir="../data/deletion_previews"):
    """Safely execute a DELETE query with optional dry-run mode."""
    logger.info("\n--- DELETE QUERY ---")
    logger.info(f"Executing DELETE query: {query} with params: {params} (dry_run={dry_run})")

    if params:
        logger.info(f"Params: {params}")
    # --- Preview matching rows before deletion ---
    preview_query = query.replace("DELETE", "SELECT *", 1)
    try:
        with sqlite3.connect(db_path) as conn:
            df_preview = pd.read_sql_query(preview_query, conn, params=params or {})
        n_preview = len(df_preview)
        logger.info(f"Preview: {n_preview} matching rows found for deletion.")
        
        if n_preview > 0:
            # Save to CSV with timestamp
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = Path(output_dir) / f"delete_preview_{timestamp}.csv"
            df_preview.to_csv(csv_path, index=False)
            logger.info(f"Preview data saved to {csv_path}")
            print(f"Preview saved: {csv_path} ({n_preview} rows)")
    except Exception as e:
        logger.error(f"Error generating preview: {e}", exc_info=True)
        print("Could not generate preview.")


    if dry_run:
        logger.info("Dry-run mode: no records deleted.")
        return 0

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(query, params or {})
        deleted = cur.rowcount
        conn.commit()

    
    logger.info(f"Deleted {deleted} rows.")
    return deleted

# ---------------------------------------------------------------------
# 1. Delete by filter (general-purpose)
# ---------------------------------------------------------------------
def delete_records_by_filter(db_path, table, filters=None, limit=None, dry_run=True):
    """
    Delete rows from a table matching given filters.

    Uses COLUMN_MAP + build_query_context to resolve joins and parameters.
    """
    filters = filters or {}
    where_clauses, params, joins = build_query_context(
        main_table=table,
        filters=filters,
        base_tables={table}
    )

    query = f"DELETE FROM {table}"
    if joins:
        query += " " + " ".join(joins)
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    if limit:
        query += f" LIMIT {limit}"

    print("Generated DELETE query:", query)  # --- IGNORE ---
    return execute_delete(db_path, query, params, dry_run)

# ---------------------------------------------------------------------
# 2. Delete orphaned records
# ---------------------------------------------------------------------
def delete_orphan_records(db_path, child_table, fk_column, parent_table,
                          finder=find_invalid_foreign_keys, filters=None,limit=None, dry_run=True):
    """
    Delete orphaned rows (children whose FK does not match any parent).
    Uses the QC finder function to identify records first.
    """
    df = finder(db_path, child_table=child_table, fk_column=fk_column, parent_table=parent_table, filters=filters)
    if df is None or df.empty:
        print(f"No orphaned rows found in {child_table}.")
        logger.info(f"No orphaned rows found in {child_table}.")
        return 0

    ids = df[f"{child_table}_id"].tolist()
    placeholders = ", ".join("?" * len(ids))
    query = f"DELETE FROM {child_table} WHERE id IN ({placeholders});"
    return execute_delete(db_path, query, ids, dry_run)

# ---------------------------------------------------------------------
# 3. Delete duplicate records
# ---------------------------------------------------------------------
def delete_duplicate_records(db_path, table, key_column="id",
                             finder=find_near_duplicates_by_columns, include_columns=None, exclude_columns=None, show_columns=None, filters= None, dry_run=True):
    """
    Delete duplicate rows based on finder output, keeping one per group.
    """
    df = finder(db_path, table=table, key_column=key_column, include_columns=include_columns, exclude_columns=exclude_columns, show_columns=show_columns, filters=filters)

    if df is None or df.empty:
        logger.info(f"No duplicate records found in {table}.")
        return 0
    logger.info("Duplicate records found saved to output directory as duplicate_records_preview.csv")
    df.to_csv(f"{OUTPUT_DIR}/duplicate_records_preview.csv", index=False)

    # Flatten duplicate ids
    ids_to_delete = []
    for _, row in df.iterrows():
        ids = row["ids"].split(",")
        if len(ids) > 1:
            ids_to_delete.extend(ids[1:])  # keep first, delete others

    if not ids_to_delete:
        logger.info(f"No duplicate records to delete in {table}.")
        return 0

    placeholders = ", ".join("?" * len(ids_to_delete))
    query = f"DELETE FROM {table} WHERE {key_column} IN ({placeholders});"
    return execute_delete(db_path, query, ids_to_delete, dry_run)



# ---------------------------------------------------------------------
# 4. Delete all records (e.g., for dev reset)
# ---------------------------------------------------------------------
def delete_all_records(db_path, table, confirm=False, dry_run=True):
    """Delete all rows from a table (with optional confirmation)."""
    if not confirm:
        print(f"Skipping: confirmation required to delete all from {table}.")
        logger.info(f"Skipping deletion of all records from {table}: confirmation not given.")
        return 0

    query = f"DELETE FROM {table};"
    return execute_delete(db_path, query, dry_run=dry_run)

# ---------------------------------------------------------------------
# 5. 

# ----------------------- GENERIC WRAPPER -----------------------

def delete_using_finder(db_path, finder_func, finder_kwargs=None, dry_run=True):
    """
    Generic deletion wrapper — runs a finder (e.g., find_orphan_children)
    and deletes the results.
    """
    finder_kwargs = finder_kwargs or {}
    df = finder_func(db_path, **finder_kwargs)
    if df is None or df.empty:
        logger.info("Finder returned no rows to delete.")
        return 0

    main_table = list(df.columns)[0].split("_")[0]
    key_col = list(df.columns)[0]
    ids = df[key_col].tolist()
    placeholders = ", ".join("?" * len(ids))
    query = f"DELETE FROM {main_table} WHERE id IN ({placeholders});"
    return execute_delete(db_path, query, ids, dry_run)


def delete_records_from_finder(db_path, finder_func, finder_kwargs=None, id_column=None, dry_run=True):
    df = finder_func(db_path, **(finder_kwargs or {}))
    if df is None or df.empty:
        logger.info(f"No records to delete from finder {finder_func.__name__}")
        return 0
    if id_column is None:
        id_column = [c for c in df.columns if c.endswith("_id")][0]
    ids = df[id_column].tolist()
    placeholders = ", ".join("?" * len(ids))
    table = id_column.split("_id")[0]
    delete_query = f"DELETE FROM {table} WHERE id IN ({placeholders});"

    return execute_delete(db_path, delete_query, ids, dry_run=dry_run)




if __name__ == "__main__":
    DB_PATH = "/Users/masoomeshafiee/Projects/data_organization/data-management-system-SQLite/db/Reyes_lab_data.db" # <-- change this
    # 1. Delete invalid experiments
    #deleted = delete_records_by_filter(DB_PATH, "Experiment", {"is_valid": "N"}, dry_run=True)
    #deleted = delete_orphan_records(DB_PATH, "TrackingFiles", "experiment_id", "Experiment", dry_run=True)
    #deleted= delete_orphan_records(DB_PATH, "Masks", "experiment_id", "Experiment", dry_run=True)
    #delete = delete_duplicate_records(DB_PATH, "Experiment", include_columns=["concentration_value", "date", "replicate"],show_columns=["*"], dry_run=True)
    #delete = delete_all_records(DB_PATH, "Experiment", confirm=True, dry_run=True)
    #deleted = delete_records_from_finder(DB_PATH, find_orphan_parents, finder_kwargs={"parent_table": "Experiment", "child_table": "Masks", "fk_column": "experiment_id"}, dry_run=True)
    #deleted = delete_records_from_finder(DB_PATH, find_experiments_missing_files, finder_kwargs={"file_types": ["tracking"]},dry_run=True)
    #deleted = delete_records_from_finder(DB_PATH,
    #find_incomplete_linked_entities_generalized,
    #finder_kwargs=dict(
    #base_table="Experiment", present_bridge=("AnalysisResultExperiments", "experiment_id", "analysis_result_id"), present_entity=("AnalysisResults", "id"), missing_bridge=("ExperimentAnalysisFiles", "experiment_id","analysis_file_id" ), missing_entity=("AnalysisFiles", "id"), filters={"is_valid": "Y"}, limit=50),
    #dry_run=True)
    #deleted = delete_records_from_finder(DB_PATH, find_invalid_categorical_values, finder_kwargs= dict(table="Masks", column="mask_type", allowed_values=["cell", "nucleus", "Nucleus-G1"],limit=100), dry_run=True)


    #orphan pareent and findi missing values , find_invalid_categorical_values



    #print(f"Deleted {deleted} invalid experiments.")


