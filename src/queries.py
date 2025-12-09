import sqlite3
import pandas as pd
import logging
from collections import deque


# ---------------------------------------------------------------------
# Module logger for queries
# ---------------------------------------------------------------------
logger = logging.getLogger("queries")
if not logger.handlers:
    handler = logging.FileHandler("/Users/masoomeshafiee/Desktop/Presentation/db_export.log")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    handler.setLevel(logging.DEBUG)      # ensures all messages pass
    logger.propagate = False# stay isolated from the root logger
'''
logging.basicConfig(
    filename="../data/db_export2.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
'''

BASE_EXPERIMENT_QUERY = """
    SELECT Experiment.id, Organism.organism_name as organism, Protein.protein_name as protein, StrainOrCellLine.strain_name as strain, Condition.condition_name as condition, Condition.concentration_value, Condition.concentration_unit,
            CaptureSetting.capture_type, CaptureSetting.exposure_time, CaptureSetting.time_interval, User.user_name as user, Experiment.date, Experiment.replicate, Experiment.is_valid, Experiment.comment, Experiment.experiment_path
    FROM Experiment Experiment
    JOIN Organism ON Experiment.organism_id = Organism.id
    JOIN Protein ON Experiment.protein_id = Protein.id
    JOIN StrainOrCellLine ON Experiment.strain_id = STRainOrCellLine.id
    JOIN Condition On Experiment.condition_id = Condition.id
    JOIN CaptureSetting ON Experiment.capture_setting_id = CaptureSetting.id
    JOIN User ON Experiment.user_id = User.id
"""

COLUMN_MAP = {
    "organism": "Organism.organism_name",
    "protein": "Protein.protein_name",
    "strain": "StrainOrCellLine.strain_name",
    "condition": "Condition.condition_name",
    "user_name": "User.user_name",
    "concentration_value": "Condition.concentration_value",
    "concentration_unit": "Condition.concentration_unit",
    "capture_setting_id": "CaptureSetting.id",
    "dye_concentration_value": "CaptureSetting.dye_concentration_value",
    "capture_type": "CaptureSetting.capture_type",
    "exposure_time": "CaptureSetting.exposure_time",
    "time_interval": "CaptureSetting.time_interval",
    "is_valid": "Experiment.is_valid",
    "date": "Experiment.date",
    "replicate": "Experiment.replicate",
    "experiment_id": "Experiment.id",
    "raw_file_id": "RawFiles.id",
    "raw_file_name": "RawFiles.file_name",
    "tracking_file_id": "TrackingFiles.id",
    "mask_id": "Masks.id",
    "analysis_file_id": "AnalysisFiles.id",
    "analysis_result_id": "AnalysisResults.id",
    "raw_file_type": "RawFiles.file_type",
    "mask_type": "Masks.mask_type",
    "mask_file_type": "Masks.file_type",
    "analysis_file_type": "AnalysisFiles.file_type",
    "analysis_result_type": "AnalysisResults.result_type",
    "comment": "Experiment.comment",
    "email": "User.email",
}
LIST_OF_TABLES = { "Organism", "Protein", "StrainOrCellLine", "Condition", "User", "CaptureSetting", "Experiment", "AnalysisFiles", "AnalysisResultExperiments"
                  , "AnylysisResults", "ExperimentAnalysisFiles", "Masks", "RawFiles", "TrackingFiles"}

TABLE_RELATIONSHIPS = { "Experiment": {
        "organism_id": "Organism",
        "protein_id": "Protein",
        "strain_id": "StrainOrCellLine",
        "condition_id": "Condition",
        "user_id": "User",
        "capture_setting_id": "CaptureSetting"
    },
    "ExperimentAnalysisFiles": {"experiment_id": "Experiment", "analysis_file_id": "AnalysisFiles"},
    "AnalysisResultExperiments": {"experiment_id": "Experiment", "analysis_result_id": "AnalysisResults"},
    "Masks": {"experiment_id": "Experiment"},
    "RawFiles": {"experiment_id": "Experiment"},
    "TrackingFiles": {"experiment_id": "Experiment"}}


#  ------------- helper functions ---------------

def get_where_clause_for_filters(filters):
    where_clauses = []
    params = {}
    for key, value in filters.items():
        if key not in COLUMN_MAP:
            raise ValueError(f"Unknown filter key: {key}")
        col = COLUMN_MAP[key]
        if isinstance(value, str):
            where_clauses.append(f"{col} = :{key} COLLATE NOCASE")
        else:
            where_clauses.append(f"{col} = :{key}")
        params[key] = value
    return where_clauses, params

def infer_joins(requested_tables, main_table = "Experiment"):
    joins = []
    visited = {main_table}
    to_visit = [main_table]

    while to_visit:
        current = to_visit.pop()
        # forward relationships
        if current in TABLE_RELATIONSHIPS:
            for fk, target in TABLE_RELATIONSHIPS[current].items():
                if target in requested_tables and target not in visited:
                    joins.append(f"JOIN {target} ON {current}.{fk} = {target}.id")
                    visited.add(target)
                    to_visit.append(target)

        # reverse relationships
        for table, rels in TABLE_RELATIONSHIPS.items():
            for fk, target in rels.items():
                if target == current and table in requested_tables and table not in visited:
                    joins.append(f"JOIN {table} ON {table}.{fk} = {current}.id")
                    visited.add(table)
                    to_visit.append(table)
    return joins

# ---------- BFS style join inference --------------
# ---------- Build a bidirectional join graph ----------

def build_join_graph():
    """
    From TABLE_RELATIONSHIPS like:
      "Experiment": {"organism_id": "Organism", ...}
    build a graph where each directed edge carries the exact JOIN SQL.
    """
    graph = {}  # {table: List[(neighbor_table, join_sql)]}

    def add_edge(frm, to, sql):
        graph.setdefault(frm, []).append((to, sql))

    for frm, rels in TABLE_RELATIONSHIPS.items():
        for fk_col, to in rels.items():
            # forward: frm -> to  (JOIN to ON frm.fk = to.id)
            fwd_sql = f"JOIN {to} ON {frm}.{fk_col} = {to}.id"
            add_edge(frm, to, fwd_sql)

            # reverse: to -> frm  (JOIN frm ON frm.fk = to.id)
            rev_sql = f"JOIN {frm} ON {frm}.{fk_col} = {to}.id"
            add_edge(to, frm, rev_sql)

    return graph

_JOIN_GRAPH = None  # cache

def _ensure_graph():
    global _JOIN_GRAPH
    if _JOIN_GRAPH is None:
        _JOIN_GRAPH = build_join_graph()
    return _JOIN_GRAPH

# ---------- BFS to get a path of JOINs ----------

def _bfs_path_joins(main_table, target_table):
    """
    Return the list of JOIN clauses to get from main_table to target_table.
    If already the same table, returns [].
    """
    if main_table == target_table:
        return []

    graph = _ensure_graph()

    q = deque([main_table])
    prev = {main_table: None}          # node -> previous node
    via_join = {main_table: None}      # node -> JOIN sql used to reach this node

    while q:
        cur = q.popleft()
        for nxt, join_sql in graph.get(cur, []):
            if nxt in prev:  # visited
                continue
            prev[nxt] = cur
            via_join[nxt] = join_sql
            if nxt == target_table:
                # reconstruct path of JOIN sqls
                path = []
                node = nxt
                while node != main_table:
                    path.append(via_join[node])
                    node = prev[node]
                path.reverse()
                return path
            q.append(nxt)

    # no path
    return None

def infer_joins_bfs(requested_tables, main_table, base_tables=None):
    """
    Compute the JOINs needed to connect main_table to all requested_tables.
    base_tables: tables already present in the base SELECT (so we should not re-join them).
    """
    base_tables = set(base_tables or [])
    requested = set(requested_tables or [])
    requested.discard(main_table)

    # Helper to extract the table being joined from a JOIN clause
    def joined_table(join_sql):
        # Format is: "JOIN <TableName> ON ..."
        return join_sql.split()[1]

    seen = set()   # dedupe join sql
    joins = []

    for target in sorted(requested):  # stable order
        path = _bfs_path_joins(main_table, target)
        if path is None:
            raise ValueError(f"No join path from {main_table} to {target}. Update TABLE_RELATIONSHIPS.")
        for j in path:
            jt = joined_table(j)
            # Skip if that table is already in the base query
            if jt in base_tables:
                continue
            if j not in seen:
                seen.add(j)
                joins.append(j)

    return joins


def build_query_context_1(filters=None, extra_where=None, extra_params=None, main_table="Experiment", required_tables_extra=None):
    """
    Build WHERE clauses, params, and JOINs for a query.
    
    Args:
        filters (dict): user filters (mapped through COLUMN_MAP).
        extra_where (list[str]): extra WHERE conditions (raw SQL).
        extra_params (dict): params for the extra WHERE conditions.
        main_table (str): usually "Experiment", can be overridden.
    
    Returns:
        (where_clauses, params, joins)
    """
    where_clauses = extra_where[:] if extra_where else []
    params = dict(extra_params) if extra_params else {}
    required_tables = set()
    if required_tables_extra:
        required_tables.update(required_tables_extra)

    if filters:
        filter_clauses, filter_params = get_where_clause_for_filters(filters)
        where_clauses.extend(filter_clauses)
        params.update(filter_params)

        for key in filters.keys():
            if key in COLUMN_MAP:
                tbl = COLUMN_MAP[key].split(".")[0]  # e.g. "Protein.name" -> "Protein"
                required_tables.add(tbl)

    joins = infer_joins_bfs(required_tables, main_table = main_table)

    return where_clauses, params, joins

def build_query_context(main_table,
                        filters=None,
                        extra_where=None,
                        extra_params=None,
                        base_tables=None,
                        required_tables_extra=None):
    """
    Build WHERE clauses, params, and JOIN clauses dynamically.

    Args:
        main_table (str): starting table (e.g., "Experiment")
        filters (dict): filter conditions
        extra_where (list[str]): additional WHERE conditions
        extra_params (dict): additional parameters
        base_tables (set[str]): tables already present in the base query
        required_tables_extra (set[str]): extra tables to force-include
    """
    where_clauses = []
    params = {}
    required_tables = set()

    # Handle filters
    if filters:
        filter_clauses, filter_params = get_where_clause_for_filters(filters)
        where_clauses.extend(filter_clauses)
        params.update(filter_params)

        for key in filters.keys():
            if key in COLUMN_MAP:
                tbl = COLUMN_MAP[key].split(".")[0]
                if not base_tables or tbl not in base_tables:
                    required_tables.add(tbl)

    # Add forced tables
    if required_tables_extra:
        required_tables.update(required_tables_extra)

    # Infer joins dynamically
    joins = infer_joins_bfs(required_tables, main_table = main_table)

    # Extra WHERE and params
    if extra_where:
        where_clauses.extend(extra_where)
    if extra_params:
        params.update(extra_params)

    return where_clauses, params, joins




def execute_query(db_path, query, params=None):
    try:
        conn = sqlite3.connect(db_path)
        if params:
            result_df = pd.read_sql_query(query, conn, params=params)
        else:
            result_df = pd.read_sql_query(query, conn)
        print("i was here")
        return result_df
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}", exc_info=True)
        print("Database error:", e)  # --- IGNORE ---
        for h in logger.handlers:
            h.flush()  # ← force flush so it writes before return
        return None
    finally:
        if conn:
            conn.close()


# Possible queries

# --------- Meta-data retrieval queries ---------

# 1. Get metadata for a given experiment ID
def get_experiment_metadata(db_path, experiment_id):
    query = BASE_EXPERIMENT_QUERY + "WHERE Experiment.id = ?;"
    return execute_query(db_path, query, (experiment_id,))

# 2. Experiment lookup
def list_experiments(db_path, filters = None, limit = 10):
    filters = filters or {}
    base_query = BASE_EXPERIMENT_QUERY
    where_clauses = []
    params = {}

    for key, value in filters.items():
        if key not in COLUMN_MAP:
            raise ValueError(f"Unknown filter key: {key}")
        col = COLUMN_MAP[key]
        if value is not None:
            if isinstance(value, str):
                where_clauses.append(f"{col} = :{key} COLLATE NOCASE")
            else:
                where_clauses.append(f"{col} = :{key}")
            params[key] = value
    if where_clauses:
        base_query += " WHERE " + " AND ".join(where_clauses)
    
    if limit:
        base_query += f" LIMIT {limit}"
    return execute_query(db_path, base_query + ";", params)

# 3. Entity-specific lookups

def expand_columns(conn, table):
    cursor = conn.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cursor.fetchall()]  # second col is column name


def list_entity(db_path, requested_columns, main_table="Experiment", filters=None, limit=10):
    filters = filters or {}
    
    sql_columns = []
    requested_tables = {main_table}
    
    # columns requested in SELECT
    if requested_columns == ["*"]:
        with sqlite3.connect(db_path) as conn:
            all_cols = expand_columns(conn, main_table)
        sql_columns = [f"{main_table}.{c}" for c in all_cols]
        # all columns already from main_table → nothing extra to add
    
    else:
        for col in requested_columns:
            if col not in COLUMN_MAP:
                raise ValueError(f"Unknown column: {col}")
            sql_col = COLUMN_MAP[col]
            sql_columns.append(sql_col)
            # extract table name for join inference
            requested_tables.add(sql_col.split(".")[0])
    
    # ALSO add tables referenced in filters
    for key in filters.keys():
        if key not in COLUMN_MAP:
            raise ValueError(f"Unknown filter key: {key}")
        col = COLUMN_MAP[key]
        requested_tables.add(col.split(".")[0])
    
    # infer joins
    joins = infer_joins_bfs(requested_tables, main_table = main_table)
    base_query = f"SELECT DISTINCT {', '.join(sql_columns)} FROM {main_table} " + " ".join(joins)
    
    # WHERE clause
    where_clauses, params = get_where_clause_for_filters(filters)
    if where_clauses:
        base_query += " WHERE " + " AND ".join(where_clauses)
    
    if limit:
        base_query += f" LIMIT {limit}"
    
    return execute_query(db_path, base_query + ";", params)

# ------------- Time-based queries -------------

# 1. list experiments by date range

def list_experiments_between_dates(db_path, start_date, end_date, filters=None, limit=10):
    extra_where = ["Experiment.date BETWEEN :start_date AND :end_date"]
    extra_params = {"start_date": start_date, "end_date": end_date}
    # not ideal to hardcode base_tables here, but it avoids unnecessary joins
    where_clauses, params, joins = build_query_context("Experiment", filters, extra_where, extra_params, base_tables={"Experiment", "Organism", "Protein", "StrainOrCellLine", "Condition", "CaptureSetting", "User"})

    query = BASE_EXPERIMENT_QUERY
    if joins:
        query += " " + " ".join(joins)
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    query += " ORDER BY Experiment.date ASC"
    if limit:
        query += f" LIMIT {limit}"

    return execute_query(db_path, query, params)

# 2. count experiments per year or month
def count_experiments_by_period(db_path, period="year", filters=None):
    if period == "year":
        date_expr = "substr(Experiment.date, 1, 4)"
    elif period == "month":
        date_expr = "substr(Experiment.date, 1, 7)"
    else:
        raise ValueError("Period must be 'year' or 'month'")

    where_clauses, params, joins = build_query_context("Experiment", filters)

    query = f"SELECT {date_expr} AS period, COUNT(DISTINCT Experiment.id) AS experiment_count FROM Experiment"
    if joins:
        query += " " + " ".join(joins)
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    query += " GROUP BY period ORDER BY period ASC"

    return execute_query(db_path, query, params)


# 3. Find the most recent experiment for a given entity
def find_most_recent_experiment(db_path, filters=None):
    """
    Find the most recent experiment matching the given filters.
    Filters can include multiple entities (e.g., protein, condition, capture_type, etc.).
    """
    # not ideal to hardcode base_tables here, but it avoids unnecessary joins
    where_clauses, params, joins = build_query_context("Experiment", filters, base_tables={"Experiment", "Organism", "Protein", "StrainOrCellLine", "Condition", "CaptureSetting", "User"})

    query = BASE_EXPERIMENT_QUERY
    if joins:
        query += " " + " ".join(joins)
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    query += " ORDER BY Experiment.date DESC LIMIT 1"

    return execute_query(db_path, query, params)

# 4. Find all experiments run in a specific year/month

def list_experiments_in_period(db_path, year=None, month=None, filters=None, limit=10):
    extra_where = []
    extra_params = {}
    if year:
        extra_where.append("substr(Experiment.date, 1, 4) = :year")
        extra_params["year"] = str(year)
    if month:
        extra_where.append("substr(Experiment.date, 5, 2) = :month")
        extra_params["month"] = f"{int(month):02d}"

    # not ideal to hardcode base_tables here, but it avoids unnecessary joins
    where_clauses, params, joins = build_query_context("Experiment", filters, extra_where, extra_params, base_tables={"Experiment", "Organism", "Protein", "StrainOrCellLine", "Condition", "CaptureSetting", "User"})

    query = BASE_EXPERIMENT_QUERY
    if joins:
        query += " " + " ".join(joins)
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    query += " ORDER BY Experiment.date ASC"
    if limit:
        query += f" LIMIT {limit}"

    return execute_query(db_path, query, params)

# 5. Earliest experiment overall or per entity
def find_earliest_experiment(db_path, filters=None):
    # not ideal to hardcode base_tables here, but it avoids unnecessary joins
    where_clauses, params, joins = build_query_context("Experiment", filters, base_tables={"Experiment", "Organism", "Protein", "StrainOrCellLine", "Condition", "CaptureSetting", "User"})

    query = BASE_EXPERIMENT_QUERY + " " + " ".join(joins)
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    query += " ORDER BY Experiment.date ASC LIMIT 1"

    return execute_query(db_path, query, params)

# 6. Trends over time (counts grouped by period + entity) | count experiments by enetity (also without time)
def count_experiments_trend(db_path, period= None, group_by="protein", filters=None):
    # choose time period
    if period is not None:
        if period == "year": 
            date_expr = "substr(Experiment.date, 1, 4)"       # YYYY
        elif period == "month":
            date_expr = "substr(Experiment.date, 1, 7)"       # YYYY-MM
        else:
            raise ValueError("Period must be 'year' or 'month'")

    group_cols = []
    group_tables = set()
    for col in group_by:
        if col not in COLUMN_MAP:
            raise ValueError(f"Unsupported group_by: {col}")
        group_cols.append(COLUMN_MAP[col])
        group_tables.add(COLUMN_MAP[col].split('.')[0])

    # Build joins
    where_clauses, params, joins = build_query_context("Experiment", filters, required_tables_extra=group_tables)

    # SELECT clause
    select_cols = [f"{col}" for col in group_cols]
    select_clause = ", ".join(select_cols)

    # build query
    if period is None:
        query = f"SELECT {select_clause}, COUNT(*) AS experiment_count FROM Experiment {' '.join(joins)}"
    else:
        query = f"SELECT {date_expr} AS period, {select_clause}, COUNT(*) AS experiment_count FROM Experiment {' '.join(joins)}"
        group_cols.insert(0, "period")  # ensure period is first in GROUP BY

    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)

    group_by_clause = ", ".join(group_cols)
    order_by_clause = ", ".join(group_cols)

    query += f" GROUP BY {group_by_clause} ORDER BY {order_by_clause} ASC"
    return execute_query(db_path, query, params)

# 7. Experiments within last N days
def list_recent_experiments(db_path, days=30, filters=None, limit=50):
    where_clauses, params, joins = build_query_context("Experiment", filters, base_tables={"Experiment", "Organism", "Protein", "StrainOrCellLine", "Condition", "CaptureSetting", "User"})

    # SQLite date math: date('now', '-30 days')
    where_clauses.append("substr(Experiment.date,1,4) || '-' || substr(Experiment.date,5,2) || '-' || substr(Experiment.date,7,2)>= date('now', :interval)")
    params["interval"] = f"-{int(days)} days"
    query = BASE_EXPERIMENT_QUERY + " " + " ".join(joins)
    query += " WHERE " + " AND ".join(where_clauses)
    query += " ORDER BY Experiment.date DESC"
    if limit:
        query += f" LIMIT {limit}"

    return execute_query(db_path, query, params)

# ----------- summary queries ------------

# 1. count experiments by entity --> use the time trend function (#6 function in the time-based queries) with period=None and group_by=[entity] 

# 2. count one entity by another ex: count proteins by organism, count strains by protein, count condition by strain etc.
def count_entity_by_another(db_path, entity, by_entities, filters=None):
    if entity == '*':
        main_table = COLUMN_MAP[by_entities[0]].split('.')[0]
    elif entity not in COLUMN_MAP:
        raise ValueError(f"Unsupported entity: {entity}")
    else: 
        main_table = COLUMN_MAP[entity].split('.')[0]
    group_cols = []
    group_tables = set()
    for by_entity in by_entities:
        if by_entity not in COLUMN_MAP:
            raise ValueError(f"Unsupported by_entity: {by_entity}")
        group_cols.append(COLUMN_MAP[by_entity])
        group_tables.add(COLUMN_MAP[by_entity].split('.')[0])

    # Build joins
    where_clauses, params, joins = build_query_context(main_table, filters, required_tables_extra=group_tables)

    # SELECT clause
    select_cols = [f"{col} AS {col.split('.')[-1]}" for col in group_cols]
    select_clause = ", ".join(select_cols)

    # build query
    if entity == '*':
        query = f"SELECT {select_clause}, COUNT(*) AS {by_entities[0]}_count FROM {main_table} {' '.join(joins)}"
    else:
        query = f"SELECT {select_clause}, COUNT(DISTINCT {COLUMN_MAP[entity]}) AS {entity}_count FROM {main_table} {' '.join(joins)}"

    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)

    group_by_clause = ", ".join(group_cols)
    if entity == '*':
        order_by_clause = ", ".join(group_cols)
    else:
        order_by_clause = ", ".join([COLUMN_MAP[entity]]+ group_cols)

    query += f" GROUP BY {group_by_clause} ORDER BY {order_by_clause} DESC"
    return execute_query(db_path, query, params)

# --------------- Cross-table health checks --------------
# 1. Find experiments missing expected files (raw, tracking, masks, analysis)
def find_experiments_missing_files(db_path, file_types = ["raw", "tracking", "mask", "analysis"], filters=None, limit=50):
    filters = filters or {}
    where_clauses, params, joins = build_query_context("Experiment", filters, base_tables={"Experiment", "Organism", "Protein", "StrainOrCellLine", "Condition", "CaptureSetting", "User"})

    # LEFT JOINs to file tables
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

    query = BASE_EXPERIMENT_QUERY + " " + " ".join(joins) + " " + " ".join(file_joins)
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    query += " ORDER BY Experiment.date DESC"
    if limit:
        query += f" LIMIT {limit}"

    return execute_query(db_path, query, params)

# -------------- Validity & Quality Control Queries --------------

# 1. List all invalid experiments (with reasons) --> use list_experiments with filters={"is_valid": "N"}

# 2. find missing values in the desired tables 

def find_missing_values(db_path, requested_columns, missing_columns, main_table="Experiment", mode = "any", filters=None, limit=50):
    """
    Find rows where one or more missing_columns are NULL or empty.

    requested_columns: columns to SELECT
    missing_columns: columns to check for NULL/empty
    main_table: starting table (default Experiment)
    filters: optional filters
    mode:
       "any"  = returns rows where *at least one* missing_column is NULL/empty
       "none" = returns entities that have *no related rows at all* in missing_columns
    group_by: in "none" mode, which column(s) to group by (default = first requested column)
    """
    filters = filters or {}

    # Normalize: allow single column as str
    if isinstance(missing_columns, str):
        missing_columns = [missing_columns]

    sql_requested_columns = []
    requested_tables = set()

    # Resolve missing columns
    sql_missing_columns = []
    for col in missing_columns:
        if col not in COLUMN_MAP:
            raise ValueError(f"Unknown missing column: {col}")
        sql_col = COLUMN_MAP[col]
        sql_missing_columns.append(sql_col)
        requested_tables.add(sql_col.split('.')[0])

    # Resolve requested columns
    if requested_columns == ["*"]:
        with sqlite3.connect(db_path) as conn:
            all_cols = expand_columns(conn, main_table)
        sql_requested_columns = [f"{main_table}.{c}" for c in all_cols]
        # all columns already from main_table → nothing extra to add
        requested_tables.add(main_table)
    else:
        for col in requested_columns:
            if col not in COLUMN_MAP:
                raise ValueError(f"Unknown requested column: {col}")
            sql_requested_columns.append(COLUMN_MAP[col])
            requested_tables.add(COLUMN_MAP[col].split('.')[0])

    # Resolve filters
    for k in filters.keys():
        requested_tables.add(COLUMN_MAP[k].split('.')[0])

    # Choose main table: use the table of the *first missing column*
    #main_table = sql_missing_columns[0].split('.')[0]

    # Build joins
    where_clauses, params, joins = build_query_context(
        main_table=main_table,
        filters=filters,
        required_tables_extra=requested_tables
    )

        # Replace inner joins with LEFT JOIN for non-main tables
    left_joins = []
    for j in joins:
        if j.startswith("JOIN "):
            left_joins.append(j.replace("JOIN", "LEFT JOIN", 1))
        else:
            left_joins.append(j)

    if mode == "any":
        # Missing-value condition: OR across all columns
        missing_conditions = [f"({c} IS NULL OR {c} = '')" for c in sql_missing_columns]
        where_clauses.append("(" + " OR ".join(missing_conditions) + ")")

        # Build query
        query = f"SELECT DISTINCT {', '.join(sql_requested_columns)} FROM {main_table} " + " ".join(left_joins)
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
        query += f" ORDER BY {sql_missing_columns[0]} ASC"
        if limit:
            query += f" LIMIT {limit}"
        
       
        return execute_query(db_path, query, params)
    
    elif mode == "none":
        # groups where *all rows* have missing values → GROUP BY + HAVING
        group_by_cols = ", ".join(sql_requested_columns)
        # Build COUNTs for each missing column being not null
        count_notnull = [f"SUM(CASE WHEN {c} IS NOT NULL AND {c} <> '' THEN 1 ELSE 0 END)" for c in sql_missing_columns]
        having_condition = " + ".join(count_notnull) + " = 0"  # no non-null rows at all

        query = f"""
        SELECT {group_by_cols}
        FROM {main_table} {' '.join(left_joins)}
        """
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
        query += f" GROUP BY {group_by_cols} HAVING {having_condition}"
        query += f" ORDER BY {sql_requested_columns[0]} ASC"
        if limit:
            query += f" LIMIT {limit}"
        return execute_query(db_path, query, params)

    else:
        raise ValueError("mode must be 'any' or 'none'")
# 3. find logical duplicates for experiments (Same organism, protein, user, condition, date, replicate, capture_setting — but potentially different comments, experiment_path)
def find_duplicate_experiments(db_path, filters=None):
    """
    Find experiments that share the same key metadata but differ in comment or other fields.
    """
    where_clauses, params, joins = build_query_context("Experiment", filters)
    query = f"""
    SELECT 
        Organism.organism_name AS organism,
        Protein.protein_name AS protein,
        Condition.condition_name AS condition,
        Experiment.date,
        Experiment.replicate,
        CaptureSetting.capture_type,
        User.user_name AS user,
        COUNT(*) AS duplicate_count,
        GROUP_CONCAT(Experiment.id) AS experiment_ids
    FROM Experiment
    JOIN Organism ON Experiment.organism_id = Organism.id
    JOIN Protein ON Experiment.protein_id = Protein.id
    JOIN Condition ON Experiment.condition_id = Condition.id
    JOIN CaptureSetting ON Experiment.capture_setting_id = CaptureSetting.id
    JOIN User ON Experiment.user_id = User.id
    {' '.join(joins)}
    """

    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)

    query += """
    GROUP BY Organism.organism_name, Protein.protein_name, Condition.condition_name, Experiment.date, Experiment.replicate, CaptureSetting.capture_type, User.user_name
    HAVING COUNT(*) > 1
    ORDER BY duplicate_count DESC
    """
    return execute_query(db_path, query, params)

# 4. find the duplicates in the tables with the desired columns

def find_near_duplicates_by_columns(
    db_path,
    table,
    key_column="id",
    include_columns=None,     # columns that definse "same"
    exclude_columns=None,     # or: all columns except these (and key)
    filters=None,
    show_columns=None         # extra descriptive columns (aliases or raw)
):
    """
    Find duplicate rows in `table` based on a chosen set of columns.
    - include_columns: list of COLUMN_MAP aliases or raw table columns to group by
    - exclude_columns: alternative: use all columns except these (and key)
    - show_columns: extra descriptive columns (aliases or raw) to show across dupes
    """
    filters = filters or {}

    # --- Expand available columns in the table
    with sqlite3.connect(db_path) as conn:
        table_cols = expand_columns(conn, table)

    if key_column not in table_cols:
        raise ValueError(f"Primary key column {key_column} not found in table {table}.")

    def resolve_col(col):
        """Resolve alias via COLUMN_MAP or assume raw column of this table."""
        if col in COLUMN_MAP:
            return COLUMN_MAP[col], col
        else:
            return f"{table}.{col}", col

    # --- Decide group-by columns
    if include_columns is not None:
        resolved = [resolve_col(c) for c in include_columns]
    else:
        excl = set(exclude_columns or [])
        resolved = [(f"{table}.{c}", c) for c in table_cols if c != key_column and c not in excl]

    sql_group_cols, group_labels = zip(*resolved) if resolved else ([], [])
    if not sql_group_cols:
        raise ValueError("No columns left to group by.")

    # --- Extra descriptive columns to show
    sql_show_cols, show_labels = [], []
    if show_columns:
        if show_columns == ["*"]:
            for c in table_cols:
                if c != key_column and c not in sql_group_cols:
                    sql_show_cols.append(f"{table}.{c}")
                    show_labels.append(c)
        else:
            for c in show_columns:
                sql_col, label = resolve_col(c)
                sql_show_cols.append(sql_col)
                show_labels.append(label)

    # --- Build joins (always include all required tables for selected cols)
    required_tables = {table}
    for sql_col in list(sql_group_cols) + sql_show_cols:
        required_tables.add(sql_col.split('.')[0])
    for k in filters.keys():
        required_tables.add(COLUMN_MAP[k].split('.')[0])

    where_clauses, params, joins = build_query_context(
        main_table=table,
        filters=filters,
        required_tables_extra=required_tables
    )

    # --- Normalized expressions for GROUP BY
    def norm(expr): return f"LOWER(TRIM({expr}))"
    group_by_exprs = [norm(c) for c in sql_group_cols]
    group_by_sql = ", ".join(group_by_exprs)

    # --- SELECT list
    group_samples = [f"MIN({c}) AS {label}" for c, label in zip(sql_group_cols, group_labels)]
    shown_values = [f"GROUP_CONCAT(DISTINCT {c}) AS {label}"
                    for c, label in zip(sql_show_cols, show_labels)]

    select_list = ", ".join(
        group_samples
        + shown_values
        + [f"COUNT(*) AS duplicate_rows",
           f"GROUP_CONCAT({table}.{key_column}) AS ids"]
    )

    # --- Final query
    query = f"SELECT {select_list} FROM {table} {' '.join(joins)}"
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    query += f" GROUP BY {group_by_sql} HAVING COUNT(*) > 1 ORDER BY duplicate_rows DESC"

    print(query)  # for debugging
    return execute_query(db_path, query, params)

# 5. count unique combinations of the desired columns in the desired table
def count_unique_combinations(db_path, columns, table="Experiment", filters=None):
    filters = filters or {}

    sql_columns = []
    requested_tables = {table}

    for col in columns:
        if col not in COLUMN_MAP:
            raise ValueError(f"Unknown column: {col}")
        sql_col = COLUMN_MAP[col]
        sql_columns.append(sql_col)
        # extract table name for join inference
        requested_tables.add(sql_col.split(".")[0])

    # ALSO add tables referenced in filters
    for key in filters.keys():
        if key not in COLUMN_MAP:
            raise ValueError(f"Unknown filter key: {key}")
        col = COLUMN_MAP[key]
        requested_tables.add(col.split(".")[0])

    where_clause, params, joins = build_query_context(main_table=table, filters=filters, required_tables_extra=requested_tables)
    base_query = f"SELECT {', '.join(sql_columns)}, COUNT(*) AS count FROM {table} " + " ".join(joins)
    if where_clause:
        base_query += " WHERE " + " AND ".join(where_clause)
    group_by_clause = ", ".join(sql_columns)
    base_query += f" GROUP BY {group_by_clause} ORDER BY count DESC"
    return execute_query(db_path, base_query + ";", params)

# 6. Experiments missing all file types (already implemented). --> use find_experiments_missing_files with all file types.

# 7. coutn experiments with/without multiple file types, grouped by chosen entity
def count_experiments_with_files(db_path, group_by="user_name", file_types=("raw", "tracking", "mask", "analysis_file", "analysis_result"), filters=None, limit=50):
    """
    Summarize experiments with/without multiple file types, grouped by chosen entity.

    group_by: alias from COLUMN_MAP (e.g. "user_name", "organism", "protein")
              or raw "Table.col" string.
    file_types: list/tuple of file types among {"raw", "tracking", "mask", "analysis"}.
    """
    filters = filters or {}

    # Resolve group_by to SQL column
    if group_by in COLUMN_MAP:
        group_col = COLUMN_MAP[group_by]
    elif "." in group_by:  # raw column string
        group_col = group_by
    else:
        raise ValueError(f"Unsupported group_by: {group_by}")

    # --- Base joins (needed for filters)
    where_clauses, params, joins = build_query_context(
        main_table="Experiment",
        filters=filters
    )

    # --- File-type joins and file_id columns
    file_joins = []
    file_checks = {}
    if "raw" in file_types:
        file_joins.append("LEFT JOIN RawFiles ON RawFiles.experiment_id = Experiment.id")
        file_checks["raw"] = "RawFiles.id"
    if "tracking" in file_types:
        file_joins.append("LEFT JOIN TrackingFiles ON TrackingFiles.experiment_id = Experiment.id")
        file_checks["tracking"] = "TrackingFiles.id"
    if "mask" in file_types:
        file_joins.append("LEFT JOIN Masks ON Masks.experiment_id = Experiment.id")
        file_checks["mask"] = "Masks.id"
    if "analysis_file" in file_types:
        file_joins.append(
            "LEFT JOIN ExperimentAnalysisFiles ON ExperimentAnalysisFiles.experiment_id = Experiment.id "
            "LEFT JOIN AnalysisFiles ON AnalysisFiles.id = ExperimentAnalysisFiles.analysis_file_id"
        )
        file_checks["analysis_file"] = "AnalysisFiles.id"
    if "analysis_result" in file_types:
        file_joins.append(
        "LEFT JOIN AnalysisResultExperiments "
        "ON AnalysisResultExperiments.experiment_id = Experiment.id "
        "LEFT JOIN AnalysisResults "
        "ON AnalysisResults.id = AnalysisResultExperiments.analysis_result_id"
        )
        file_checks["analysis_result"] = "AnalysisResults.id"


    # --- SELECT list
    select_parts = [
        f"{group_col} AS group_value",
        "COUNT(DISTINCT Experiment.id) AS total_experiments",
    ]
    for ftype, col in file_checks.items():
        select_parts.append(f"COUNT(DISTINCT CASE WHEN {col} IS NOT NULL THEN Experiment.id END) AS with_{ftype}")
        select_parts.append(f"COUNT(DISTINCT CASE WHEN Experiment.id IS NOT NULL AND {col} IS NULL THEN Experiment.id END) AS without_{ftype}")

    select_clause = ",\n       ".join(select_parts)

    # --- Build query
    if group_col.split(".")[0] not in {"Experiment"}:
        joins.insert(0,f"RIGHT JOIN {group_col.split('.')[0]} ON Experiment.{group_col.split('.')[0].lower()}_id = {group_col.split('.')[0]}.id")
    query = f"""
    SELECT {select_clause}
    FROM Experiment
    {' '.join(joins)}
    {' '.join(file_joins)}
    """
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)

    query += f" GROUP BY {group_col} ORDER BY total_experiments DESC"
    if limit:
        query += f" LIMIT {limit}"

    return execute_query(db_path, query, params)

# --------- Orphaned data detection --------

# 9. forign key integrity checks 
def find_invalid_foreign_keys(db_path, child_table, fk_column, parent_table, parent_key="id", filters=None, limit=1000):
    """
    Find rows in child_table where fk_column references a non-existent row in parent_table.
    Example: find_invalid_foreign_keys(DB_PATH, "Experiment", "capture_setting_id", "CaptureSetting")
    """
    filters = filters or {}
    requested_tables = set()
    for key in filters.keys():
        if key not in COLUMN_MAP:
            raise ValueError(f"Unknown filter key: {key}")
        col = COLUMN_MAP[key]
        requested_tables.add(col.split(".")[0])
    where_clauses, params, joins = build_query_context(child_table, filters, required_tables_extra=requested_tables, base_tables={child_table, parent_table})

    query = f"""
    SELECT {child_table}.id AS {child_table}_id, {child_table}.{fk_column}
    FROM {child_table}
    LEFT JOIN {parent_table} ON {child_table}.{fk_column} = {parent_table}.{parent_key}
    {' '.join(joins) if joins else ''}
    WHERE {child_table}.{fk_column} IS NOT NULL
      AND {parent_table}.{parent_key} IS NULL
    {' AND ' + ' AND '.join(where_clauses) if where_clauses else ''}
    ORDER BY {child_table}.id ASC
    LIMIT {limit}
    """
    return execute_query(db_path, query,params)

# 10. find Orphan records (parent never referenced by any child)
def find_orphan_parents(db_path, parent_table, child_table, fk_column, parent_key="id", filters= None, limit=50):
    """
    Find rows in parent_table that are never referenced by child_table.fk_column.
    Example: find_orphan_parents(DB_PATH, "CaptureSetting", "Experiment", "capture_setting_id")
    """
    filters = filters or {}
    requested_tables = set()
    for key in filters.keys():
        if key not in COLUMN_MAP:
            raise ValueError(f"Unknown filter key: {key}")
        col = COLUMN_MAP[key]
        requested_tables.add(col.split(".")[0])
    where_clauses, params, joins = build_query_context(parent_table, filters, required_tables_extra=requested_tables, base_tables={parent_table, child_table})


    query = f"""
    SELECT {parent_table}.{parent_key} AS {parent_table}_id, {parent_table}.*
    FROM {parent_table}
    LEFT JOIN {child_table} ON {child_table}.{fk_column} = {parent_table}.{parent_key}
    {' '.join(joins) if joins else ''}
    WHERE {child_table}.{fk_column} IS NULL
    {' AND ' + ' AND '.join(where_clauses) if where_clauses else ''}
    ORDER BY {parent_table}.{parent_key} ASC
    LIMIT {limit}
    """
    return execute_query(db_path, query, params)

# 11. Categorial  data validation (e.g. check if values in a column belong to a predefined set)
def find_invalid_categorical_values(
    db_path,
    table,
    column,
    allowed_values,
    filters=None,
    limit=50,
    summarize=False
):
    """
    Find rows in `table` where `column` contains a value not in allowed_values.
    - summarize=True → return counts of unique invalid values
    """
    filters = filters or {}
    requested_tables = {table}
    for key in filters.keys():
        if key not in COLUMN_MAP:
            raise ValueError(f"Unknown filter key: {key}")
        col = COLUMN_MAP[key]
        requested_tables.add(col.split(".")[0])

    where_clauses, params, joins = build_query_context(
        main_table=table,
        filters=filters,
        required_tables_extra=requested_tables,
        base_tables={table}
    )

    allowed_list = ", ".join(f"'{v}'" for v in allowed_values)

    # --- Base WHERE condition
    invalid_condition = (
        f"{table}.{column} NOT IN ({allowed_list}) "
        f"AND {table}.{column} IS NOT NULL "
        f"AND TRIM({table}.{column}) <> ''"
    )

    if summarize:
        # Just count invalid values
        query = f"""
        SELECT {table}.{column} AS invalid_value,
               COUNT(*) AS count
        FROM {table}
        {' '.join(joins) if joins else ''}
        WHERE {invalid_condition}
        {' AND ' + ' AND '.join(where_clauses) if where_clauses else ''}
        GROUP BY {table}.{column}
        ORDER BY count DESC
        """
    else:
        # Show offending rows
        query = f"""
        SELECT {table}.id, {table}.{column}
        FROM {table}
        {' '.join(joins) if joins else ''}
        WHERE {invalid_condition}
        {' AND ' + ' AND '.join(where_clauses) if where_clauses else ''}
        ORDER BY {table}.id ASC
        """
        if limit:
            query += f" LIMIT {limit}"

    return execute_query(db_path, query, params)

# 11. find_incomplete_linked_entities
def find_incomplete_linked_entities_generalized(
    db_path,
    base_table="Experiment",
    present_entity=("RawFiles", "experiment_id"),   # (child_table, FK to base_table)
    missing_entity=("TrackingFiles", "experiment_id"),
    present_bridge=None,  # (bridge_table, present_fk, present_target_fk) if many-to-many
    missing_bridge=None,  # (bridge_table, missing_fk, missing_target_fk) if many-to-many
    filters=None,
    limit=50,
):
    """
    Find records in `base_table` that have one related entity (present_entity)
    but are missing another (missing_entity).

    Supports both direct one-to-many and many-to-many via bridge tables.

    Examples:
      - Experiments that have RawFiles but no TrackingFiles:
          present_entity=("RawFiles", "experiment_id"), missing_entity=("TrackingFiles", "experiment_id")

      - Experiments that have AnalysisFiles but no AnalysisResults:
          present_bridge=("ExperimentAnalysisFiles", "experiment_id", "analysis_file_id"),
          present_entity=("AnalysisFiles", "id"),
          missing_bridge=("AnalysisResultExperiments", "experiment_id", "analysis_result_id"),
          missing_entity=("AnalysisResults", "id")
    """
    filters = filters or {}

    # Resolve entities
    present_table, present_fk = present_entity
    missing_table, missing_fk = missing_entity

    # Build base filters and joins
    where_clauses, params, joins = build_query_context(
        main_table=base_table,
        filters=filters,
        base_tables={base_table, present_table, missing_table},
    )

    # --- Join logic (direct vs via bridge)
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

    # --- Build query
    query = f"""
    SELECT DISTINCT {base_table}.id AS {base_table}_id,
           {base_table}.date,
           {base_table}.replicate,
           {base_table}.is_valid
    FROM {base_table}
    {present_join}
    {missing_join}
    {' '.join(joins) if joins else ''}
    WHERE {missing_null_cond}
    {' AND ' + ' AND '.join(where_clauses) if where_clauses else ''}
    ORDER BY {base_table}.id ASC
    LIMIT {limit}
    """

    print(query)
    return execute_query(db_path, query, params)



if __name__ == "__main__":
    DB_PATH = "/Users/masoomeshafiee/Projects/data_organization/data-management-system-SQLite/db/Reyes_lab_data.db" # <-- change this

    #result = get_experiment_metadata(DB_PATH, 1)
    #print(list_experiments_by_protein(DB_PATH, "Rfa1", 2))
    #print(list_experiments_by_user(DB_PATH, "masoumeh", "rfa1"))

    #result = list_experiments(DB_PATH, filters={"user_name": "Masoumeh" }, limit=200)

    #result = list_entity(DB_PATH,requested_columns=["*"],main_table="Experiment",filters={"condition": "HU", "organism": "yeast", "protein":"Rfa1" },limit=20)

    #result = list_experiments_between_dates(DB_PATH, "20230901", "20231030", filters = {"condition": "cpt", "is_valid":"Y"}, limit=50)

    #result = count_experiments_by_period(DB_PATH, period="year", filters = {"condition": "HU", "is_valid":"Y", "capture_type":"fast"})

    #result = find_most_recent_experiment(DB_PATH, {"condition": "HU", "concentration_value" : "200"})

    #result = list_experiments_in_period(DB_PATH, year=2023, filters = {"condition": "cpt", "capture_type": "fast"}, limit=50)

    #result = find_earliest_experiment(DB_PATH, {"condition": "cpt", "concentration_value" : "40", "dye_concentration_value": "50", "time_interval": "1"})

    #result = count_experiments_trend(DB_PATH, group_by=["capture_type"], filters={"is_valid":"Y"})

    #result = list_recent_experiments(DB_PATH, days=580, filters={"condition": "cpt"}, limit=50)

    #count_entity_by_another(DB_PATH, "*", ["analysis_file_type"], filters=None)

    #result = count_entity_by_another(DB_PATH, "experiment_id", ["user_name"])  

    result = find_experiments_missing_files(DB_PATH, file_types = ["tracking", "mask"], filters={"is_valid":"Y"}, limit=50)

    #result = find_missing_values(DB_PATH, ["user_name"],["email"], main_table= "User", mode = "any", limit=50)

    #result = find_duplicate_experiments(DB_PATH, filters=None)

    #find_duplicates_by_columns(DB_PATH, table="Condition", key_column="id", include_columns=["condition_name"], filters=None)

    #result = find_near_duplicates_by_columns(DB_PATH,table="Experiment", include_columns= ["organism", "protein", "condition"],show_columns=["user_name", "date"], filters=None)
    
    #result = count_unique_combinations(DB_PATH, columns=["capture_type", "protein"], table="CaptureSetting", filters=None)

    
    #****result = count_experiments_with_files(DB_PATH, group_by="user_name", file_types=("raw", "tracking", "mask", "analysis_result"), limit=50)
    
    #result = find_invalid_foreign_keys(DB_PATH, child_table="RawFiles", fk_column="experiment_id", parent_table="Experiment", filters={"raw_file_type": "w1bf"},limit=10000)
    
    #result = find_orphan_parents(DB_PATH, parent_table="Experiment", child_table="Masks", fk_column="experiment_id", filters={"is_valid": "Y"}, limit=50)

    #result = find_invalid_categorical_values(DB_PATH, table="Masks", column="mask_type", allowed_values=["cell", "nucleus", "Nucleus-G1"], filters=None, limit=500)

    # find_incomplete_linked_entities_generalized(DB_PATH, base_table="Experiment", present_entity=("RawFiles", "experiment_id"), missing_entity=("TrackingFiles", "experiment_id"),filters={"is_valid": "Y"}, limit=50)
    # find_incomplete_linked_entities_generalized(DB_PATH, base_table="Experiment", present_bridge=("AnalysisResultExperiments", "experiment_id", "analysis_result_id"), present_entity=("AnalysisResults", "id"), missing_bridge=("ExperimentAnalysisFiles", "experiment_id","analysis_file_id" ), missing_entity=("AnalysisFiles", "id"), filters={"is_valid": "Y"}, limit=50)



    # save to CSV
    if result is not None:
        print(result)
        result.to_csv("/Users/masoomeshafiee/Desktop/Presentation/experiment_query_result.csv", index=False)
        print("Query results saved to experiment_query_result.csv")
    else:
        print("No results found or an error occurred.")

