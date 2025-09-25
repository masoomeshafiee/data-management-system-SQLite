import sqlite3
import pandas as pd
import logging
from collections import deque

logging.basicConfig(
    filename="/Users/masoomeshafiee/Projects/data_organization/db_export.log", # <-- change this
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

BASE_EXPERIMENT_QUERY = """
    SELECT Experiment.id, Organism.name as organism, Protein.name as protein, StrainOrCellLine.name as strain, Condition.name as condition, Condition.concentration_value, Condition.concentration_unit,
            CaptureSetting.capture_type, CaptureSetting.exposure_time, CaptureSetting.time_interval, User.name as user, Experiment.date, Experiment.replicate, Experiment.is_valid, Experiment.comment, Experiment.experiment_path
    FROM Experiment Experiment
    JOIN Organism ON Experiment.organism_id = Organism.id
    JOIN Protein ON Experiment.protein_id = Protein.id
    JOIN StrainOrCellLine ON Experiment.strain_id = STRainOrCellLine.id
    JOIN Condition On Experiment.condition_id = Condition.id
    JOIN CaptureSetting ON Experiment.capture_setting_id = CaptureSetting.id
    JOIN User ON Experiment.user_id = User.id
"""

COLUMN_MAP = {
    "organism": "Organism.name",
    "protein": "Protein.name",
    "strain": "StrainOrCellLine.name",
    "condition": "Condition.name",
    "user_name": "User.name",
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
        
        return result_df
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        return None
    finally:
        if conn:
            conn.close()


# Possible queries

# --------- Meta-data retrieval queries ---------

# 1. Get metadata for a given experiment ID
def get_experiment_metadata(db_path, experiment_id):
    query = BASE_EXPERIMENT_QUERY + "WHERE e.id = ?;"
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

    
    
if __name__ == "__main__":
    DB_PATH = "/Users/masoomeshafiee/Projects/data_organization/Reyes_lab_data.db" # <-- change this

    #print(get_experiment_metadata(DB_PATH, 1))
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
    #result = count_entity_by_another(DB_PATH, "*", ["analysis_file_type"], filters=None)
    #result = find_experiments_missing_files(DB_PATH, file_types = ["raw", "tracking", "mask", "analysis"], filters={"is_valid":"Y"}, limit=50)

    result = find_missing_values(DB_PATH, ["user_name"],["email"], main_table= "User", mode = "any", limit=50)
    # save to CSV
    if result is not None:
        print(result)
        result.to_csv("/Users/masoomeshafiee/Projects/data_organization/experiment_query_result.csv", index=False)
        print("Query results saved to experiment_query_result.csv")
    else:
        print("No results found or an error occurred.")

