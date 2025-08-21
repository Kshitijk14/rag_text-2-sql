import sqlite3
from typing import List, Dict, Any, Optional

def connect_sqlite(db_path: str, logger) -> sqlite3.Connection:
    """
    Connect to a SQLite database.
    Input:
      - db_path: str path to the .db file.
      - logger: logging.Logger-like object.
    Output:
      - sqlite3.Connection
    Raises:
      - sqlite3.Error if connection fails.
    """
    logger.info(f"[utils.sql_schema_utils] Connecting to SQLite DB: {db_path}")
    conn = sqlite3.connect(db_path)
    # Pragmas to get consistent behavior
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA case_sensitive_like = OFF;")
    return conn


def get_all_table_names(conn: sqlite3.Connection, logger) -> List[str]:
    """
    Return a list of user tables (excluding SQLite internal tables).
    """
    sql = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
    cur = conn.cursor()
    cur.execute(sql)
    names = [row[0] for row in cur.fetchall()]
    # if there are views you want, you can add them here or create a separate function
    return names


def get_table_info(conn: sqlite3.Connection, table_name: str, logger) -> List[Dict[str, Any]]:
    """
    Returns PRAGMA table_info as a structured list of dicts.
    Fields:
      - table_name: str
      - column_name: str
      - data_type: str
      - not_null: int (0/1)
      - default_value: Optional[str]
      - pk: int (0/1)
      - ordinal_position: int
    """
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info('{table_name}')")
    rows = cur.fetchall()
    result: List[Dict[str, Any]] = []
    for idx, (cid, name, ctype, notnull, dflt_value, pk) in enumerate(rows):
        result.append({
            "table_name": table_name,
            "column_name": name,
            "data_type": ctype,
            "not_null": int(notnull or 0),
            "default_value": dflt_value,
            "pk": int(pk or 0),
            "ordinal_position": int(idx),
        })
    return result


def get_foreign_keys(conn: sqlite3.Connection, table_name: str, logger) -> List[Dict[str, Any]]:
    """
    Returns PRAGMA foreign_key_list as a structured list of dicts.
    Fields:
      - id: int
      - seq: int
      - fk_column: str
      - ref_table: str
      - ref_column: str
      - on_update: Optional[str]
      - on_delete: Optional[str]
      - match: Optional[str]
    """
    cur = conn.cursor()
    cur.execute(f"PRAGMA foreign_key_list('{table_name}')")
    rows = cur.fetchall()
    # pragma fk list returns columns: id, seq, table, from, to, on_update, on_delete, match
    result: List[Dict[str, Any]] = []
    for (fid, seq, ref_table, from_col, to_col, on_update, on_delete, match) in rows:
        result.append({
            "id": int(fid),
            "seq": int(seq),
            "fk_column": from_col,
            "ref_table": ref_table,
            "ref_column": to_col,
            "on_update": on_update,
            "on_delete": on_delete,
            "match": match,
        })
    return result


def estimate_row_count(conn: sqlite3.Connection, table_name: str, logger, limit: Optional[int] = None) -> Optional[int]:
    """
    Roughly estimate the row count for a table.
    If 'limit' is provided, we return min(actual_count, limit) (as a cheap guard for huge tables).
    For SQLite, COUNT(*) is usually fine for small datasets (like Chinook).

    Output:
      - int row count (possibly capped by limit) or None on error.
    """
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM '{table_name}'")
        (cnt,) = cur.fetchone()
        if limit is not None and cnt is not None:
            return int(min(int(cnt), int(limit)))
        return int(cnt)
    except Exception as e:
        logger.warning(f"[utils.sql_schema_utils] Row count estimate failed for {table_name}: {e}")
        return None


def build_schema_text(table_name: str, columns: List[Dict[str, Any]], fks: List[Dict[str, Any]]) -> str:
    """
    Build a human-readable schema text for embeddings and summaries.
    Example output (single string):
      "Schema of table Customer: columns: CustomerId (INTEGER, PK), FirstName (TEXT, NOT NULL), ...;
       Foreign keys: ['SupportRepId'] -> Employee.EmployeeId;"
    """
    col_parts = []
    for c in columns:
        bits = [c["column_name"], f"({c['data_type'] or 'TEXT'}"]
        flags = []
        if c.get("pk", 0):
            flags.append("PK")
        if c.get("not_null", 0):
            flags.append("NOT NULL")
        if flags:
            bits.append(", " + ", ".join(flags))
        bits.append(")")
        col_parts.append(" ".join(bits))
    cols_str = ", ".join(col_parts) if col_parts else "None"

    fk_parts = []
    for fk in fks:
        fk_parts.append(f"{fk['fk_column']} -> {fk['ref_table']}.{fk['ref_column']}")
    fks_str = ", ".join(fk_parts) if fk_parts else "None"

    schema_text = (
        f"Schema of table {table_name}:\n"
        f"Columns: {cols_str}.\n"
        f"Foreign keys: {fks_str}."
    )
    return schema_text


def summarize_table_heuristic(schema_text: str, table_name: str) -> str:
    """
    Deterministic, simple heuristic summary fallback.
    """
    base = f"Summary of table '{table_name}': "
    # extremely simple heuristic; keep it stable and short
    if "Foreign keys: None" in schema_text:
        return base + "Stores core records with no declared foreign key relationships."
    return base + "Stores core records with relationships to other tables as described."