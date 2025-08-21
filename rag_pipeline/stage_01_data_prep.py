import os
import sqlite3
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional

from utils.config import CONFIG
from utils.logger import setup_logger
from utils.stage_01.schema_helpers import (
    connect_sqlite,
    get_all_table_names,
    get_table_info,
    get_foreign_keys,
    estimate_row_count,
    build_schema_text,
    summarize_table_heuristic,
)
from llama_index.core.base.llms.types import ChatMessage

try:
    from utils.llm.get_llm_func import get_llm_func
    _LLM_AVAILABLE = True
except Exception:
    _LLM_AVAILABLE = False


# configs
LOG_PATH: Path = Path(CONFIG["LOG_PATH"])
CHINOOK_DBEAVER_DB_PATH: Path = Path(CONFIG["CHINOOK_DBEAVER_DB_PATH"])
CHINOOK_TABLE_SUMMARIES_DB_PATH: Path = Path(CONFIG["CHINOOK_TABLE_SUMMARIES_DB_PATH"])

USE_LLM_FOR_SUMMARY: bool = (CONFIG["USE_LLM_FOR_SUMMARY"])
ROWCOUNT_ESTIMATE_LIMIT: Optional[int] = CONFIG["ROWCOUNT_ESTIMATE_LIMIT"]
SUMMARY_STYLE: str = CONFIG["SUMMARY_STYLE"]

# logger
LOG_DIR = os.path.join(os.getcwd(), str(LOG_PATH))
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "stage_01_data_prep.log")


def ensure_output_db_schema(conn: sqlite3.Connection, logger) -> None:
    """
    Create (if not exists) the normalized schema inside table_summaries.db.

    Tables created:
        - table_info(
            table_name TEXT PRIMARY KEY, 
            table_type TEXT, 
            row_estimate INTEGER,
            schema_text TEXT, 
            summary_text TEXT
        )
        - column_info(
            table_name TEXT, 
            column_name TEXT, 
            data_type TEXT, 
            not_null INTEGER,
            default_value TEXT,
            pk INTEGER,
            ordinal_position INTEGER,
            PRIMARY KEY(table_name, column_name)
        )
        - foreign_keys(
            table_name TEXT, 
            id INTEGER, 
            seq INTEGER, 
            fk_column TEXT,
            ref_table TEXT, 
            ref_column TEXT, 
            on_update TEXT, 
            on_delete TEXT, 
            match TEXT,
            PRIMARY KEY(table_name, id, seq)
        )
    """
    try:
        logger.info("[Stage 01] Ensuring output DB schema exists...")
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS table_info (
                table_name     TEXT PRIMARY KEY,
                table_type     TEXT,
                row_estimate   INTEGER,
                schema_text    TEXT,
                summary_text   TEXT
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS column_info (
                table_name        TEXT,
                column_name       TEXT,
                data_type         TEXT,
                not_null          INTEGER,
                default_value     TEXT,
                pk                INTEGER,
                ordinal_position  INTEGER,
                PRIMARY KEY (table_name, column_name)
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS foreign_keys (
                table_name  TEXT,
                id          INTEGER,
                seq         INTEGER,
                fk_column   TEXT,
                ref_table   TEXT,
                ref_column  TEXT,
                on_update   TEXT,
                on_delete   TEXT,
                match       TEXT,
                PRIMARY KEY (table_name, id, seq)
            );
        """)
        conn.commit()
        logger.info("[Stage 01] Output DB schema ensured.")
    except Exception as e:
        logger.error(f"[Stage 01] Failed ensuring output DB schema: {e}")
        logger.debug(traceback.format_exc())
        raise

def has_summary(conn: sqlite3.Connection, table_name: str) -> bool:
    """Return True if summary already exists for this table."""
    cur = conn.cursor()
    cur.execute("SELECT summary_text FROM table_info WHERE table_name=?", (table_name,))
    row = cur.fetchone()
    return bool(row and row[0] and row[0].strip())

def upsert_table_info(conn: sqlite3.Connection, row: Dict[str, Any], logger) -> None:
    """
    Insert or replace a single row in table_info.
    row keys (types):
        - table_name: str
        - table_type: str
        - row_estimate: Optional[int]
        - schema_text: str
        - summary_text: str
    """
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO table_info (table_name, table_type, row_estimate, schema_text, summary_text)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(table_name) DO UPDATE SET
                table_type=excluded.table_type,
                row_estimate=excluded.row_estimate,
                schema_text=excluded.schema_text,
                summary_text=excluded.summary_text;
            """,
            (
                row["table_name"],
                row.get("table_type", "table"),
                row.get("row_estimate"),
                row.get("schema_text", ""),
                row.get("summary_text", ""),
            ),
        )
        conn.commit()
    except Exception as e:
        logger.error(f"[Stage 01] Failed upserting table_info for {row.get('table_name')}: {e}")
        logger.debug(traceback.format_exc())
        raise


def upsert_column_info(conn: sqlite3.Connection, items: List[Dict[str, Any]], logger) -> None:
    """
    Bulk upsert column_info rows.
    Each dict requires:
        - table_name: str
        - column_name: str
        - data_type: str
        - not_null: int (0/1)
        - default_value: Optional[str]
        - pk: int (0/1)
        - ordinal_position: int
    """
    try:
        cur = conn.cursor()
        cur.executemany(
            """
            INSERT INTO column_info (table_name, column_name, data_type, not_null, default_value, pk, ordinal_position)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(table_name, column_name) DO UPDATE SET
                data_type=excluded.data_type,
                not_null=excluded.not_null,
                default_value=excluded.default_value,
                pk=excluded.pk,
                ordinal_position=excluded.ordinal_position;
            """,
            [
                (
                    item["table_name"],
                    item["column_name"],
                    item.get("data_type"),
                    int(item.get("not_null", 0)),
                    item.get("default_value"),
                    int(item.get("pk", 0)),
                    int(item.get("ordinal_position", 0)),
                )
                for item in items
            ],
        )
        conn.commit()
    except Exception as e:
        logger.error(f"[Stage 01] Failed bulk upserting column_info: {e}")
        logger.debug(traceback.format_exc())
        raise


def upsert_foreign_keys(conn: sqlite3.Connection, items: List[Dict[str, Any]], logger) -> None:
    """
    Bulk upsert for foreign_keys table.
    Each dict requires:
        - table_name: str
        - id: int
        - seq: int
        - fk_column: str
        - ref_table: str
        - ref_column: str
        - on_update: Optional[str]
        - on_delete: Optional[str]
        - match: Optional[str]
    """
    try:
        cur = conn.cursor()
        cur.executemany(
            """
            INSERT INTO foreign_keys (table_name, id, seq, fk_column, ref_table, ref_column, on_update, on_delete, match)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(table_name, id, seq) DO UPDATE SET
                fk_column=excluded.fk_column,
                ref_table=excluded.ref_table,
                ref_column=excluded.ref_column,
                on_update=excluded.on_update,
                on_delete=excluded.on_delete,
                match=excluded.match;
            """,
            [
                (
                    item["table_name"],
                    int(item["id"]),
                    int(item["seq"]),
                    item["fk_column"],
                    item["ref_table"],
                    item["ref_column"],
                    item.get("on_update"),
                    item.get("on_delete"),
                    item.get("match"),
                )
                for item in items
            ],
        )
        conn.commit()
    except Exception as e:
        logger.error(f"[Stage 01] Failed bulk upserting foreign_keys: {e}")
        logger.debug(traceback.format_exc())
        raise


def maybe_summarize(schema_text: str, table_name: str, logger) -> str:
    """
    Summarize a table's schema using either an LLM (if enabled and available) or a deterministic heuristic fallback.
    """
    try:
        if USE_LLM_FOR_SUMMARY and _LLM_AVAILABLE:
            logger.info(f"[Stage 01] Summarizing table '{table_name}' with LLM...")
            chain = get_llm_func()
            
            if SUMMARY_STYLE == "one_liner":
                system_prompt = (
                    "You are an expert data engineer. Write a concise one-line summary of the given SQLite table."
                )
            else:
                system_prompt = (
                    "You are an expert data engineer. Write a 2-3 sentence summary of the given SQLite table, "
                    "describing its purpose and key relationships. Avoid quoting column types."
                )

            user_prompt = f"Table: {table_name}\n\nSchema:\n{schema_text}"

            try:
                summary = chain.chat(
                    messages=[
                        ChatMessage(role="system", content=system_prompt),
                        ChatMessage(role="user", content=user_prompt),
                    ]
                )
                # normalize output
                if hasattr(summary, "message"):
                    return summary.message.content.strip()
                return str(summary).strip()
            except Exception as e:
                logger.warning(f"[Stage 01] LLM summarization failed for '{table_name}': {e}. Falling back to heuristic.")
                logger.debug(traceback.format_exc())
            
        return summarize_table_heuristic(schema_text, table_name)
    except Exception as e:
        logger.error(f"[Stage 01] Unexpected error in summarization for '{table_name}': {e}")
        logger.debug(traceback.format_exc())
        return summarize_table_heuristic(schema_text, table_name)


def run_data_prep(source_db_path: Path = CHINOOK_DBEAVER_DB_PATH, output_db_path: Path = CHINOOK_TABLE_SUMMARIES_DB_PATH) -> None:
    """
    Orchestrates Stage 01:
        1) Connect to Chinook.db
        2) Read schema (tables, columns, foreign keys)
        3) Estimate row counts (optionally limited by ROWCOUNT_ESTIMATE_LIMIT)
        4) Build schema_text + summary_text per table
        5) Save normalized info into table_summaries.db

    Inputs:
        - source_db_path: Path to the SQLite DB to introspect.
        - output_db_path: Path to the SQLite DB to write summaries into.

    Outputs:
        - Creates/updates 'output_db_path' with 3 tables: table_info, column_info, foreign_keys.
            No return value; errors are logged and raised.
    """
    logger = setup_logger("stage_01_data_prep_logger", LOG_FILE)
    logger.info(" ")
    logger.info("======== Starting Stage 01: Data Prep (schema -> table_summaries.db) ========")

    try:
        # 1) Connect to source (Chinook.db)
        if not source_db_path.exists():
            raise FileNotFoundError(f"Source DB not found at: {source_db_path}")
        src_conn = connect_sqlite(str(source_db_path), logger)

        # 2) Connect/create output DB (table_summaries.db)
        output_db_path.parent.mkdir(parents=True, exist_ok=True)
        out_conn = connect_sqlite(str(output_db_path), logger)
        ensure_output_db_schema(out_conn, logger)

        # 3) List all user tables
        tables = get_all_table_names(src_conn, logger)
        logger.info(f"[Stage 01] Found {len(tables)} user tables: {tables}")

        for tname in tables:
            logger.info(f"[Stage 01] Processing table: {tname}")
            
            # Skip if already summarized
            if has_summary(out_conn, tname):
                logger.info(f"[Stage 01] Skipping '{tname}' (summary already exists).")
                continue

            # Column metadata
            cols = get_table_info(src_conn, tname, logger)  # List[Dict]
            logger.debug(f"[Stage 01] Columns for {tname}: {cols}")

            # Foreign keys
            fks = get_foreign_keys(src_conn, tname, logger)  # List[Dict]
            logger.debug(f"[Stage 01] Foreign keys for {tname}: {fks}")

            # Row estimate
            row_est = estimate_row_count(src_conn, tname, logger, limit=ROWCOUNT_ESTIMATE_LIMIT)

            # Build human-readable schema text
            schema_text = build_schema_text(tname, cols, fks)

            # Summarize schema into one-line text
            summary_text = maybe_summarize(schema_text, tname, logger)

            # Upsert into output DB
            upsert_table_info(
                out_conn,
                {
                    "table_name": tname,
                    "table_type": "table",
                    "row_estimate": row_est,
                    "schema_text": schema_text,
                    "summary_text": summary_text,
                },
                logger,
            )

            # Upsert column rows
            upsert_column_info(out_conn, cols, logger)

            # Upsert FK rows
            # Attach table_name to each FK row
            for fk in fks:
                fk["table_name"] = tname
            upsert_foreign_keys(out_conn, fks, logger)

        logger.info("======== Stage 01 completed successfully. ========")
        logger.info(" ")
    except Exception as e:
        logger.error(f"[Stage 01] Fatal error: {e}")
        logger.debug(traceback.format_exc())
        raise
    finally:
        try:
            src_conn.close()
        except Exception:
            pass
        try:
            out_conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    run_data_prep()