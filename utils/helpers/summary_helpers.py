import os
import json
from pathlib import Path
from sqlalchemy import text, inspect
from llama_index.core.objects import SQLTableSchema


def create_summaries_table(summary_engine, logger):
    """
    Ensure the 'table_summaries' table exists in the SQLite DB.
    """
    try:
        logger.info(" - Ensuring the table exists (id, table_name, table_summary, created_at)")
        with summary_engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS table_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    table_name TEXT NOT NULL UNIQUE,
                    table_summary TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
        logger.info("Table 'table_summaries' ensured/created.")
    except Exception as e:
        logger.error(f"Failed to ensure table_summaries exists: {e}")

def dump_summaries_sqlite(summary_engine, table_infos, logger):
    """
    Save table summaries into SQLite DB.
    Skips duplicates based on table_name.
    """
    try:
        with summary_engine.begin() as conn:
            for t in table_infos:
                conn.execute(
                    text("""
                        INSERT INTO table_summaries (table_name, table_summary) 
                        VALUES (:name, :summary)
                        ON CONFLICT(table_name) DO NOTHING
                    """),
                    {"name": t.table_name, "summary": t.table_summary},
                )
        logger.info(f"Saved {len(table_infos)} summaries to SQLite")
    except Exception as e:
        logger.error(f"Failed to save summaries to SQLite: {e}")

def dump_summaries_json(sqlite_db_dir: Path, table_infos, logger):
    """
    Save table summaries into JSON file.
    Appends/merges with existing file instead of overwriting.
    """
    json_path = os.path.join(sqlite_db_dir, "table_summaries.json")

    try:
        # Load old summaries if file exists
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                old_data = json.load(f)
        else:
            old_data = []

        # Merge with new summaries (skip duplicates by table_name)
        old_map = {d["table_name"]: d for d in old_data}
        for t in table_infos:
            old_map[t.table_name] = t.model_dump()

        merged_data = list(old_map.values())

        # Write back merged data
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)

        logger.info(f"JSON dump updated at {json_path} with {len(table_infos)} new/updated entries")
    except Exception as e:
        logger.error(f"Failed to dump JSON summaries: {e}")


def load_summaries_from_sqlite(summary_engine, logger):
    """Load table summaries stored in SQLite DB."""
    logger.info("Loading all existing summaries from SQLite DB")
    with summary_engine.connect() as conn:
        rows = conn.execute(
            text("SELECT table_name, table_summary FROM table_summaries")
        ).fetchall()
    return rows

def filter_valid_summaries(rows, engine, logger):
    """
    Keep only summaries where the table still exists in the main DB.
    Returns a list of SQLTableSchema objects.
    """
    logger.info("Filtering out only valid tables from loaded summaries that exist in the db")
    table_schema_objs = []

    with engine.connect() as conn:
        inspector = inspect(conn)
        existing_tables = inspector.get_table_names()

    for row in rows:
        if row.table_name in existing_tables and row.table_summary:
            table_schema_objs.append(
                SQLTableSchema(table_name=row.table_name, context_str=row.table_summary)
            )
            logger.info(f"Adding table: {row.table_name} with summary: {row.table_summary}")
        else:
            logger.warning(f"Skipping missing/unextracted table: {row.table_name}")

    return table_schema_objs

def load_summaries_from_json(sqlite_db_dir: Path, logger):
    """Load table summaries stored in JSON file."""
    os.makedirs(sqlite_db_dir, exist_ok=True)
    summary_db_path = os.path.join(sqlite_db_dir, "table_summaries.json")
    
    if not os.path.exists(summary_db_path):
        logger.warning(f"No summary JSON found at {summary_db_path}")
        return []

    logger.info(f"Loading summaries from JSON at {summary_db_path}")
    with open(summary_db_path, "r", encoding="utf-8") as f:
        summaries = json.load(f)

    return summaries  # list[{"table_name": str, "table_summary": str}]


def filter_valid_summaries_from_json(rows, engine, logger):
    """
    Keep only summaries where the table still exists in the main DB.
    Returns a list of SQLTableSchema objects.
    """
    logger.info("Filtering JSON summaries to only valid tables")
    table_schema_objs = []

    with engine.connect() as conn:
        inspector = inspect(conn)
        existing_tables = inspector.get_table_names()

    for row in rows:
        table_name = row.get("table_name")
        table_summary = row.get("table_summary")
        if table_name in existing_tables and table_summary:
            table_schema_objs.append(
                SQLTableSchema(table_name=table_name, context_str=table_summary)
            )
            logger.info(f"Adding table: {table_name} with summary: {table_summary}")
        else:
            logger.warning(f"Skipping missing/unextracted table: {table_name}")

    return table_schema_objs