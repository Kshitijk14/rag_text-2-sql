import os
from pathlib import Path

from pydantic import BaseModel, Field
from sqlalchemy import create_engine, inspect

from llama_index.core import SQLDatabase
from llama_index.core.objects import SQLTableNodeMapping


class TableInfo(BaseModel):
    """Information regarding a structured table."""

    table_name: str = Field(
        ..., description="table name (must be underscores and NO spaces)"
    )
    table_summary: str = Field(
        ..., description="short, concise summary/caption of the table"
    )


def prep_db_engine(main_db_dir: Path, logger):
    """Prepare the SQLite engine for the existing database."""
    try:
        logger.info(f"Creating SQLite DB Engine for the existing Chinook database at {main_db_dir}")
        engine = create_engine(f"sqlite:///{main_db_dir}")
        inspector = inspect(engine)

        return engine, inspector
    except Exception as e:
        logger.error(f"Error preparing summary engine: {e}")
        return None, None

def prep_summary_engine(sqlite_db_dir: Path, logger):
    """Prepare the SQLite engine for storing table summaries."""
    os.makedirs(sqlite_db_dir, exist_ok=True)
    summary_db_path = os.path.join(sqlite_db_dir, "table_summaries.db")

    try:
        logger.info(f"Creating SQLite DB Engine for the new summaries database: {summary_db_path}")
        summary_engine = create_engine(f"sqlite:///{summary_db_path}")

        return summary_engine, summary_db_path
    except Exception as e:
        logger.error(f"Error preparing summary engine: {e}")
        return None, None

def wrap_sql_engine(engine, logger):
    """Wrap SQLAlchemy engine into LlamaIndex SQLDatabase + Node Mapping."""
    logger.info("Wrapping engine into LlamaIndex SQLDatabase")
    sql_database = SQLDatabase(engine)

    logger.info("Creating table node mapping, i.e. mapping from SQL tables -> nodes")
    table_node_mapping = SQLTableNodeMapping(sql_database)

    return sql_database, table_node_mapping