import os
import shutil
import traceback

import chromadb
from pathlib import Path
from typing import Dict

from llama_index.core import (
    SQLDatabase, 
    VectorStoreIndex,
)
from llama_index.core.objects import SQLTableNodeMapping

from utils.config import CONFIG
from utils.logger import setup_logger

from utils.helpers.population_helpers import (
    get_chroma_collection,
    fetch_table_rows,
    prepare_new_nodes,
    update_index
)


# configs
LOG_PATH = Path(CONFIG["LOG_PATH"])
CHROMA_DB_DIR = Path(CONFIG["CHROMA_DB_DIR"])
MAX_RETRIES = CONFIG["MAX_RETRIES"]

# logger
LOG_DIR = os.path.join(os.getcwd(), str(LOG_PATH))
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "stage_02_populate_vector_db.log")


def wrap_sql_engine(engine, logger):
    """Wrap SQLAlchemy engine into LlamaIndex SQLDatabase + Node Mapping."""
    logger.info("Wrapping engine into LlamaIndex SQLDatabase")
    sql_database = SQLDatabase(engine)

    logger.info("Creating table node mapping, i.e. mapping from SQL tables -> nodes")
    table_node_mapping = SQLTableNodeMapping(sql_database)

    return sql_database, table_node_mapping


def index_all_tables_with_chroma(sql_database, chroma_db_dir: str, logger) -> Dict[str, VectorStoreIndex]:
    """Main driver to index all tables in the SQL database with ChromaDB."""
    os.makedirs(chroma_db_dir, exist_ok=True)
    vector_index_dict = {}
    engine = sql_database.engine

    logger.info(f"[00] Creating persistent Chroma client at: {chroma_db_dir}")
    chroma_client = chromadb.PersistentClient(path=chroma_db_dir)

    for table_name in sql_database.get_usable_table_names():
        try:
            collection = get_chroma_collection(chroma_client, table_name, logger)
            col_names, rows = fetch_table_rows(engine, table_name, logger)
            if not rows:
                continue

            new_nodes = prepare_new_nodes(collection, table_name, col_names, rows, logger)
            index = update_index(collection, new_nodes)

            vector_index_dict[table_name] = index
            logger.info(f"[04] Table {table_name} indexed successfully (total={collection.count()})")

        except Exception as e:
            logger.error(f"[ERROR] Failed to index table {table_name}: {str(e)}")
            raise

    logger.info(f"[05] Successfully indexed {len(vector_index_dict)} tables")
    return vector_index_dict


def clear_vector_database(chroma_db_dir):
    if os.path.exists(chroma_db_dir):
        shutil.rmtree(chroma_db_dir)


def run_db_population(
    engine,
    reset=False, 
    chroma_db_dir: Path= CHROMA_DB_DIR,
):
    
    logger = setup_logger("vector_db_population_logger", LOG_FILE)
    logger.info(" ")
    logger.info("--------++++++++Starting DB Population stage.....")
    
    if reset:
            logger.info("[Stage 01, Part 00.1] (RESET DB) Clearing the vector db...")
            clear_vector_database(chroma_db_dir)
            logger.info("[Stage 01, Part 00.1] (RESET DB) Vector db cleared successfully.")

    try:
        sql_database, table_node_mapping = wrap_sql_engine(engine, logger)

        # Build vector indexes for all tables using ChromaDB
        vector_index_dict = index_all_tables_with_chroma(sql_database, chroma_db_dir, logger)

        logger.info("--------++++++++Db Population stage successfully completed.")
        logger.info(" ")
        
        return sql_database, table_node_mapping, vector_index_dict
    except Exception as e:
        logger.error(f"Error at [Stage 01]: {e}")
        logger.debug(traceback.format_exc())

if __name__ == "__main__":
    run_db_population()