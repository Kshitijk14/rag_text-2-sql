import os
import gc
import shutil
import traceback

from pathlib import Path

from utils.config import CONFIG
from utils.logger import setup_logger

from utils.helpers.common import prep_db_engine, wrap_sql_engine
from utils.populate.custom_index_load import index_all_tables_with_chroma


# configs
LOG_PATH = Path(CONFIG["LOG_PATH"])
CHINOOK_DB_PATH = Path(CONFIG["CHINOOK_DB_PATH"])
CHROMA_DB_DIR = Path(CONFIG["CHROMA_DB_DIR"])
MAX_RETRIES = CONFIG["MAX_RETRIES"]

# logger
LOG_DIR = os.path.join(os.getcwd(), str(LOG_PATH))
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "stage_02_populate_vector_db.log")


def clear_vector_database(chroma_db_dir):
    if os.path.exists(chroma_db_dir):
        shutil.rmtree(chroma_db_dir)


def run_db_population(
    reset: bool = False, 
    main_db_dir: Path = CHINOOK_DB_PATH, 
    chroma_db_dir: Path= CHROMA_DB_DIR,
):
    
    logger = setup_logger("vector_db_population_logger", LOG_FILE)
    logger.info(" ")
    logger.info("--------++++++++Starting DB Population stage.....")
    
    if reset:
            logger.info("(RESET DB) Clearing the vector db...")
            clear_vector_database(chroma_db_dir)
            logger.info("(RESET DB) Vector db cleared successfully.")

    try:
        logger.info("Preparing engines...")
        engine, _ = prep_db_engine(main_db_dir, logger)

        logger.info("Wrapping SQL engine...")
        sql_database, table_node_mapping = wrap_sql_engine(engine, logger)

        logger.info("Build vector indexes & indexing all tables with ChromaDB...")
        vector_index_dict = index_all_tables_with_chroma(engine, sql_database, chroma_db_dir, logger)

        logger.info("--------++++++++Db Population stage successfully completed.")
        logger.info(" ")
        
        # Manual memory cleanup
        logger.info("Cleaning up resources...")
        del (
            engine, 
            sql_database, table_node_mapping, 
            vector_index_dict
        )
        gc.collect()
    except Exception as e:
        logger.error(f"Error at [Stage 01]: {e}")
        logger.debug(traceback.format_exc())


if __name__ == "__main__":
    run_db_population()