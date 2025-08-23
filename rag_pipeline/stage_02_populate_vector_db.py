import os
import gc
import traceback

import chromadb
from pathlib import Path
from typing import Dict
from sqlalchemy import text

from llama_index.core import (
    SQLDatabase, 
    VectorStoreIndex,
)
from llama_index.core.objects import SQLTableNodeMapping
from llama_index.core.schema import TextNode
from llama_index.core.storage import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

from utils.config import CONFIG
from utils.logger import setup_logger

from utils.llm.get_llm_func import get_embedding_func


# configs
LOG_PATH = Path(CONFIG["LOG_PATH"])
CHINOOK_DB_PATH = Path(CONFIG["CHINOOK_DB_PATH"])
SQLITE_DB_DIR = Path(CONFIG["SQLITE_DB_DIR"])
CHROMA_DB_DIR = Path(CONFIG["CHROMA_DB_DIR"])
MAX_RETRIES = CONFIG["MAX_RETRIES"]
TOP_K = CONFIG["TOP_K"]

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
    """Index all tables in the SQL database using ChromaDB as the backend.
    Args:
        sql_database: SQLDatabase instance
        chroma_db_dir: Directory for ChromaDB persistence
        
    Returns:
        Dict mapping table names to VectorStoreIndex instances
    """
    os.makedirs(chroma_db_dir, exist_ok=True)

    vector_index_dict = {}
    engine = sql_database.engine

    logger.info(f" [00] Creating persistent Chroma client at: {chroma_db_dir}")
    chroma_client = chromadb.PersistentClient(path=chroma_db_dir)

    for table_name in sql_database.get_usable_table_names():
        logger.info(f"[01] Processing table: {table_name}")
        
        try:
            # Create or get collection - ChromaDB handles persistence internally
            collection = chroma_client.get_or_create_collection(name=f"table_{table_name}")
            
            # Check if collection already has data
            if collection.count() == 0:
                logger.info(f"[02] Building new index for empty collection: {table_name}")
                
                # Fetch data from database
                with engine.connect() as conn:
                    result = conn.execute(text(f'SELECT * FROM "{table_name}"'))
                    col_names = list(result.keys())
                    rows = result.fetchall()
                
                if not rows:
                    logger.warning(f"[02.1] Table {table_name} is empty, skipping...")
                    continue
                
                logger.info(f"[02.2] Converting {len(rows)} rows to structured text")
                row_texts = [
                    " | ".join([f"{col}={val}" for col, val in zip(col_names, row)])
                    for row in rows
                ]
                
                # Create TextNodes with proper IDs
                nodes = [
                    TextNode(
                        text=row_text, 
                        id_=f"{table_name}_row_{idx}"
                    ) 
                    for idx, row_text in enumerate(row_texts)
                ]
                
                logger.info(f"[02.3] Creating vector store for table: {table_name}")
                vector_store = ChromaVectorStore(chroma_collection=collection)
                
                # Create index - this will automatically add nodes to ChromaDB
                logger.info(f"[02.4] Building vector index with {len(nodes)} nodes")
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                index = VectorStoreIndex(
                    nodes, 
                    storage_context=storage_context,
                    embed_model=get_embedding_func()
                    )
                
                logger.info(f"[02.5] Index created successfully for table: {table_name}")
                
            else:
                logger.info(f"[03] Reusing existing collection with {collection.count()} items: {table_name}")
                
                # Create vector store from existing collection
                vector_store = ChromaVectorStore(chroma_collection=collection)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                
                # Create index from existing vector store
                index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    storage_context=storage_context,
                    embed_model=get_embedding_func()
                )
            
            vector_index_dict[table_name] = index
            logger.info(f"[04] Successfully indexed table: {table_name}")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to index table {table_name}: {str(e)}")
            raise
    
    logger.info(f"[05] Successfully indexed {len(vector_index_dict)} tables")
    return vector_index_dict


def run_db_population(
    engine, 
    chroma_db_dir=CHROMA_DB_DIR
):
    
    logger = setup_logger("vector_db_population_logger", LOG_FILE)
    logger.info(" ")
    logger.info("--------++++++++Starting DB Population stage.....")

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