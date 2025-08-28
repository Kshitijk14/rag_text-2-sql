import os

from typing import Dict

from llama_index.core import VectorStoreIndex
from llama_index.core.storage import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

from ..helpers.population_helpers import (
    create_chroma_client,
    get_chroma_collection,
    fetch_table_rows,
    prepare_new_nodes,
    update_index
)
from utils.llm.get_llm_func import get_embedding_func


def index_all_tables_with_chroma(engine, sql_database, chroma_db_dir: str, logger) -> Dict[str, VectorStoreIndex]:
    """Main driver to index all tables in the SQL database with ChromaDB."""
    os.makedirs(chroma_db_dir, exist_ok=True)
    
    vector_index_dict = {}
    chroma_client = create_chroma_client(chroma_db_dir, logger)

    for table_name in sql_database.get_usable_table_names():
        try:
            collection = get_chroma_collection(chroma_client, table_name, logger)
            col_names, rows = fetch_table_rows(engine, table_name, logger)
            if not rows:
                continue

            new_nodes = prepare_new_nodes(collection, table_name, col_names, rows, logger)
            index = update_index(collection, new_nodes) # <-- uses ChromaVectorStore wrapper

            vector_index_dict[table_name] = index
            logger.info(f"[04] Table {table_name} indexed successfully (total={collection.count()})")

        except Exception as e:
            logger.error(f"[ERROR] Failed to index table {table_name}: {str(e)}")
            raise

    logger.info(f"[05] Successfully indexed {len(vector_index_dict)} tables")
    return vector_index_dict

def load_all_indexes_from_chroma(chroma_db_dir: str, logger) -> Dict[str, VectorStoreIndex]:
    """Reload existing Chroma collections and wrap them in VectorStoreIndex objects."""
    if not os.path.exists(chroma_db_dir):
        logger.error(f"[ERROR] Chroma DB directory not found: {chroma_db_dir}")
        return {}

    vector_index_dict = {}
    chroma_client = create_chroma_client(chroma_db_dir, logger)

    # list collections already stored in Chroma
    collections = chroma_client.list_collections()
    logger.info(f"[00] Found {len(collections)} collections in Chroma.")

    for collection in collections:
        try:
            raw_name = collection.name
            table_name = raw_name.replace("table_", "", 1)  # -> normalize to match SQLTableSchema
            
            # wrap raw collection inside LlamaIndex ChromaVectorStore
            vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context,
                embed_model=get_embedding_func()  # -> (safe) embeddings are persisted
            )
            
            vector_index_dict[table_name] = index
            logger.info(f"[01] Loaded collection for table '{raw_name}' as '{table_name}' (total={collection.count()})")
        except Exception as e:
            logger.error(f"[ERROR] Failed to load collection {collection.name}: {e}")
            raise

    logger.info(f"[02] Successfully loaded {len(vector_index_dict)} indexes from Chroma.")
    return vector_index_dict
