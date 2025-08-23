import os
import gc
import traceback

from pathlib import Path
from typing import List, Dict

from llama_index.core import VectorStoreIndex
from llama_index.core.objects import (
    ObjectIndex,
    SQLTableSchema,
)
from llama_index.core.retrievers import SQLRetriever

from utils.config import CONFIG
from utils.logger import setup_logger

from utils.helpers.common import TableInfo
from utils.llm.get_llm_func import get_embedding_func
from utils.helpers.summary_helpers import (
    load_summaries_from_sqlite,
    filter_valid_summaries,
    load_summaries_from_json,
    filter_valid_summaries_from_json
)


# configs
LOG_PATH = Path(CONFIG["LOG_PATH"])
CHINOOK_DB_PATH = Path(CONFIG["CHINOOK_DB_PATH"])
SQLITE_DB_DIR = Path(CONFIG["SQLITE_DB_DIR"])
CHROMA_DB_DIR = Path(CONFIG["CHROMA_DB_DIR"])
MAX_RETRIES = CONFIG["MAX_RETRIES"]
TOP_K = CONFIG["TOP_K"]
TOP_N = CONFIG["TOP_N"]
QUERY_TEXT = CONFIG["QUERY_TEXT"]

# logger
LOG_DIR = os.path.join(os.getcwd(), str(LOG_PATH))
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "stage_02_retrievals.log")


def build_object_index(table_schema_objs, table_node_mapping, logger):
    """Build ObjectIndex for retrieval from table summaries."""
    logger.info("Building object index for table retrieval")
    obj_index = ObjectIndex.from_objects(
        table_schema_objs,
        table_node_mapping,
        VectorStoreIndex,
        embed_model=get_embedding_func(),
    )
    return obj_index


def create_retrievers(sql_database, obj_index, top_k: int, logger):
    """Create object retriever and SQL retriever."""
    logger.info("Creating retrievers for query execution")
    obj_retriever = obj_index.as_retriever(similarity_top_k=top_k)
    sql_retriever = SQLRetriever(sql_database)
    return obj_retriever, sql_retriever


def get_table_context_str(sql_database, table_schema_objs: List[SQLTableSchema], logger) -> Dict[str, str]:
    """Get table context (schema + summary) for multiple tables.
    
    Returns a dict {table_name: schema_context}
    """
    table_context = {}
    
    for table_schema_obj in table_schema_objs:
        try:            
            # pull schema directly from DB
            table_info = sql_database.get_single_table_info(
                table_schema_obj.table_name
            )
            if table_schema_obj.context_str:
                table_info += f" The table description is: {table_schema_obj.context_str}"
            
            table_context[table_schema_obj.table_name] = table_info
        
        except Exception as e:
            logger.error(f"Skipping table {table_schema_obj.table_name}: {e}")
    
    return table_context

def get_table_context_and_rows_str(sql_database, vector_index_dict, query_str: str, table_schema_objs: List[TableInfo], top_n: int, logger) -> str:
    """Get table context string (schema + relevant example rows)."""
    
    base_contexts = get_table_context_str(sql_database, table_schema_objs, logger)
    context_strs = []
    
    for table_name, schema_context in base_contexts.items():
        try:
            # Check if we have a vector index for this table
            if table_name not in vector_index_dict:
                logger.warning(f"No vector index found for table: {table_name}")
                context_strs.append(schema_context)
                continue
            
            logger.info(f"[01] Retrieving example rows for table: {table_name}")
            vector_retriever = vector_index_dict[table_name].as_retriever(
                similarity_top_k=top_n
            )
            
            relevant_nodes = vector_retriever.retrieve(query_str)
            logger.info(f"[02] Retrieved {len(relevant_nodes)} relevant nodes for table: {table_name}")
            
            if relevant_nodes:
                row_context = "\nHere are some relevant example rows (column=value):\n"
                row_context += "\n".join([f"- {node.get_content()}" for node in relevant_nodes])
                schema_context += "\n" + row_context
            else:
                logger.info(f"[02.1] No relevant rows found for query in table: {table_name}")
            
            context_strs.append(schema_context)
        
        except Exception as e:
            logger.error(f"Failed to enrich context for table {table_name}: {str(e)}")
            context_strs.append(schema_context)  # fallback to just schema
    
    return "\n\n".join(context_strs)


def run_retrievals(
    summary_engine,
    engine, 
    sql_database, 
    table_node_mapping, 
    vector_index_dict,
    sqlite_db_dir=SQLITE_DB_DIR, 
    top_k=TOP_K, 
    top_n=TOP_N,
    query_text=QUERY_TEXT
):
    
    logger = setup_logger("retrievals_logger", LOG_FILE)
    logger.info(" ")
    logger.info("--------++++++++Starting Retriever Creation stage.....")

    try:
        # rows = load_summaries_from_sqlite(summary_engine, logger)
        # table_schema_objs = filter_valid_summaries(rows, engine, logger)

        rows = load_summaries_from_json(sqlite_db_dir, logger)
        table_schema_objs = filter_valid_summaries_from_json(rows, engine, logger)

        obj_index = build_object_index(table_schema_objs, table_node_mapping, logger)

        obj_retriever, sql_retriever = create_retrievers(sql_database, obj_index, top_k, logger)
        
        # table_parser_component = get_table_context_and_rows_str(
        #     sql_database, vector_index_dict, query_text, table_schema_objs, top_n, logger
        # )
        # logger.info(f"Updated table context with rows:\n{table_parser_component}")
        
        logger.info("--------++++++++Retriever Creation stage successfully completed.")
        logger.info(" ")
        
        # Manual memory cleanup
        del rows, table_schema_objs, obj_index
        gc.collect()
        
        return obj_retriever, sql_retriever
    except Exception as e:
        logger.error(f"Error at [Stage 02]: {e}")
        logger.debug(traceback.format_exc())

if __name__ == "__main__":
    run_retrievals()