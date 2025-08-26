import os
import gc
import traceback
import asyncio

from pathlib import Path
from typing import List, Dict

from llama_index.core import Settings
from llama_index.core.objects import SQLTableSchema
from llama_index.core.workflow import (
    Workflow, 
    step, 
    StartEvent, 
    StopEvent,
)

from utils.config import CONFIG
from utils.logger import setup_logger

from utils.helpers.common import TableInfo
from utils.helpers.retriever_helpers import (
    build_object_index,
    create_retrievers
)
from utils.helpers.summary_helpers import (
    load_summaries_from_sqlite,
    filter_valid_summaries,
    load_summaries_from_json,
    filter_valid_summaries_from_json
)
from utils.llm.get_prompt_temp import RESPONSE_SYNTHESIS_PROMPT
from utils.workflow.custom_events import (
    TableRetrievedEvent,
    SchemaProcessedEvent,
    SQLPromptReadyEvent,
    SQLGeneratedEvent,
    SQLParsedEvent,
    SQLResultsEvent,
    ResponsePromptReadyEvent,
)
from utils.workflow.custom_fallbacks import (
    extract_sql_from_response,
    analyze_sql_error,
    create_t2s_prompt,
)
from utils.helpers.workflow_helpers import parse_response_to_sql
from utils.llm.get_llm_func import get_llm_func


# configs
LOG_PATH = Path(CONFIG["LOG_PATH"])
CHINOOK_DB_PATH = Path(CONFIG["CHINOOK_DB_PATH"])
SQLITE_DB_DIR = Path(CONFIG["SQLITE_DB_DIR"])
CHROMA_DB_DIR = Path(CONFIG["CHROMA_DB_DIR"])
MAX_RETRIES = CONFIG["MAX_RETRIES"]
TOP_K = CONFIG["TOP_K"]
TOP_N = CONFIG["TOP_N"]
WORKFLOW_TIMEOUT = CONFIG["WORKFLOW_TIMEOUT"]
QUERY_TEXT = CONFIG["QUERY_TEXT"]

# logger
LOG_DIR = os.path.join(os.getcwd(), str(LOG_PATH))
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "stage_03_query_workflow.log")


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
            logger.info(f"[02] Retrieved {len(relevant_nodes)} relevant rows for table: {table_name}")
            
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


class Text2SQLWorkflow(Workflow):
    def __init__(self, obj_retriever, sql_database, vector_index_dict, sql_retriever, top_n, local_model, response_synthesis_prompt, logger):
        super().__init__()
        self.obj_retriever = obj_retriever
        self.sql_database = sql_database
        self.vector_index_dict = vector_index_dict
        self.sql_retriever = sql_retriever
        self.top_n = top_n
        self.local_model = local_model
        self.response_synthesis_prompt = response_synthesis_prompt
        self.logger = logger
    

    @step
    async def input_step(self, ev: StartEvent) -> TableRetrievedEvent:
        self.logger.info(f"[Step 01] Process initial query and retrieve relevant tables")
        query = ev.query

        self.logger.info(f" - Use object retriever built from your table summaries")
        tables = self.obj_retriever.retrieve(query)  # candidate schemas
        self.logger.info(f" - Retrieved {len(tables)} candidate tables for query: {query}")
        
        return TableRetrievedEvent(
            tables=tables, 
            query_str=query
        )

    @step
    async def table_output_parser_step(self, ev: TableRetrievedEvent) -> SchemaProcessedEvent:
        self.logger.info(f"[Step 02] Parsing schemas and retrieving relevant rows for query: {ev.query_str}")

        self.logger.info(f" - Enriching context function with vector row retrieval for tables: {ev.tables}")
        schema_str = get_table_context_and_rows_str(
            self.sql_database,
            self.vector_index_dict,
            ev.query_str, 
            ev.tables,
            self.top_n,
            self.logger
            )

        return SchemaProcessedEvent(
            table_schema=schema_str, 
            query_str=ev.query_str
        )

    @step
    async def text2sql_prompt_step(self, ev: SchemaProcessedEvent | SQLResultsEvent) -> SQLPromptReadyEvent:
        self.logger.info(f"[Step 03] Creating SQL prompt for query: {ev.query_str}")
        if isinstance(ev, SchemaProcessedEvent):
            table_schema = ev.table_schema
            query_str = ev.query_str
            retry_count = 0
            error_message = ""
        else:
            table_schema = getattr(ev, 'table_schema', '')
            query_str = ev.query_str
            retry_count = getattr(ev, 'retry_count', 0) + 1
            error_message = getattr(ev, 'error_message', '')

        prompt = create_t2s_prompt(table_schema, query_str, retry_count, error_message)
        
        return SQLPromptReadyEvent(
            t2s_prompt=prompt,
            query_str=query_str,
            table_schema=table_schema,
            retry_count=retry_count,
            error_message=error_message
        )

    @step
    async def text2sql_llm_step(self, ev: SQLPromptReadyEvent) -> SQLGeneratedEvent:
        self.logger.info(f"[Step 04] Running LLM to generate SQL for query: {ev.query_str}")
        # sql_response = await self.local_model.acomplete(ev.t2s_prompt)
        # sql_response = await self.local_model.acomplete("SELECT 1;")
        
        try:
            sql_response = await asyncio.wait_for(
                self.local_model.acomplete(ev.t2s_prompt),
                timeout=300  # step-specific
            )
        except asyncio.TimeoutError:
            self.logger.error("LLM call exceeded 300s, aborting step.")
            # maybe return a fallback event instead of crashing
            raise

        return SQLGeneratedEvent(
            sql_query=str(sql_response).strip(),
            query_str=ev.query_str,
            table_schema=ev.table_schema,
            retry_count=ev.retry_count,
            error_message=ev.error_message
        )

    @step
    async def sql_output_parser_step(self, ev: SQLGeneratedEvent) -> SQLParsedEvent:
        self.logger.info(f"[Step 05] Parsing LLM response to extract clean SQL for query: {ev.query_str}")
        try:
            clean_sql = parse_response_to_sql(ev.sql_query)  # primary parser
        except Exception:
            clean_sql = extract_sql_from_response(ev.sql_query, self.logger)  # fallback
        
        if not clean_sql:
            clean_sql = extract_sql_from_response(ev.sql_query, self.logger)

        self.logger.info(f"Attempt #{ev.retry_count + 1}")
        self.logger.info(f"LLM Response: {ev.sql_query}")
        self.logger.info(f"Cleaned SQL: {clean_sql}")

        return SQLParsedEvent(
            sql_query=clean_sql,
            query_str=ev.query_str,
            table_schema=ev.table_schema,
            retry_count=ev.retry_count,
            error_message=ev.error_message
        )

    @step
    async def sql_retriever_step(self, ev: SQLParsedEvent) -> SQLResultsEvent:
        self.logger.info(f"[Step 06] Executing SQL for query: {ev.query_str}")
        try:
            results = self.sql_retriever.retrieve(ev.sql_query)
            self.logger.info(f"[SUCCESS] Executed on Attempt #{ev.retry_count + 1}")

            return SQLResultsEvent(
                context_str=str(results),
                sql_query=ev.sql_query,
                query_str=ev.query_str,
                success=True
            )
        except Exception as e:
            # self.logger.error(f"Execution failed (Attempt #{ev.retry_count + 1}): {e}")
            error_msg = str(e)
            self.logger.error(f"Execution failed (Attempt #{ev.retry_count + 1}): {error_msg}")

            if ev.retry_count < MAX_RETRIES:
                retry_event = SQLResultsEvent(
                    context_str="",
                    sql_query=ev.sql_query,
                    query_str=ev.query_str,
                    success=False,
                    retry_count=ev.retry_count + 1,
                )
                retry_event.retry_count = ev.retry_count + 1
                retry_event.error_message = analyze_sql_error(error_msg, ev.sql_query, ev.table_schema, self.logger)
                retry_event.table_schema = ev.table_schema
                
                return retry_event
            else:
                return SQLResultsEvent(
                    context_str=(f"Failed after {MAX_RETRIES+1} attempts. Final error: {error_msg}"),
                    sql_query=ev.sql_query,
                    query_str=ev.query_str,
                    success=False,
                    retry_count=ev.retry_count + 1,
                )

    @step
    async def retry_handler_step(self, ev: SQLResultsEvent) -> SQLPromptReadyEvent:
        self.logger.info(f"[Step 07] Handling retry for query: {ev.query_str}")
        if ev.success:
            return None
        
        return SQLPromptReadyEvent(
            t2s_prompt="",  # regenerated later
            query_str=ev.query_str,
            table_schema=getattr(ev, 'table_schema', ''),
            retry_count=ev.retry_count,
            error_message=getattr(ev, 'error_message', 'Unknown error')
        )

    @step
    async def response_synthesis_prompt_step(self, ev: SQLResultsEvent) -> ResponsePromptReadyEvent:
        self.logger.info(f"[Step 08] Preparing synthesis prompt for query: {ev.query_str}")
        if not ev.success:
            return None
        prompt = self.response_synthesis_prompt.format(
            query_str=ev.query_str,
            context_str=ev.context_str,
            sql_query=ev.sql_query
        )
        
        return ResponsePromptReadyEvent(
            query_str=ev.query_str,
            rs_prompt=prompt
        )

    @step
    async def response_synthesis_llm_step(self, ev: ResponsePromptReadyEvent) -> StopEvent:
        self.logger.info(f"[Step 09] Generating final answer for query: {ev.query_str}")
        answer = await self.local_model.acomplete(ev.rs_prompt)
        
        return StopEvent(result=str(answer))


async def run_text2sql_workflow(
    summary_engine, 
    engine, 
    sql_database, 
    table_node_mapping, 
    vector_index_dict, 
    sqlite_db_dir=SQLITE_DB_DIR,
    top_k=TOP_K, 
    top_n=TOP_N, 
    response_synthesis_prompt: str = RESPONSE_SYNTHESIS_PROMPT, 
    workflow_timeout: float = WORKFLOW_TIMEOUT,
    query_text: str = QUERY_TEXT
):
    logger = setup_logger("workflow_logger", LOG_FILE)

    try:
        try:
            logger.info(" ")
            logger.info("--------++++++++Starting Retrieval sub-stage.....")
            
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
            
            logger.info("--------++++++++Retrieval sub-stage successfully completed.")
            logger.info(" ")
        except Exception as e:
            logger.error(f"Error at retrieval sub-stage: {e}")
            logger.debug(traceback.format_exc())

        try:
            logger.info(" ")
            logger.info("--------++++++++Starting Query Workflow stage.....")
            
            local_model = get_llm_func()
            workflow = Text2SQLWorkflow(obj_retriever, sql_database, vector_index_dict, sql_retriever, top_n, local_model, response_synthesis_prompt, logger)
            
            # result = await asyncio.wait_for(
            #         workflow.run(query=query_text), 
            #         timeout=workflow_timeout
            #     )
            result = await workflow.run(query=query_text, timeout=workflow_timeout)
            logger.info(f"Stage 03 completed. Final Result:\n{result}")
            
            print(result)

            logger.info("--------++++++++Query Workflow stage successfully completed.")
            logger.info(" ")
        except Exception as e:
            logger.error(f"Error at Query Workflow stage: {e}")
            logger.debug(traceback.format_exc())

        finally:
            # Manual memory cleanup
            del summary_engine, engine, sql_database, table_node_mapping, vector_index_dict, rows, table_schema_objs, obj_index, obj_retriever, sql_retriever
            gc.collect()
        
        return result
    except Exception as e:
        logger.error(f"Error at [Stage 03]: {e}")
        logger.debug(traceback.format_exc())