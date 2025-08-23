import os
import gc
import traceback

from pathlib import Path

from llama_index.core import Settings
from llama_index.core.workflow import (
    Workflow, 
    step, 
    StartEvent, 
    StopEvent,
)

from utils.config import CONFIG
from utils.logger import setup_logger

from .stage_03_retrievals import get_table_context_and_rows_str

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


class Text2SQLWorkflow(Workflow):
    def __init__(self, obj_retriever, sql_database, vector_index_dict, sql_retriever, top_n, response_synthesis_prompt, logger=None):
        super().__init__()
        self.obj_retriever = obj_retriever
        self.sql_database = sql_database
        self.vector_index_dict = vector_index_dict
        self.sql_retriever = sql_retriever
        self.top_n = top_n
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
        sql_response = await Settings.llm.acomplete(ev.t2s_prompt)
        
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
                retry_event.error_message = analyze_sql_error(error_msg, ev.sql_query, ev.table_schema)
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
        answer = await Settings.llm.acomplete(ev.rs_prompt)
        
        return StopEvent(result=str(answer))

# # Runner
# async def run_text2sql_workflow_row(query: str):
#     workflow = Text2SQLWorkflow(timeout=480)
#     result = await workflow.run(query=query)
#     return result


async def run_text2sql_workflow(obj_retriever, sql_database, vector_index_dict, sql_retriever, top_n=TOP_N, response_synthesis_prompt: str = RESPONSE_SYNTHESIS_PROMPT, query_text: str = QUERY_TEXT):
    logger = setup_logger("retrievals_logger", LOG_FILE)
    logger.info(" ")
    logger.info("--------++++++++Starting Query Workflow stage.....")

    try:
        workflow = Text2SQLWorkflow(obj_retriever, sql_database, vector_index_dict, sql_retriever, top_n, response_synthesis_prompt)
        result = await workflow.run(query=query_text)
        
        # result = await run_text2sql_workflow_row(query_text)
        print(result)

        logger.info("--------++++++++Query Workflow stage successfully completed.")
        logger.info(" ")
        
        # Manual memory cleanup
        del workflow
        gc.collect()
        
        return result
    except Exception as e:
        logger.error(f"Error at [Stage 03]: {e}")
        logger.debug(traceback.format_exc())