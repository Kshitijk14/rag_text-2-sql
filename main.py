import os
import argparse
import traceback
import asyncio

import phoenix as px
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from utils.config import CONFIG
from utils.logger import setup_logger
from rag_pipeline.stage_01_data_prep import run_data_preparation
from rag_pipeline.stage_02_populate_vector_db import run_db_population
from rag_pipeline.stage_03_retrievals import run_retrievals
from rag_pipeline.stage_04_query_workflow import run_text2sql_workflow


# configurations & setup logging
QUERY_TEXT = CONFIG["QUERY_TEXT"]
LOG_PATH = CONFIG["LOG_PATH"]
LOG_DIR = os.path.join(os.getcwd(), LOG_PATH)
os.makedirs(LOG_DIR, exist_ok=True)  # Create the logs directory if it doesn't exist
LOG_FILE = os.path.join(LOG_DIR, "main.log")


# initialize llamaindex auto-instrumentation
endpoint = "http://localhost:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)


async def main():
    logger = setup_logger("main_logger", LOG_FILE)

    # Create CLI.
    parser = argparse.ArgumentParser(description="MAIN WORKFLOW")
    # parser.add_argument("--reset", action="store_true", help="Reset DB before population")
    # parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    
    try:
        logger.info(" ")
        logger.info("////--//--//----STARTING [PIPELINE 01] RAG PIPELINE----//--//--////")
        
        try:
            logger.info(" ")
            logger.info("----------STARTING [STAGE 01] DATA PREPARATION----------")
            summary_engine, engine = run_data_preparation()
            # logger.info("Already Done. Skipping...")
            logger.info("----------FINISHED [STAGE 01] DATA PREPARATION----------")
            logger.info(" ")
        except Exception as e:
            logger.error(f"ERROR RUNNING [STAGE 01] DATA PREPARATION: {e}")
            logger.debug(traceback.format_exc())
            return
        
        try:
            logger.info(" ")
            logger.info("----------STARTING [STAGE 02] DB POPULATION----------")
            sql_database, table_node_mapping, vector_index_dict = run_db_population(engine)
            # logger.info("Already Done. Skipping...")
            logger.info("----------FINISHED [STAGE 02] DB POPULATION----------")
            logger.info(" ")
        except Exception as e:
            logger.error(f"ERROR RUNNING [STAGE 02] DB POPULATION: {e}")
            logger.debug(traceback.format_exc())
            return
        
        try:
            logger.info(" ")
            logger.info("----------STARTING [STAGE 03] RETRIEVER CREATION----------")
            obj_retriever, sql_retriever = run_retrievals(
                summary_engine, engine, sql_database, table_node_mapping, vector_index_dict
            )
            # logger.info("Already Done. Skipping...")
            logger.info("----------FINISHED [STAGE 03] RETRIEVER CREATION----------")
            logger.info(" ")
        except Exception as e:
            logger.error(f"ERROR RUNNING [STAGE 03] RETRIEVER CREATION: {e}")
            logger.debug(traceback.format_exc())
            return
        
        # try:
        #     logger.info(" ")
        #     logger.info("----------STARTING [STAGE 04] TEXT 2 SQL WORKFLOW----------")
        #     await run_text2sql_workflow(obj_retriever, sql_database, vector_index_dict, sql_retriever)
        #     # logger.info("Already Done. Skipping...")
        #     logger.info("----------FINISHED [STAGE 04] TEXT 2 SQL WORKFLOW----------")
        #     logger.info(" ")
        # except Exception as e:
        #     logger.error(f"ERROR RUNNING [STAGE 04] TEXT 2 SQL WORKFLOW: {e}")
        #     logger.debug(traceback.format_exc())
        #     return
        
        logger.info("////--//--//----FINISHED [PIPELINE 01] RAG PIPELINE----//--//--////")
        logger.info(" ")
    except Exception as e:
        logger.error(f"ERROR RUNNING [PIPELINE 01] RAG PIPELINE: {e}")
        logger.debug(traceback.format_exc())
        return


if __name__ == "__main__":
    asyncio.run(main())