import os
import time
import re
import gc
import traceback
import json

from pathlib import Path
import pandas as pd

from sqlalchemy import create_engine, text, inspect
from llama_index.core.program import LLMTextCompletionProgram

from utils.config import CONFIG
from utils.logger import setup_logger

from utils.helpers.common import TableInfo
from utils.llm.get_prompt_temp import TABLE_INFO_PROMPT
from utils.llm.get_llm_func import get_llm_func
from utils.helpers.summary_helpers import ( 
    create_summaries_table, 
    dump_summaries_sqlite, 
    dump_summaries_json 
)


# configs
LOG_PATH = Path(CONFIG["LOG_PATH"])
CHINOOK_DB_PATH = Path(CONFIG["CHINOOK_DB_PATH"])
SQLITE_DB_DIR = Path(CONFIG["SQLITE_DB_DIR"])
MAX_RETRIES = CONFIG["MAX_RETRIES"]

# logger
LOG_DIR = os.path.join(os.getcwd(), str(LOG_PATH))
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "stage_01_data_prep.log")


def text_completion_program(table_info_prompt_temp: str):
    
    return LLMTextCompletionProgram.from_defaults(
        output_cls=TableInfo,
        prompt_template_str=table_info_prompt_temp,
        llm=get_llm_func(),
    )


def extract_first_json_block(text: str, logger):
    logger.info("Extracting the first valid JSON object from text, ignoring extra trailing text.")
    
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        logger.error(f"No JSON object found in text: {text}")
        logger.debug(traceback.format_exc())
    
    try:
        logger.info(f"Extracted JSON: {match.group()}")
        return json.loads(match.group())
    except Exception as e:
        logger.error(f"Failed to parse JSON: {e}\nRaw text: {text}")
        logger.debug(traceback.format_exc())
        return None


def prep_summary_engine(sqlite_db_dir: Path, main_db_dir: Path, logger):
    """Prepare the SQLite engine for storing table summaries."""
    os.makedirs(sqlite_db_dir, exist_ok=True)
    summary_db_path = os.path.join(sqlite_db_dir, "table_summaries.db")

    try:
        logger.info(f"Creating SQLite DB Engine for the new summaries database: {summary_db_path}")
        summary_engine = create_engine(f"sqlite:///{summary_db_path}")

        logger.info(f"Creating SQLite DB Engine for the existing Chinook database at {main_db_dir}")
        engine = create_engine(f"sqlite:///{main_db_dir}")
        inspector = inspect(engine)

        return summary_engine, summary_db_path, engine, inspector
    except Exception as e:
        logger.error(f"Error preparing summary engine: {e}")
        return None, None, None, None

def summary_parser(program, df_str, inspector, table, logger):
    raw_output = program(
        table_str=df_str,
        exclude_table_name_list=str(list(inspector.get_table_names())),
    )

    logger.info(f"Normalize LLM output")
    if isinstance(raw_output, str):
        parsed_dict = extract_first_json_block(raw_output, logger)
    elif isinstance(raw_output, dict):
        parsed_dict = raw_output
    elif isinstance(raw_output, TableInfo):
        parsed_dict = raw_output.model_dump()
    else:
        logger.error(f"Unexpected return type: {type(raw_output)}")
        logger.debug(traceback.format_exc())

    table_info = TableInfo(
        table_name=table,
        table_summary=parsed_dict["table_summary"],
    )

    logger.info(f"Processed table: {table_info.table_name}")

    return table_info

def generate_table_summary(
    program, 
    summary_engine, 
    summary_db_path, 
    engine, 
    inspector, 
    sqlite_db_dir: Path, 
    max_retries: int, 
    logger
):
    
    table_infos = []
    create_summaries_table(summary_engine, logger)

    logger.info("Generating table summaries...")
    with engine.connect() as conn:
        existing_tables = set()
        
        logger.info("Fetching existing summaries from the summaries database...")
        with summary_engine.connect() as summary_conn:
            rows = summary_conn.execute(text("SELECT table_name FROM table_summaries")).fetchall()
            existing_tables = {row[0] for row in rows}
            logger.info(f"Found {len(existing_tables)} existing summaries in DB")
        
        for idx, table in enumerate(inspector.get_table_names()):
            if table in existing_tables:
                logger.info(f" - Skipping table '{table}' — summary already exists.")
                continue
            
            logger.info(f" - Processing new table: {table}")
            df = pd.read_sql(f"SELECT * FROM {table} LIMIT 10;", conn)
            df_str = df.to_csv(index=False)

            table_info = None
            for attempt in range(max_retries):
                try:
                    table_info = summary_parser(program, df_str, inspector, table, logger)
                    break  # success → next table

                except Exception as e:
                    logger.error(f"Error with attempt {attempt+1} for {table}: {e}")
                    logger.debug(traceback.format_exc())
                    time.sleep(2)

            if table_info:
                table_infos.append(table_info)
                
                try:
                    logger.info(f"Saving table summary for {table_info.table_name} immediately to summaries DB")
                    dump_summaries_sqlite(summary_engine, table_infos, logger)
                    dump_summaries_json(sqlite_db_dir, table_infos, logger)

                except Exception as e:
                    logger.error(f"Failed to save table summary for {table_info.table_name}: {e}")
                    continue

    return table_infos, summary_db_path


def run_data_preparation(
    table_info_prompt_temp: str = TABLE_INFO_PROMPT, 
    sqlite_db_dir: Path = SQLITE_DB_DIR, 
    main_db_dir: Path = CHINOOK_DB_PATH, 
    max_retries: int = MAX_RETRIES
):
    
    logger = setup_logger("data_preparation_logger", LOG_FILE)
    logger.info(" ")
    logger.info("--------++++++++Starting Data Preparation stage.....")

    try:
        program = text_completion_program(table_info_prompt_temp)

        summary_engine, summary_db_path, engine, inspector = prep_summary_engine(sqlite_db_dir, main_db_dir, logger)

        table_infos, summary_db_path = generate_table_summary(
            program, summary_engine, summary_db_path, engine, inspector, sqlite_db_dir, max_retries, logger
        )

        logger.info("--------++++++++Data Preparation stage successfully completed.")
        logger.info(" ")
        
        logger.info("\n FINAL TABLE SUMMARIES")
        for t in table_infos:
            logger.info(f"- {t.table_name}: {t.table_summary}")

        logger.info(f"\nSaved {len(table_infos)} summaries")
        
        # Manual memory cleanup
        del program, summary_db_path, inspector, table_infos
        gc.collect()
        
        return summary_engine, engine
    except Exception as e:
        logger.error(f"Error at [Stage 01]: {e}")
        logger.debug(traceback.format_exc())

if __name__ == "__main__":
    run_data_preparation()