import os, argparse, traceback
from utils.logger import setup_logger
from utils.config import CONFIG
from rag_pipeline.stage_01_data_prep import run_data_prep


# configurations & setup logging
QUERY_TEXT = CONFIG["QUERY_TEXT"]
LOG_PATH = CONFIG["LOG_PATH"]
LOG_DIR = os.path.join(os.getcwd(), LOG_PATH)
os.makedirs(LOG_DIR, exist_ok=True)  # Create the logs directory if it doesn't exist
LOG_FILE = os.path.join(LOG_DIR, "main.log")


def main():
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
            run_data_prep()
            # logger.info("Already Done. Skipping...")
            logger.info("----------FINISHED [STAGE 01] DATA PREPARATION----------")
            logger.info(" ")
        except Exception as e:
            logger.error(f"ERROR RUNNING [STAGE 01] DATA PREPARATION: {e}")
            logger.debug(traceback.format_exc())
            return
        
        logger.info("////--//--//----FINISHED [PIPELINE 01] RAG PIPELINE----//--//--////")
        logger.info(" ")
    except Exception as e:
        logger.error(f"ERROR RUNNING [PIPELINE 01] RAG PIPELINE: {e}")
        logger.debug(traceback.format_exc())
        return


if __name__ == "__main__":
    main()