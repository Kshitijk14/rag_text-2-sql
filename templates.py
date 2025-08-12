import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


project_name = "rag_query_resolution"

list_of_files = [
    "data/.gitkeep",
    "chromadb/.gitkeep",
    "results/.gitkeep",
    "outputs/.gitkeep",
    # "notebooks/trials.ipynb",
    
    "rag_pipeline/__init__.py",
    "rag_pipeline/stage_01_populate_db.py",
    "rag_pipeline/stage_02_query_data.py",
    "rag_pipeline/stage_03_rag_eval.py",
    
    "utils/__init__.py",
    "utils/config.py",
    "utils/logger.py",
    "utils/llm/get_prompt_temp.py",
    "utils/llm/get_llm_func.py",
    "utils/eval/get_retrieval_eval_metrics.py",
    "utils/eval/get_generation_eval_metrics.py",
    
    "main.py",
    "test_rag.py",
    
    "params.yaml",
    "DVC.yaml",
    ".env.local",
    "requirements.txt",
]


for filepath in list_of_files:
    filepath = Path(filepath) #to solve the windows path issue
    filedir, filename = os.path.split(filepath) # to handle the project_name folder


    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    else:
        logging.info(f"{filename} is already exists")