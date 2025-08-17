import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


project_name = "rag_text-2-sql"

list_of_files = [
    "data/.gitkeep",
    "db/chromadb/.gitkeep",
    "db/sqlite/.gitkeep",
    
    "results/.gitkeep",
    "outputs/.gitkeep",
    "notebooks/trials.ipynb",

    # pipeline
    "pipelines/data_pipeline/__init__.py",
    "pipelines/data_pipeline/stage_01_db_connect.py",
    "pipelines/data_pipeline/stage_02_data_flow.py",
    "pipelines/data_pipeline/stage_03_data_prepro.py",
    "pipelines/data_pipeline/stage_04_data_validate.py",
    "pipelines/data_pipeline/stage_05_data_transform.py",

    "pipelines/rag_pipeline/__init__.py",
    "pipelines/rag_pipeline/stage_01_populate_db.py",
    "pipelines/rag_pipeline/stage_02_query_data.py",

    "pipelines/evaluation_pipeline/__init__.py",
    "pipelines/evaluation_pipeline/stage_01_retrieve.py",
    "pipelines/evaluation_pipeline/stage_02_rank.py",
    "pipelines/evaluation_pipeline/stage_03_aggregate.py",
    "pipelines/evaluation_pipeline/stage_04_generate.py",

    # utils
    "utils/__init__.py",
    "utils/config.py",
    "utils/logger.py",
    
    "utils/helpers/__init__.py",
    
    "utils/llm/__init__.py",
    "utils/llm/get_prompt_temp.py",
    "utils/llm/get_llm_func.py",

    "utils/eval/__init__.py",
    "utils/eval/get_retrieval_eval_metrics.py",
    "utils/eval/get_generation_eval_metrics.py",

    # main
    "main.py",
    "app.py",

    # tests
    "tests/__init__.py",
    "tests/test_workflow_arch.py",

    "tests/test_table_retrieval.py",
    "tests/test_row_retrieval.py",
    "tests/test_table_relationships.py",
    
    "tests/test_data_pipeline.py",
    "tests/test_rag_pipeline.py",
    "tests/test_evaluation_pipeline.py",
    
    "tests/test_app.py",
    "tests/test_main.py",
    
    # docs
    "docs/INDEX.md",
    "docs/REFERENCE.md",
    "docs/ARCHITECTURE.md",
    "docs/INSTALLATION.md",
    "docs/USAGE.md",
    "docs/CONTRIBUTING.md",

    "params.yaml",
    "DVC.yaml",
    ".env.local",
    ".env.example",
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