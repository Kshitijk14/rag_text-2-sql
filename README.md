# rag_text-2-sql

## setup:

1. fork -> clone -> cd to root
2. setup the local sample DBeaver data path in "params.yaml", under "CHINOOK_DB_PATH"
3. download local llm from ollama: ``ollama pull qwen3:0.6b``
4. install libraries: ``uv add -r requirements.txt``

## run:

1. start phoenix server: ``uv run -m phoenix.server.main serve``
2. start ollama server: ``ollama server``
3. run pipeline: ``uv run main.py``

## On a surface level

![1755674544635](image/README/1755674544635.png)
