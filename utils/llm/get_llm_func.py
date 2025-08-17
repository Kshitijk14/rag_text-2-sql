from ..config import CONFIG
from ..helpers.llama_index_imports import (
    Settings, 
    Ollama, 
    HuggingFaceEmbedding
)


LOCAL_EMBEDDING_MODEL = CONFIG["LOCAL_EMBEDDING_MODEL"]
LOCAL_LLM_MODEL = CONFIG["LOCAL_LLM_MODEL"]

MODEL_TEMPERATURE = CONFIG["MODEL_TEMPERATURE"]
MODEL_REQUEST_TIMEOUT = CONFIG["MODEL_REQUEST_TIMEOUT"]
MODEL_CONTEXT_WINDOW = CONFIG["MODEL_CONTEXT_WINDOW"]

MODEL_OUTPUT_FORMAT = CONFIG["MODEL_OUTPUT_FORMAT"]
MODEL_OUTPUT_FORMAT_JSON = CONFIG["MODEL_OUTPUT_FORMAT_JSON"]
MODEL_THINKING = CONFIG["MODEL_THINKING"]


def get_embedding_func(
    local_embedding_model: str = LOCAL_EMBEDDING_MODEL
):
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=local_embedding_model
    )
    return Settings.embed_model

def get_llm_func(
    local_llm_model: str = LOCAL_LLM_MODEL, 
    model_temperature: float = MODEL_TEMPERATURE,
    model_output_format: str = MODEL_OUTPUT_FORMAT, 
    model_output_format_json: bool = MODEL_OUTPUT_FORMAT_JSON, 
    model_thinking: bool = MODEL_THINKING, 
    model_request_timeout: int = MODEL_REQUEST_TIMEOUT, 
    model_context_window: int = MODEL_CONTEXT_WINDOW
):
    Settings.llm = Ollama(
        model=local_llm_model,
        temperature=model_temperature,

        format=model_output_format,
        # json_mode=model_output_format_json,
        # thinking=model_thinking,
        
        request_timeout=model_request_timeout,

        # # Manually set the context window to limit memory usage
        # context_window=model_context_window,
        
        streaming=False,
        verbose=True,
    )
    return Settings.llm