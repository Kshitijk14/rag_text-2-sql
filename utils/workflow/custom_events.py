from llama_index.core.workflow.events import Event


class TableRetrievedEvent(Event):
    tables: list
    query_str: str

class SchemaProcessedEvent(Event):
    table_schema: str
    query_str: str

class SQLPromptReadyEvent(Event):
    t2s_prompt: str
    query_str: str
    table_schema: str
    retry_count: int = 0
    error_message: str = ""

class SQLGeneratedEvent(Event):
    sql_query: str
    query_str: str
    table_schema: str
    retry_count: int = 0
    error_message: str = ""

class SQLParsedEvent(Event):
    sql_query: str
    query_str: str
    table_schema: str
    retry_count: int = 0
    error_message: str = ""

# class SQLResultsEvent(Event):
#     context_str: str
#     sql_query: str
#     query_str: str
#     success: bool = True
#     retry_count: int = 0

class SQLResultSuccessEvent(Event):
    context_str: str
    sql_query: str
    query_str: str

class SQLResultFailureEvent(Event):
    error_message: str
    sql_query: str
    query_str: str
    table_schema: str
    retry_count: int = 0

class RetryPromptEvent(Event):
    query_str: str
    table_schema: str
    retry_count: int
    error_message: str

class ResponsePromptReadyEvent(Event):
    query_str: str
    rs_prompt: str