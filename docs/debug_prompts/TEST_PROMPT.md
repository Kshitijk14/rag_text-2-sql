"
import io, os, time, re, requests, zipfile, json
import json as pyjson
import pandas as pd
from pathlib import Path
from typing import List, Dict
from pydantic import BaseModel, Field

# put data into sqlite db
from sqlalchemy import (
    create_engine,
    text,
    MetaData,
    Table,
    Column,
    String,
    Integer,
)

# setup Arize Phoenix for logging/observability
import phoenix as px

from llama_index.core import (
    Settings, 
    SQLDatabase, 
    VectorStoreIndex, 
    load_index_from_storage,
    set_global_handler,
)
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.llms.ollama import Ollama
from llama_index.core.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import SQLRetriever
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_TO_SQL_PROMPT
from llama_index.core.prompts import PromptTemplate
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import ChatResponse
from llama_index.core.callbacks import CallbackManager
from llama_index.core.schema import TextNode
from llama_index.core.storage import StorageContext
from llama_index.core.workflow import (
    Workflow, 
    step, 
    StartEvent, 
    StopEvent,
)
from llama_index.core.workflow.events import Event
from llama_index.utils.workflow import (
    draw_all_possible_flows, 
    draw_most_recent_execution,
)

DATA_DIR = Path("../data/WikiTableQuestions/csv/200-csv")
CSV_FILES = sorted([f for f in DATA_DIR.glob("*.csv")])

TABLEINFO_DIR = "../data/WikiTableQuestions_TableInfo"
os.makedirs(TABLEINFO_DIR, exist_ok=True)

MAX_RETRIES = 3
dfs = []

for csv_file in CSV_FILES:
    print(f"processing file: {csv_file}")
    try:
        df = pd.read_csv(csv_file)
        dfs.append(df)
    except Exception as e:
        print(f"Error parsing {csv_file}: {str(e)}")

class TableInfo(BaseModel):
    """Information regarding a structured table."""

    table_name: str = Field(
        ..., description="table name (must be underscores and NO spaces)"
    )
    table_summary: str = Field(
        ..., description="short, concise summary/caption of the table"
    )

PROMPT_STR = """\
    Return only a JSON object, with no explanation, no prose, no markdown, and no trailing text.
    You are to produce **only** a JSON object matching the following exact schema:

    {
        "table_name": "<short_name_in_snake_case_without_spaces>",
        "table_summary": "<short concise caption of the table>"
    }

    Example:
    {"table_name": "movie_info", "table_summary": "Summary of movie data"}

    Rules:
    - The table_name must be unique to the table, describe it clearly, and be in snake_case.
    - Do NOT output a generic table name (e.g., "table", "my_table").
    - Do NOT make the table name one of the following: {exclude_table_name_list}.
    - Do NOT include any keys other than "table_name" and "table_summary".
    - Do NOT include extra text before/after the JSON.
    - Do NOT include any other keys or text before/after the JSON.
    - Do NOT wrap in ```json.

    Table:
    {table_str}
"""

Settings.llm = Ollama(
    model="qwen3:0.6b", 
    request_timeout=240,
    format="json",
)

program = LLMTextCompletionProgram.from_defaults(
    output_cls=TableInfo,
    prompt_template_str=PROMPT_STR,
    llm=Settings.llm,
)


def extract_first_json_block(text: str):
    match = re.search(r"\{.*\}", text, re.S)  # grab first {...} block
    if not match:
        raise ValueError("No JSON object found in output")
    return pyjson.loads(match.group())


def _get_tableinfo_with_index(idx: int) -> str:
    results_gen = Path(TABLEINFO_DIR).glob(f"{idx}_*")
    results_list = list(results_gen)
    
    if len(results_list) == 0:
        return None
    elif len(results_list) == 1:
        path = results_list[0]
        json_str = path.read_text(encoding="utf-8")
        return TableInfo.model_validate_json(json_str)
    else:
        raise ValueError(f"More than one file matching index: {list(results_gen)}")


table_names = set()
table_infos = []

for idx, df in enumerate(dfs):
    table_info = _get_tableinfo_with_index(idx)
    if table_info:
        table_infos.append(table_info)
        continue

    df_str = df.head(10).to_csv()

    for attempt in range(MAX_RETRIES):
        try:
            raw_output = program(
                table_str=df_str,
                exclude_table_name_list=str(list(table_names)),
            )

            if isinstance(raw_output, TableInfo):
                table_info = raw_output
            elif isinstance(raw_output, dict):
                table_info = TableInfo(**raw_output)
            elif isinstance(raw_output, str):
                parsed_dict = extract_first_json_block(raw_output)
                table_info = TableInfo(**parsed_dict)
            else:
                raise TypeError(f"Unexpected return type from program(): {type(raw_output)}")

            table_name = table_info.table_name
            print(f"Processed table: {table_name}")

            if table_name in table_names:
                print(f"Table name '{table_name}' already exists, skipping this table.")
                table_info = None  # don’t append duplicate
                break  # skip

            # save table info
            table_names.add(table_name)
            out_file = f"{TABLEINFO_DIR}/{idx}_{table_name}.json"
            json.dump(table_info.model_dump(), open(out_file, "w"))
            break  # move to next table

        except Exception as e:
            print(f"Error with attempt {attempt+1}: {e}")
            time.sleep(2)

    if table_info:
        table_infos.append(table_info)


# Function to create a sanitized column name
def sanitize_column_name(col_name):
    # Remove special characters and replace spaces with underscores
    return re.sub(r"\W+", "_", col_name)


# Function to create a table from a DataFrame using SQLAlchemy
def create_table_from_dataframe(
    df: pd.DataFrame, table_name: str, engine, metadata_obj
):
    # Sanitize column names
    sanitized_columns = {col: sanitize_column_name(col) for col in df.columns}
    df = df.rename(columns=sanitized_columns)

    # Dynamically create columns based on DataFrame columns and data types
    columns = [
        Column(col, String if dtype == "object" else Integer)
        for col, dtype in zip(df.columns, df.dtypes)
    ]

    # Create a table with the defined columns
    table = Table(table_name, metadata_obj, *columns)

    # Create the table in the database
    metadata_obj.create_all(engine)

    # Insert data from DataFrame into the table
    with engine.connect() as conn:
        for _, row in df.iterrows():
            insert_stmt = table.insert().values(**row.to_dict())
            conn.execute(insert_stmt)
        conn.commit()


# engine = create_engine("sqlite:///:memory:")
engine = create_engine("sqlite:///../sqlite/SQLite_db.db")
metadata_obj = MetaData()
for idx, df in enumerate(dfs):
    tableinfo = _get_tableinfo_with_index(idx)
    if tableinfo is None:
        print(f"[ERROR] No TableInfo for index {idx}")
        continue  # skip this one or handle it differently
    print(f"Creating table: {tableinfo.table_name}")
    create_table_from_dataframe(df, tableinfo.table_name, engine, metadata_obj)


# Object index, retriever, SQLDatabase
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

sql_database = SQLDatabase(engine)
table_node_mapping = SQLTableNodeMapping(sql_database)

table_schema_objs = [
    SQLTableSchema(table_name=t.table_name, context_str=t.table_summary)
    for t in table_infos
]  # add a SQLTableSchema for each table

obj_index = ObjectIndex.from_objects(
    table_schema_objs,
    table_node_mapping,
    VectorStoreIndex,
    embed_model=Settings.embed_model,
)
obj_retriever = obj_index.as_retriever(similarity_top_k=5)



# SQLRetriever + Table Parser
sql_retriever = SQLRetriever(sql_database)


def get_table_context_str(table_schema_objs: List[SQLTableSchema]):
    """Get table context string."""
    context_strs = []
    for table_schema_obj in table_schema_objs:
        table_info = sql_database.get_single_table_info(
            table_schema_obj.table_name
        )
        if table_schema_obj.context_str:
            table_opt_context = " The table description is: "
            table_opt_context += table_schema_obj.context_str
            table_info += table_opt_context

        context_strs.append(table_info)
    return "\n\n".join(context_strs)


table_parser_component = get_table_context_str(table_schema_objs)



# Text-to-SQL Prompt + Output Parser
def parse_response_to_sql(response: ChatResponse) -> str:
    """Parse response to SQL."""
    response = response.message.content
    sql_query_start = response.find("SQLQuery:")
    if sql_query_start != -1:
        response = response[sql_query_start:]
        
        if response.startswith("SQLQuery:"):
            response = response[len("SQLQuery:") :]
    sql_result_start = response.find("SQLResult:")
    if sql_result_start != -1:
        response = response[:sql_result_start]
    return response.strip().strip("```").strip()


sql_parser_component = FunctionTool.from_defaults(fn=parse_response_to_sql)

text2sql_prompt = DEFAULT_TEXT_TO_SQL_PROMPT.partial_format(
    dialect=engine.dialect.name
)
print(text2sql_prompt.template)



# Response Synthesis Prompt
response_synthesis_prompt_str = (
    "Given an input question, synthesize a response from the query results.\n"
    "Query: {query_str}\n"
    "SQL: {sql_query}\n"
    "SQL Response: {context_str}\n"
    "Response: "
)
response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)

# Step 1: Configure Settings
callback_manager = CallbackManager()
Settings.callback_manager = callback_manager

def index_all_tables(sql_database, table_index_dir: str = "../data/table_index_dir") -> Dict[str, VectorStoreIndex]:
    """Index all tables in the SQL database."""
    Path(table_index_dir).mkdir(parents=True, exist_ok=True)

    vector_index_dict = {}
    engine = sql_database.engine

    for table_name in sql_database.get_usable_table_names():
        print(f"Indexing rows in table: {table_name}")
        table_path = Path(table_index_dir) / table_name

        if not table_path.exists():
            # Fetch all rows from the table
            with engine.connect() as conn:
                result = conn.execute(text(f'SELECT * FROM "{table_name}"'))
                row_tuples = [tuple(row) for row in result.fetchall()]

            # Create TextNode objects from rows
            nodes = [TextNode(text=str(row)) for row in row_tuples]

            # Build the index using current global Settings
            index = VectorStoreIndex(nodes)

            # Save index
            index.set_index_id("vector_index")
            index.storage_context.persist(persist_dir=str(table_path))

        else:
            # Rebuild storage context from saved directory
            storage_context = StorageContext.from_defaults(
                persist_dir=str(table_path)
            )

            # Load existing index
            index = load_index_from_storage(
                storage_context, index_id="vector_index"
            )

        vector_index_dict[table_name] = index

    return vector_index_dict

vector_index_dict = index_all_tables(sql_database)

def get_table_context_and_rows_str(query_str: str, table_schema_objs: List[SQLTableSchema]):
    """Get table context string."""
    context_strs = []
    for table_schema_obj in table_schema_objs:
        # first append table info + additional context
        table_info = sql_database.get_single_table_info(
            table_schema_obj.table_name
        )
        if table_schema_obj.context_str:
            table_opt_context = " The table description is: "
            table_opt_context += table_schema_obj.context_str
            table_info += table_opt_context

        # also lookup vector index to return relevant table rows
        vector_retriever = vector_index_dict[
            table_schema_obj.table_name
        ].as_retriever(similarity_top_k=2)
        relevant_nodes = vector_retriever.retrieve(query_str)
        if len(relevant_nodes) > 0:
            table_row_context = "\nHere are some relevant example rows (values in the same order as columns above)\n"
            for node in relevant_nodes:
                table_row_context += str(node.get_content()) + "\n"
            table_info += table_row_context

        context_strs.append(table_info)
    return "\n\n".join(context_strs)


# custom events
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

class SQLResultsEvent(Event):
    context_str: str
    sql_query: str
    query_str: str
    success: bool = True

class ResponsePromptReadyEvent(Event):
    rs_prompt: str


# helpers
def _is_valid_sql_start(text: str) -> bool:
    """Check if text starts with valid SQL"""
    if not text:
        return False
    
    sql_keywords = ['SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE']
    text_upper = text.upper().strip()
    return any(text_upper.startswith(keyword) for keyword in sql_keywords)

def _clean_sql_query(sql: str) -> str:
    """
    Clean and standardize SQL query.
    """
    if not sql:
        return "SELECT 1"
    
    # Remove extra whitespace
    sql = ' '.join(sql.split())
    
    # Fix quote issues - convert double quotes to single quotes for string literals
    # This is a simple approach - for more complex cases, you'd need a proper SQL parser
    sql = re.sub(r'"([^"]*)"', r"'\1'", sql)
    
    # Remove multiple semicolons
    sql = re.sub(r';+', ';', sql)
    
    # Remove trailing semicolon and add it back cleanly
    sql = sql.rstrip(';').strip()
    
    # Don't add semicolon for now since it might be causing issues
    return sql


# custom fallbacks
def extract_sql_from_response(llm_response: str) -> str:
    """
    Extract SQL query from LLM response that might contain reasoning or formatting.
    """
    response = llm_response.strip()
    
    # First, remove <think> blocks entirely
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    
    # Remove any non-SQL content at the beginning
    response = re.sub(r'^[^S]*(?=SELECT|WITH|INSERT|UPDATE|DELETE)', '', response, flags=re.IGNORECASE)
    
    # Method 1: Look for SQLQuery: pattern
    sql_query_match = re.search(r'SQLQuery:\s*([^;]+;?)', response, re.IGNORECASE | re.DOTALL)
    if sql_query_match:
        sql = sql_query_match.group(1).strip()
        return _clean_sql_query(sql)
    
    # Method 2: Look for SQL in code blocks
    code_block_patterns = [
        r'```sql\s*\n(.*?)\n```',
        r'```\s*\n(.*?)\n```',
        r'`([^`]+)`'
    ]
    
    for pattern in code_block_patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            sql = match.group(1).strip()
            if _is_valid_sql_start(sql):
                return _clean_sql_query(sql)
    
    # Method 3: Look for standalone SQL statements
    sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH']
    
    # Split by lines and look for SQL statements
    lines = response.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if line starts with SQL keyword
        if any(line.upper().startswith(keyword.upper()) for keyword in sql_keywords):
            return _clean_sql_query(line)
    
    # Method 4: Look for multi-line SQL statements
    for keyword in sql_keywords:
        pattern = rf'\b{keyword}\b.*?(?=\n\s*\n|\nSQLResult|\nAnswer|$)'
        sql_match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if sql_match:
            sql = sql_match.group(0).strip()
            return _clean_sql_query(sql)
    
    # Fallback: if nothing found, return empty string to avoid errors
    print(f"Warning: Could not extract SQL from response: {response[:100]}...")
    return "SELECT [-]"  # Safe fallback query

def analyze_sql_error(error_message: str, sql_query: str, table_schema: str) -> str:
    """
    Analyze SQL error and provide suggestions for fixing the query.
    """
    error_lower = error_message.lower()
    
    if "no such column" in error_lower:
        # Extract the problematic column name
        column_match = re.search(r'no such column:\s*(\w+)', error_lower)
        if column_match:
            bad_column = column_match.group(1)
            
            # Try to suggest correct column names from schema
            schema_lower = table_schema.lower()
            possible_columns = re.findall(r'(\w+):', schema_lower)
            
            suggestions = []
            for col in possible_columns:
                if bad_column.lower() in col.lower() or col.lower() in bad_column.lower():
                    suggestions.append(col)
            
            error_msg = f"Column '{bad_column}' does not exist."
            if suggestions:
                error_msg += f" Did you mean: {', '.join(suggestions[:3])}?"
            error_msg += f"\n\nAvailable columns from schema:\n{table_schema}"
            return error_msg
    
    elif "no such table" in error_lower:
        table_match = re.search(r'no such table:\s*([\w\s\[\]]+)', error_lower)
        if table_match:
            bad_table = table_match.group(1).strip()
            return f"Table '{bad_table}' does not exist. Available tables from schema:\n{table_schema}"
    
    elif "syntax error" in error_lower:
        return f"SQL syntax error. Please check:\n- Missing quotes around strings\n- Proper parentheses\n- Correct SQL keywords\n\nFailed query: {sql_query}"
    
    return f"SQL execution error: {error_message}\n\nFailed query: {sql_query}\n\nSchema: {table_schema}"

def create_enhanced_prompt(table_schema: str, query_str: str, retry_count: int = 0, error_message: str = ""):
    if retry_count == 0:
        # Initial attempt
        ENHANCED_PROMPT = f"""Given the table schema and user question below, generate ONLY a valid SQL query.

            Table Schema:
            {table_schema}

            User Question: {query_str}

            IMPORTANT RULES:
            1. Return ONLY the SQL query, nothing else
            2. Use single quotes for string literals, not double quotes
            3. Do not include any explanations, reasoning, or additional text
            4. Do not include labels like "SQLQuery:", "Answer:", etc.
            5. Do not wrap in code blocks or markdown formatting
            6. Do not include semicolons at the end
            7. Do not include any <think> tags or reasoning
            8. Only use column names that exist in the provided schema

            Example format:
            SELECT column_name FROM table_name WHERE condition

            Your SQL query:
        """
    else:
        # Retry attempt with error information
        ENHANCED_PROMPT = f"""The previous SQL query failed with an error. Please generate a corrected SQL query.

            Table Schema:
            {table_schema}

            User Question: {query_str}

            Previous Error: {error_message}

            IMPORTANT RULES:
            1. Return ONLY the corrected SQL query, nothing else
            2. Use single quotes for string literals, not double quotes
            3. Carefully check that all column names exist in the provided schema
            4. Do not include any explanations, reasoning, or additional text
            5. Do not include labels like "SQLQuery:", "Answer:", etc.
            6. Do not wrap in code blocks or markdown formatting
            7. Do not include semicolons at the end
            8. Only use column names that are explicitly listed in the schema above

            Your corrected SQL query:
        """
    
    return ENHANCED_PROMPT


class Text2SQLWorkflowRowRetrieval(Workflow):
    @step
    async def input_step(self, ev: StartEvent) -> TableRetrievedEvent:
        """Step 1: Process initial query and retrieve relevant tables"""
        query = ev.query
        tables = obj_retriever.retrieve(query)  # retrieve candidate schemas
        
        return TableRetrievedEvent(
            tables=tables, 
            query_str=query
        )

    @step
    async def table_output_parser_step(self, ev: TableRetrievedEvent) -> SchemaProcessedEvent:
        """Step 2: Parse schemas + retrieve relevant rows"""
        schema_str = get_table_context_and_rows_str(ev.query_str, ev.tables)
        
        return SchemaProcessedEvent(
            table_schema=schema_str, 
            query_str=ev.query_str
        )

    @step
    async def text2sql_prompt_step(self, ev: SchemaProcessedEvent | SQLResultsEvent) -> SQLPromptReadyEvent:
        """Step 3: Create prompt (initial or retry)"""
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

        prompt = create_enhanced_prompt(table_schema, query_str, retry_count, error_message)
        
        return SQLPromptReadyEvent(
            t2s_prompt=prompt,
            query_str=query_str,
            table_schema=table_schema,
            retry_count=retry_count,
            error_message=error_message
        )

    @step
    async def text2sql_llm_step(self, ev: SQLPromptReadyEvent) -> SQLGeneratedEvent:
        """Step 4: Run LLM to generate SQL"""
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
        """Step 5: Parse/clean SQL"""
        try:
            clean_sql = parse_response_to_sql(ev)  # primary parser
        except Exception:
            clean_sql = extract_sql_from_response(ev.sql_query)  # fallback

        if not clean_sql:
            clean_sql = extract_sql_from_response(ev.sql_query)

        print(f"Attempt #{ev.retry_count + 1}")
        print(f"LLM Response: {ev.sql_query}")
        print(f"Cleaned SQL: {clean_sql}")

        return SQLParsedEvent(
            sql_query=clean_sql,
            query_str=ev.query_str,
            table_schema=ev.table_schema,
            retry_count=ev.retry_count,
            error_message=ev.error_message
        )

    @step
    async def sql_retriever_step(self, ev: SQLParsedEvent) -> SQLResultsEvent:
        """Step 6: Execute SQL with retries"""
        try:
            results = sql_retriever.retrieve(ev.sql_query)
            print(f"[SUCCESS] Executed on attempt #{ev.retry_count + 1}")
            
            return SQLResultsEvent(
                context_str=str(results),
                sql_query=ev.sql_query,
                query_str=ev.query_str,
                success=True
            )
        except Exception as e:
            error_msg = str(e)
            print(f"[ERROR] Execution failed (Attempt #{ev.retry_count + 1}): {error_msg}")

            if ev.retry_count < MAX_RETRIES:
                retry_event = SQLResultsEvent(
                    context_str="",
                    sql_query=ev.sql_query,
                    query_str=ev.query_str,
                    success=False
                )
                retry_event.retry_count = ev.retry_count + 1
                retry_event.error_message = analyze_sql_error(error_msg, ev.sql_query, ev.table_schema)
                retry_event.table_schema = ev.table_schema
                
                return retry_event
            else:
                return SQLResultsEvent(
                    context_str=f"Failed after {MAX_RETRIES+1} attempts. Final error: {error_msg}",
                    sql_query=ev.sql_query,
                    query_str=ev.query_str,
                    success=False
                )

    @step
    async def retry_handler_step(self, ev: SQLResultsEvent) -> SQLPromptReadyEvent:
        """Step 7: Retry failed SQL by regenerating prompt"""
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
        """Step 8: Prepare final synthesis prompt"""
        if not ev.success:
            return None
        prompt = response_synthesis_prompt.format(
            query_str=ev.query_str,
            context_str=ev.context_str,
            sql_query=ev.sql_query
        )
        
        return ResponsePromptReadyEvent(rs_prompt=prompt)

    @step
    async def response_synthesis_llm_step(self, ev: ResponsePromptReadyEvent) -> StopEvent:
        """Step 9: Generate final human-readable answer"""
        answer = await Settings.llm.acomplete(ev.rs_prompt)
        
        return StopEvent(result=str(answer))


# Runner
async def run_text2sql_workflow_row(query: str):
    workflow = Text2SQLWorkflowRowRetrieval(timeout=240)
    result = await workflow.run(query=query)
    return result


async def visualize_text2sql_workflow(sample_query: str, execution_name: str):
    """
    Function to visualize the Text2SQL workflow both as all possible flows
    and a specific execution example
    """
    output_dir = ("../outputs/trials_v2/execution")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Draw ALL possible flows through your workflow
    print("Drawing all possible flows...")
    all_flows_path = os.path.join(output_dir, f"{execution_name}_text2sql_workflow_flow.html")
    draw_all_possible_flows(
        Text2SQLWorkflowRowRetrieval, 
        filename=all_flows_path
    )
    print(f"[SUCCESS] All possible flows saved to: {all_flows_path}")

    # 2. Run workflow + visualize the execution path
    print("Running workflow and drawing execution path...")
    try:
        workflow = Text2SQLWorkflowRowRetrieval(timeout=240)
        result = await workflow.run(query=sample_query)

        # Draw the execution path
        execution_path = os.path.join(output_dir, f"{execution_name}_text2sql_workflow_execution.html")
        draw_most_recent_execution(
            workflow,
            filename=execution_path
        )
        print(f"[SUCCESS] Recent execution path saved to: {execution_path}")
        print(f"Workflow result: {result}")
        
    except Exception as e:
        print(f"[ERROR] Error during workflow execution: {e}")
        print("Note: You may need to set up your retriever and LLM settings first")

# Execution
await visualize_text2sql_workflow("Who won best director in the 1972 academy awards?", "best_dir_1972")"