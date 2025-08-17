You're a Senior ML Engineer who's an expert in their field, like God Tier Dev. & I'm their intern, now help me out like you'd do it.


## QUERY PIPELINE:
```
qp.add_modules({
    "input": InputComponent(),
    "table_retriever": obj_retriever,
    "table_output_parser": table_parser_component,
    "text2sql_prompt": text2sql_prompt,
    "text2sql_llm": Settings.llm,
    "sql_output_parser": sql_parser_component,
    "sql_retriever": sql_retriever,
    "response_synthesis_prompt": response_synthesis_prompt,
    "response_synthesis_llm": Settings.llm,
})
```

## QP CONNECTIONS:
```
qp.add_link("input", "table_retriever")
qp.add_link("input", "table_output_parser", dest_key="query_str")
qp.add_link(
    "table_retriever", "table_output_parser", dest_key="table_schema_objs"
)
qp.add_link("input", "text2sql_prompt", dest_key="query_str")
qp.add_link("table_output_parser", "text2sql_prompt", dest_key="schema")
qp.add_chain(
    ["text2sql_prompt", "text2sql_llm", "sql_output_parser", "sql_retriever"]
)
qp.add_link(
    "sql_output_parser", "response_synthesis_prompt", dest_key="sql_query"
)
qp.add_link(
    "sql_retriever", "response_synthesis_prompt", dest_key="context_str"
)
qp.add_link("input", "response_synthesis_prompt", dest_key="query_str")
qp.add_link("response_synthesis_prompt", "response_synthesis_llm")
```


## WORKFLOW REFERENCE EXAMPLE:
```
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
    return "SELECT 1"  # Safe fallback query

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


class Text2SQLWorkflow(Workflow):
    @step
    async def input_step(self, ev: StartEvent) -> TableRetrievedEvent:
        """Process the initial query and retrieve relevant tables"""
        query = ev.query
        # Retrieve table schemas
        table_schema_obj = obj_retriever.retrieve(query)
        
        return TableRetrievedEvent(
            tables=table_schema_obj,
            query_str=query
        )
    
    @step
    async def table_output_parser_step(self, ev: TableRetrievedEvent) -> SchemaProcessedEvent:
        """Parse table schemas into string format"""
        schema_str = get_table_context_str(ev.tables)
        
        return SchemaProcessedEvent(
            table_schema=schema_str,
            query_str=ev.query_str
        )
    
    @step
    async def text2sql_prompt_step(self, ev: SchemaProcessedEvent | SQLResultsEvent) -> SQLPromptReadyEvent:
        """Create the text-to-SQL prompt with optional error correction"""
        # Handle both initial attempt and retry attempts
        if isinstance(ev, SchemaProcessedEvent):
            table_schema = ev.table_schema
            query_str = ev.query_str
            retry_count = 0
            error_message = ""
        else:  # SQLResultsEvent (retry case)
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
        """Generate SQL query using LLM"""
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
        """Parse and clean the generated SQL query"""
        # Extract clean SQL from the LLM response
        try:
            clean_sql = parse_response_to_sql(ev)  # Built-in parser
        except Exception:
            clean_sql = extract_sql_from_response(ev.sql_query)  # Fallback
        
        if not clean_sql:
            clean_sql = extract_sql_from_response(ev.sql_query)
        
        print(f"Attempt #{ev.retry_count + 1}")
        print(f"Original LLM Response: {ev.sql_query}")
        print(f"Cleaned SQL Query: {clean_sql}")
        
        return SQLParsedEvent(
            sql_query=clean_sql,
            query_str=ev.query_str,
            table_schema=ev.table_schema,
            retry_count=ev.retry_count,
            error_message=ev.error_message
        )
    
    @step
    async def sql_retriever_step(self, ev: SQLParsedEvent) -> SQLResultsEvent:
        """Execute SQL query and get results with retry logic"""
        
        try:
            results = sql_retriever.retrieve(ev.sql_query)
            print(f"[SUCCESS] SQL executed successfully on attempt #{ev.retry_count + 1}")
            return SQLResultsEvent(
                context_str=str(results),
                sql_query=ev.sql_query,
                query_str=ev.query_str,
                success=True
            )
        except Exception as e:
            error_msg = str(e)
            print(f"[ERROR] SQL Execution Error (Attempt #{ev.retry_count + 1}): {error_msg}")
            print(f"Failed SQL Query: {ev.sql_query}")
            
            # Check if we should retry
            if ev.retry_count < MAX_RETRIES:
                print(f"[RETRY] Retrying... (Attempt #{ev.retry_count + 2}/{MAX_RETRIES + 1})")
                
                # Return an SQLResultsEvent that will trigger a retry
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
                print(f"[ERROR, RETRY FAILED] Max retries ({MAX_RETRIES}) reached. Giving up.")
                return SQLResultsEvent(
                    context_str=f"Failed to execute SQL after {MAX_RETRIES + 1} attempts. Final error: {error_msg}",
                    sql_query=ev.sql_query,
                    query_str=ev.query_str,
                    success=False
                )
    
    @step
    async def retry_handler_step(self, ev: SQLResultsEvent) -> SQLPromptReadyEvent:
        """Handle retry logic - only triggered when SQL execution fails"""
        # This step only processes failed SQL results that need retrying
        if ev.success or not hasattr(ev, 'retry_count'):
            return None  # Let successful results pass through to response synthesis
        
        print(f"[RETRY] Preparing retry #{ev.retry_count + 1}")
        
        # Create a new prompt event for retry
        return SQLPromptReadyEvent(
            t2s_prompt="",  # Will be filled in text2sql_prompt_step
            query_str=ev.query_str,
            table_schema=getattr(ev, 'table_schema', ''),
            retry_count=ev.retry_count,
            error_message=getattr(ev, 'error_message', 'Unknown error')
        )
    
    @step
    async def response_synthesis_prompt_step(self, ev: SQLResultsEvent) -> ResponsePromptReadyEvent:
        """Create the response synthesis prompt - only for successful SQL results"""
        # Only process successful SQL results
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
        """Generate final answer using LLM"""
        answer = await Settings.llm.acomplete(ev.rs_prompt)
        
        return StopEvent(result=str(answer))


async def run_text2sql_workflow(query: str):
    workflow = Text2SQLWorkflow(timeout=240)
    result = await workflow.run(query=query)
    return result
```

WITH THIS NEW TABLE PARSER HAVING ROW RETRIEVALS -> I WANT TO WRITE THE NEW WORKFLOW
USING THE WORKFLOW REFERENCE EXAMPLE FOR ONLY TABLE RETRIEVALS. NOW, I WANT TO REPLICATE THE DEPRECATED QUERY PIPELINE FLOW WITH TABLE & ROW RETRIEVALS, USING WORKFLOWS AS PROVIDED IN THE REFERENCE EXAMPLE ABOVE.
THE NEW WORKFLOW will be inside of a class named 'class Text2SQLWorkflowRowRetrieval(Workflow):', where there will be multiple steps (from StartEvent, to StopEvent)

FOLLOW ALL THE PRACTICAL CASES FOR THE SAME, & re-write the whole thing in workflows now (also explain everything to me step by step)