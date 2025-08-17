from ..helpers.other_imports import re
from .helpers import (
    _is_valid_sql_start,
    _clean_sql_query,
)


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

def create_t2s_prompt(table_schema: str, query_str: str, retry_count: int = 0, error_message: str = ""):
    if retry_count == 0:
        # Initial attempt
        TEXT_2_SQL_PROMPT = (f"""
            Given the table schema and user question below, generate ONLY a valid SQL query.

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
        """)
    else:
        # Retry attempt with error information
        TEXT_2_SQL_PROMPT = (f"""
            The previous SQL query failed with an error. Please generate a corrected SQL query.

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
        """)
    
    return TEXT_2_SQL_PROMPT