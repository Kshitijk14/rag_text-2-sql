import re
from llama_index.core.llms import ChatResponse


def _is_valid_sql_start(text: str) -> bool:
    """Check if text starts with valid SQL"""
    if not text:
        return False
    
    sql_keywords = ['SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE', 'CASE']
    text_upper = text.upper().strip()
    return any(text_upper.startswith(keyword) for keyword in sql_keywords)

def _clean_sql_query(sql: str) -> str:
    """
    Clean and standardize SQL query.
    """
    if not sql:
        return "SELECT none"
    
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


def parse_response_to_sql(response: ChatResponse) -> str:
    """Parse response into a clean SQL string."""
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