from ..helpers.other_imports import re


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