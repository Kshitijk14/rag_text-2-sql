TABLE_INFO_PROMPT = ("""\
    You are an expert data engineer. Return only a JSON object, with no explanation, no prose, no markdown, and no trailing text. Write a 2-3 sentence summary of the given SQL table, describing its purpose and key relationships.
    You are to produce **ONLY** a JSON object matching the following exact schema:

    {
        "table_name": "<short_name_in_snake_case_without_spaces>",
        "table_summary": "<short concise caption of the table>"
    }

    Example:
    {
        "table_name": "Album", 
        "table_summary": "Summary of album and artist data. Table contains albums with titles and artist IDs, with AlbumId as the primary key and ArtistId referencing an artist table."
    }

    RULES:
    - The table_name must be unique to the table, describe it clearly, and be in snake_case.
    - Do NOT output a generic table name (e.g., "table", "my_table").
    - Do NOT make the table name one of the following: {exclude_table_name_list}.
    - Do NOT include any keys other than "table_name" and "table_summary".
    - Do NOT include extra text before/after the JSON.
    - Do NOT include any other keys or text before/after the JSON.
    - Do NOT wrap in ```json.

    Table:
    {table_str}
""")

TEXT_2_SQL_PROMPT = ("""
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

RETRY_TEXT_2_SQL_PROMPT = ("""
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

# RESPONSE_SYNTHESIS_PROMPT = (
#     "Given an input question, synthesize a response from the query results.\n"
#     "Query: {query_str}\n"
#     "SQL: {sql_query}\n"
#     "SQL Response: {context_str}\n"
#     "Response: "
# )

RESPONSE_SYNTHESIS_PROMPT = ("""
    Given an input question and the SQL query results, synthesize a response from the query results. 
    Return only a JSON object, with no explanation, no prose, no markdown, and no trailing text. 
    Finally, you are to produce **ONLY** a JSON object returning the final answer, matching the following exact schema:

    {{
        "query": "<the input question>",
        "sql": "<the executed SQL query>",
        "sql_response": "<the SQL query results>",
        "answer": "<concise natural language answer>"
    }}
    
    RULES:
    - Do NOT include extra text before/after the JSON.
    - Do NOT include any other keys or text before/after the JSON.
    - Do NOT wrap in ```json.
    
    Query: {query_str}
    SQL: {sql_query}
    SQL Response: {context_str}
    Final Response:\n 
""")