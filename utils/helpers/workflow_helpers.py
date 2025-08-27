import re
import json

from typing import Any, Union

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


def _repair_json_string(text: str) -> str:
    """Apply common repairs to malformed JSON from LLMs."""
    # Remove trailing commas before } or ]
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    # Replace single quotes with double quotes (naive but works for most LLM JSON)
    text = re.sub(r"(?<!\\)'", '"', text)
    # Ensure proper escaping of backslashes
    text = text.replace("\\", "\\\\")
    return text


def parse_llm_json(raw_output: Union[str, Any], logger, mode: str) -> Union[str, dict]:
    """
    Parse LLM output expected to be JSON.

    Args:
        raw_output (str|Any): The raw output from the LLM.
        logger: Logger for errors/debugging.
        mode (str): 
            - "answer" → return only the `answer` field (default).
            - "full"   → return the full parsed JSON dict.
            - "raw"    → return the raw string output.

    Returns:
        str | dict: Parsed result based on mode, or fallback string if parsing fails.
    """
    text = str(raw_output).strip()

    # 01: strip markdown fences if model wrapped output in ```json ... ```
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)  # remove opening ```json or ```
        text = re.sub(r"\n?```$", "", text)  # remove closing ```

    # 02: try direct JSON parse
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("[WARN] JSON parsing failed, attempting auto-repair...")
        repaired = _repair_json_string(text)
        
        try:
            parsed = json.loads(repaired)
        except Exception as e:
            logger.error(f"[ERROR] Failed to parse JSON even after repair: {e}")
            logger.debug(f"Raw LLM output: {raw_output}")
            return text  # fallback: raw string
        
    except Exception as e:
        logger.error(f"[ERROR] Unexpected JSON parsing issue: {e}")
        logger.debug(f"Raw LLM output: {raw_output}")
        return text  # fallback: raw string

    if mode == "answer":
        return parsed.get("answer", "").strip()
    elif mode == "full":
        return parsed
    else:  # raw
        return text
