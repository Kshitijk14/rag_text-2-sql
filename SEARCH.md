## METHODS:
1. obj_retriever = obj_index.as_retriever(similarity_top_k=5)
2. table_parser_component = get_table_context_str(table_schema_objs)
3. text2sql_prompt = DEFAULT_TEXT_TO_SQL_PROMPT.partial_format(dialect=engine.dialect.name)
4. Settings.llm = Ollama(model="qwen3:0.6b", request_timeout=240,format="json")
5. sql_parser_component = FunctionTool.from_defaults(fn=parse_response_to_sql)
6. sql_retriever = SQLRetriever(sql_database)
7. response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)
8. Settings.llm = Ollama(model="qwen3:0.6b", request_timeout=240,format="json")


## QUERY PIPELINE:
```
qp = QP(
    modules={
        "input": InputComponent(),
        "table_retriever": obj_retriever,
        "table_output_parser": table_parser_component,
        "text2sql_prompt": text2sql_prompt,
        "text2sql_llm": llm,
        "sql_output_parser": sql_parser_component,
        "sql_retriever": sql_retriever,
        "response_synthesis_prompt": response_synthesis_prompt,
        "response_synthesis_llm": llm,
    },
    verbose=True,
)
```

## QP CONNECTIONS:
```
qp.add_chain(["input", "table_retriever", "table_output_parser"])
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


## WORKFLOW:




# PROMPT:
"You're a Senior ML Engineer who's an expert in their field, like God Tier Dev. & I'm their intern, now help me out like you'd do it.

COMMON_METHODS:
"1. obj_retriever = obj_index.as_retriever(similarity_top_k=5)
2. table_parser_component = get_table_context_str(table_schema_objs)
3. text2sql_prompt = DEFAULT_TEXT_TO_SQL_PROMPT.partial_format(dialect=engine.dialect.name)
4. Settings.llm = Ollama(model="qwen3:0.6b", request_timeout=240,format="json")
5. sql_parser_component = FunctionTool.from_defaults(fn=parse_response_to_sql)
6. sql_retriever = SQLRetriever(sql_database)
7. response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)
8. Settings.llm = Ollama(model="qwen3:0.6b", request_timeout=240,format="json")"

OLD_LOGIC:
"QUERY PIPELINE:
```
qp = QP(
    modules={
        "input": InputComponent(),
        "table_retriever": obj_retriever,
        "table_output_parser": table_parser_component,
        "text2sql_prompt": text2sql_prompt,
        "text2sql_llm": llm,
        "sql_output_parser": sql_parser_component,
        "sql_retriever": sql_retriever,
        "response_synthesis_prompt": response_synthesis_prompt,
        "response_synthesis_llm": llm,
    },
    verbose=True,
)
```

CONNECTIONS:
```
qp.add_chain(["input", "table_retriever", "table_output_parser"])
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
```"

USING THESE COMMON METHODS, I WANT TO REPLICATE THE QUERY PIPELINE FLOW, USING WORKFLOWS NOW, AS QP IS BEING DEPRECATED (i.e. it doesn't work optimally now)

THE WORKFLOW will be inside of a class named 'class Text2SQLWorkflow(Workflow):', where there will be multiple steps (from StartEvent, to StopEvent)

FOLLOW ALL THE PRACTICAL CASES FOR THE SAME, & re-write the whole thing in workflows now (also explain everything to me step by step)"