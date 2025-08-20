INTRO:
"You're a Senior NLP Eng., who's an expert in their field and on stuff from like a last decade. I'm your intern, & you've to help me out, and expect only production grade results. Also if necessary, let's write tests for each step, just to be sure."

CODE SNIPPET:
```
def index_all_tables_with_chroma(sql_database, chroma_db_dir: str = CHROMA_DB_DIR) -> Dict[str, VectorStoreIndex]:
    """Index all tables in the SQL database using ChromaDB as the backend."""
    os.makedirs(chroma_db_dir, exist_ok=True)

    vector_index_dict = {}
    engine = sql_database.engine

    logger.info(f" [00] Creating persistent Chroma client at: {chroma_db_dir}")
    chroma_client = chromadb.PersistentClient(path=chroma_db_dir)

    for table_name in sql_database.get_usable_table_names():
        logger.info(f" [01] Indexing rows in table: {table_name}")

        # Each table = separate Chroma collection
        collection = chroma_client.get_or_create_collection(name=f"table_{table_name}")
        vector_store = ChromaVectorStore(chroma_collection=collection)
        persist_dir = os.path.join(chroma_db_dir, f"table_{table_name}")

        logger.info(f" [02] Fetching all rows from table: {table_name}")
        
        if collection.count() == 0:
            logger.info(f"  [02.1.1] No existing index found → building new index for: {table_name}")
            
            with engine.connect() as conn:
                result = conn.execute(text(f'SELECT * FROM "{table_name}"'))
                
                # row_tuples = [tuple(row) for row in result.fetchall()]
                
                logger.info(f" - Converting rows to structured strings with col=value format")
                col_names = result.keys()
                row_texts = [
                    " | ".join([f"{col}={val}" for col, val in zip(col_names, row)])
                    for row in result.fetchall()
                ]

            logger.info(f"  [02.1.2] Converting rows to text nodes for table: {table_name}")
            # nodes = [TextNode(text=str(row)) for row in row_tuples]
            nodes = [TextNode(text=row_text) for row_text in row_texts]

            logger.info(f"  [02.1.3] Building vector index for table: {table_name}")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex(nodes, storage_context=storage_context)

            logger.info(f"  [02.1.4] Persisting index to: {persist_dir}")
            storage_context.persist(persist_dir=persist_dir)

        else:
            logger.info(f"  [02.2] Reloading existing Chroma index for table: {table_name}")
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context)
            
        vector_index_dict[table_name] = index

    return vector_index_dict

# Build vector indexes for all tables using ChromaDB
vector_index_dict = index_all_tables_with_chroma(sql_database)

def get_table_context_and_rows_str(query_str: str, table_schema_objs: List[TableInfo]):
    """Get table context string for your TableInfo objects."""
    context_strs = []

    for table_info_obj in table_schema_objs:
        logger.info("[01] Getting schema for table (use .table_name instead of .name)")
        table_info = sql_database.get_single_table_info(table_info_obj.table_name)

        logger.info("[02] Retrieving example rows for table")
        vector_retriever = vector_index_dict[table_info_obj.table_name].as_retriever(similarity_top_k=TOP_N)
        relevant_nodes = vector_retriever.retrieve(query_str) # this will return the TextNodes we stored as vector indexes 
        logger.info(f"Retrieved {len(relevant_nodes)} relevant nodes for table: {table_info_obj.table_name}")

        if len(relevant_nodes) > 0:
            table_row_context = ("\nHere are some relevant example rows (column=value):\n")
            for node in relevant_nodes:
                table_row_context += str(node.get_content()) + "\n"
            table_info += table_row_context

        context_strs.append(table_info)
        
        logger.error(f"No vector index found for {table_info_obj.table_name}")
        continue

    return "\n\n".join(context_strs)

table_parser_component = get_table_context_and_rows_str(QUERY_1, table_schema_objs)
logger.info(f"Updated table context with rows:\n{table_parser_component}")
```

OUTPUT:
```
2025-08-20 13:48:00,876 [INFO] [01] Getting schema for table (use .table_name instead of .name)
2025-08-20 13:48:00,877 [INFO] [02] Retrieving example rows for table
2025-08-20 13:48:00,900 [INFO] Retrieved 2 relevant nodes for table: Album
2025-08-20 13:48:00,900 [INFO] [01] Getting schema for table (use .table_name instead of .name)
2025-08-20 13:48:00,901 [INFO] [02] Retrieving example rows for table
2025-08-20 13:48:00,921 [INFO] Retrieved 2 relevant nodes for table: Artist
2025-08-20 13:48:00,922 [INFO] [01] Getting schema for table (use .table_name instead of .name)
2025-08-20 13:48:00,923 [INFO] [02] Retrieving example rows for table
2025-08-20 13:48:00,940 [INFO] Retrieved 2 relevant nodes for table: Customer
2025-08-20 13:48:00,941 [INFO] [01] Getting schema for table (use .table_name instead of .name)
2025-08-20 13:48:00,942 [INFO] [02] Retrieving example rows for table
```

ERROR:
```
---------------------------------------------------------------------------
InternalError                             Traceback (most recent call last)
Cell In[11], line 27
     23         continue
     25     return "\n\n".join(context_strs)
---> 27 table_parser_component = get_table_context_and_rows_str(QUERY_1, table_schema_objs)
     28 logger.info(f"Updated table context with rows:\n{table_parser_component}")

Cell In[11], line 11, in get_table_context_and_rows_str(query_str, table_schema_objs)
      9 logger.info("[02] Retrieving example rows for table")
     10 vector_retriever = vector_index_dict[table_info_obj.table_name].as_retriever(similarity_top_k=TOP_N)
---> 11 relevant_nodes = vector_retriever.retrieve(query_str) # this will return the TextNodes we stored as vector indexes 
     12 logger.info(f"Retrieved {len(relevant_nodes)} relevant nodes for table: {table_info_obj.table_name}")
     14 if len(relevant_nodes) > 0:

File c:\Users\Hp\Documents\GitHub\rag_text-2-sql\.venv\Lib\site-packages\llama_index_instrumentation\dispatcher.py:317, in Dispatcher.span.<locals>.wrapper(func, instance, args, kwargs)
    314             _logger.debug(f"Failed to reset active_span_id: {e}")
    316 try:
--> 317     result = func(*args, **kwargs)
    318     if isinstance(result, asyncio.Future):
    319         # If the result is a Future, wrap it
    320         new_future = asyncio.ensure_future(result)

File c:\Users\Hp\Documents\GitHub\rag_text-2-sql\.venv\Lib\site-packages\llama_index\core\base\base_retriever.py:210, in BaseRetriever.retrieve(self, str_or_query_bundle)
    205 with self.callback_manager.as_trace("query"):
    206     with self.callback_manager.event(
    207         CBEventType.RETRIEVE,
    208         payload={EventPayload.QUERY_STR: query_bundle.query_str},
    209     ) as retrieve_event:
--> 210         nodes = self._retrieve(query_bundle)
    211         nodes = self._handle_recursive_retrieval(query_bundle, nodes)
    212         retrieve_event.on_end(
    213             payload={EventPayload.NODES: nodes},
    214         )

File c:\Users\Hp\Documents\GitHub\rag_text-2-sql\.venv\Lib\site-packages\llama_index_instrumentation\dispatcher.py:317, in Dispatcher.span.<locals>.wrapper(func, instance, args, kwargs)
    314             _logger.debug(f"Failed to reset active_span_id: {e}")
    316 try:
--> 317     result = func(*args, **kwargs)
    318     if isinstance(result, asyncio.Future):
    319         # If the result is a Future, wrap it
    320         new_future = asyncio.ensure_future(result)

File c:\Users\Hp\Documents\GitHub\rag_text-2-sql\.venv\Lib\site-packages\llama_index\core\indices\vector_store\retrievers\retriever.py:104, in VectorIndexRetriever._retrieve(self, query_bundle)
     98     if query_bundle.embedding is None and len(query_bundle.embedding_strs) > 0:
     99         query_bundle.embedding = (
    100             self._embed_model.get_agg_embedding_from_queries(
    101                 query_bundle.embedding_strs
    102             )
    103         )
--> 104 return self._get_nodes_with_embeddings(query_bundle)

File c:\Users\Hp\Documents\GitHub\rag_text-2-sql\.venv\Lib\site-packages\llama_index\core\indices\vector_store\retrievers\retriever.py:220, in VectorIndexRetriever._get_nodes_with_embeddings(self, query_bundle_with_embeddings)
    216 def _get_nodes_with_embeddings(
    217     self, query_bundle_with_embeddings: QueryBundle
    218 ) -> List[NodeWithScore]:
    219     query = self._build_vector_store_query(query_bundle_with_embeddings)
--> 220     query_result = self._vector_store.query(query, **self._kwargs)
    222     nodes_to_fetch = self._determine_nodes_to_fetch(query_result)
    223     if nodes_to_fetch:
    224         # Fetch any missing nodes from the docstore and insert them into the query result

File c:\Users\Hp\Documents\GitHub\rag_text-2-sql\.venv\Lib\site-packages\llama_index\vector_stores\chroma\base.py:378, in ChromaVectorStore.query(self, query, **kwargs)
    375 if not query.query_embedding:
    376     return self._get(limit=query.similarity_top_k, where=where, **kwargs)
--> 378 return self._query(
    379     query_embeddings=query.query_embedding,
    380     n_results=query.similarity_top_k,
    381     where=where,
    382     **kwargs,
    383 )

File c:\Users\Hp\Documents\GitHub\rag_text-2-sql\.venv\Lib\site-packages\llama_index\vector_stores\chroma\base.py:396, in ChromaVectorStore._query(self, query_embeddings, n_results, where, **kwargs)
    389     results = self._collection.query(
    390         query_embeddings=query_embeddings,
    391         n_results=n_results,
    392         where=where,
    393         **kwargs,
    394     )
    395 else:
--> 396     results = self._collection.query(
    397         query_embeddings=query_embeddings,
    398         n_results=n_results,
    399         **kwargs,
    400     )
    402 logger.debug(f"> Top {len(results['documents'][0])} nodes:")
    403 nodes = []

File c:\Users\Hp\Documents\GitHub\rag_text-2-sql\.venv\Lib\site-packages\chromadb\api\models\Collection.py:221, in Collection.query(self, query_embeddings, query_texts, query_images, query_uris, ids, n_results, where, where_document, include)
    185 """Get the n_results nearest neighbor embeddings for provided query_embeddings or query_texts.
    186 
    187 Args:
   (...)    206 
    207 """
    209 query_request = self._validate_and_prepare_query_request(
    210     query_embeddings=query_embeddings,
    211     query_texts=query_texts,
   (...)    218     include=include,
    219 )
--> 221 query_results = self._client._query(
    222     collection_id=self.id,
    223     ids=query_request["ids"],
    224     query_embeddings=query_request["embeddings"],
    225     n_results=query_request["n_results"],
    226     where=query_request["where"],
    227     where_document=query_request["where_document"],
    228     include=query_request["include"],
    229     tenant=self.tenant,
    230     database=self.database,
    231 )
    233 return self._transform_query_response(
    234     response=query_results, include=query_request["include"]
    235 )

File c:\Users\Hp\Documents\GitHub\rag_text-2-sql\.venv\Lib\site-packages\chromadb\api\rust.py:505, in RustBindingsAPI._query(self, collection_id, query_embeddings, ids, n_results, where, where_document, include, tenant, database)
    489 filtered_ids_amount = len(ids) if ids else 0
    490 self.product_telemetry_client.capture(
    491     CollectionQueryEvent(
    492         collection_uuid=str(collection_id),
   (...)    502     )
    503 )
--> 505 rust_response = self.bindings.query(
    506     str(collection_id),
    507     ids,
    508     query_embeddings,
    509     n_results,
    510     json.dumps(where) if where else None,
    511     json.dumps(where_document) if where_document else None,
    512     include,
    513     tenant,
    514     database,
    515 )
    517 return QueryResult(
    518     ids=rust_response.ids,
    519     embeddings=rust_response.embeddings,
   (...)    525     distances=rust_response.distances,
    526 )

InternalError: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk
```

CAN YOU HELP ME??, BE DETAILED, & EXPLAIN WHY IS IT BREAKING, i feel like it's due to the way i'm persisting data into the vector store (chroma & llamaindex clash i suppose)??