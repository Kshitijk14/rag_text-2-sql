from sqlalchemy import text

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.storage import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

from utils.llm.get_llm_func import get_embedding_func


def get_chroma_collection(chroma_client, table_name, logger):
    """Get or create a persistent ChromaDB collection for the given table."""
    logger.info(f"[01] Processing table: {table_name}")
    return chroma_client.get_or_create_collection(name=f"table_{table_name}")


def fetch_table_rows(engine, table_name, logger):
    """Fetch all rows and column names from a SQL table."""
    with engine.connect() as conn:
        result = conn.execute(text(f'SELECT * FROM "{table_name}"'))
        col_names = list(result.keys())
        rows = result.fetchall()

    if not rows:
        logger.warning(f"[02.1] Table {table_name} is empty, skipping...")
        return None, None
    logger.info(f"[02.2] Preparing {len(rows)} rows for indexing...")
    return col_names, rows


def prepare_new_nodes(collection, table_name, col_names, rows, logger):
    """Check which rows are new and prepare nodes for embedding."""
    row_ids = [f"{table_name}_row_{idx}" for idx, _ in enumerate(rows)]
    existing_ids = set(collection.get(ids=row_ids)["ids"]) if collection.count() > 0 else set()
    logger.info(f"[02.3] Found {len(existing_ids)} existing rows in Chroma for {table_name}")

    new_nodes = []
    for idx, row in enumerate(rows):
        row_id = row_ids[idx]
        if row_id not in existing_ids:  # new row
            row_text = " | ".join([f"{col}={val}" for col, val in zip(col_names, row)])
            new_nodes.append(TextNode(text=row_text, id_=row_id))

    if new_nodes:
        logger.info(f"[02.4] Adding {len(new_nodes)} new rows to index for {table_name}")
    else:
        logger.info(f"[02.4] No new rows to add for {table_name}")

    return new_nodes


def update_index(collection, new_nodes):
    """Embed and update the index with new nodes, then rebuild the view."""
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if new_nodes:
        _ = VectorStoreIndex(
            new_nodes,
            storage_context=storage_context,
            embed_model=get_embedding_func()
        )

    # Always rebuild view
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=get_embedding_func()
    )