from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex
from llama_index.core.retrievers import SQLRetriever

from utils.llm.get_llm_func import get_embedding_func


def build_object_index(table_schema_objs, table_node_mapping, logger):
    """Build ObjectIndex for retrieval from table summaries."""
    logger.info("Building object index for table retrieval")
    obj_index = ObjectIndex.from_objects(
        table_schema_objs,
        table_node_mapping,
        VectorStoreIndex,
        embed_model=get_embedding_func(),
    )
    return obj_index


def create_retrievers(sql_database, obj_index, top_k: int, logger):
    """Create object retriever and SQL retriever."""
    logger.info("Creating retrievers for query execution")
    obj_retriever = obj_index.as_retriever(similarity_top_k=top_k)
    sql_retriever = SQLRetriever(sql_database)
    return obj_retriever, sql_retriever