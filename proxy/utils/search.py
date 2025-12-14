import logging
from proxy.utils.MilvusSingleton_impl import MilvusSingleton
from proxy.utils.TextEncoder_impl import TextEmbedding
import numpy as np

logger = logging.getLogger(__name__)

# Ленивая инициализация модели - загружается только при первом использовании
_emb = None

def get_embedding_model():
    """Получить модель эмбеддингов (ленивая инициализация)"""
    global _emb
    if _emb is None:
        logger.info("Initializing embedding model (first use)")
        _emb = TextEmbedding()
        logger.info("Embedding model initialized successfully")
    return _emb


def poisk(query, name_db="rag_db", collec="docs"):
    logger.info(
        "Starting search",
        extra={
            "query": query,
            "database": name_db,
            "collection": collec
        }
    )
    
    try:
        milvus = MilvusSingleton(host="localhost", port="19530")
        milvus.setup_database(name_db)
        logger.debug("Database setup completed", extra={"database": name_db})

        logger.debug("Encoding query to vector")
        emb = get_embedding_model()
        query_vec = np.asarray(emb.embedding_model.encode(query), dtype=np.float32).tolist()
        
        logger.debug("Searching by vector", extra={"limit": 15})
        milv_id = milvus.search_by_vector(query_vec, collec, limit=15)

        retry_count = 0
        while not milv_id['id']:
            retry_count += 1
            logger.warning(
                "Milvus no results found, retrying",
                extra={
                    "retry_count": retry_count,
                    "query": query
                }
            )
            milv_id = milvus.search_milvus(query, collec, limit=10)
        
        logger.info(
            "Search results from Milvus",
            extra={
                "result_ids": milv_id['id'],
                "results_count": len(milv_id['id']) if milv_id['id'] else 0
            }
        )

        res_chunks = []
        for i in range(len(milv_id['id'])):
            res_chunks.append({"text": milv_id['content'][i], "source": milv_id['source'][i]})

        logger.info(
            "Search completed successfully",
            extra={
                "chunks_count": len(res_chunks),
                "query": query
            }
        )
        logger.debug("Retrieved chunks", extra={"chunks": res_chunks})
        
        return res_chunks
    except Exception as e:
        logger.error(
            "Error during search",
            extra={
                "query": query,
                "database": name_db,
                "collection": collec,
                "error": str(e)
            },
            exc_info=True
        )
        raise
