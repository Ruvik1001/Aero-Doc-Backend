import json
import numpy as np
from typing import List
from pathlib import Path
from datetime import datetime
import logging

from proxy.utils.TextChunker_impl import TextChunker
from proxy.utils.TextEncoder_impl import TextEmbedding
from proxy.utils.MilvusSingleton_impl import MilvusSingleton

import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DOC_DIR = Path(os.getenv("DOC_DIR"))

# Ленивая инициализация моделей - загружаются только при первом использовании
_emb = None
_text_docs = None

def get_embedding_model():
    """Получить модель эмбеддингов (ленивая инициализация)"""
    global _emb
    if _emb is None:
        logger.info("Initializing embedding model (first use)")
        _emb = TextEmbedding()
        logger.info("Embedding model initialized successfully")
    return _emb

def get_text_chunker():
    """Получить TextChunker (ленивая инициализация)"""
    global _text_docs
    if _text_docs is None:
        logger.info("Initializing TextChunker (first use)")
        _text_docs = TextChunker()
        logger.info("TextChunker initialized successfully")
    return _text_docs


def poisk(query, name_db="rag_db", collec="docs"):
    milvus = MilvusSingleton(host="standalone", port="19530")
    milvus.setup_database(name_db)

    emb = get_embedding_model()
    query_vec = np.asarray(emb.embedding_model.encode(query), dtype=np.float32).tolist()
    milv_id = milvus.search_by_vector(query_vec, collec, limit=15)

    while not milv_id['id']:
        print("[INFO]: Milvus no results found, retrying...")
        milv_id = milvus.search_milvus(query, collec, limit=10)
    print("[INFO]: Search results milvus:", milv_id['id'])

    print("[INFO]: Relevant chunks found:", milv_id['id'])

    res_chunks = []
    for i in range(len(milv_id['id'])):
        res_chunks.append({"text": milv_id['content'][i], "source": milv_id['source'][i]})

    print("[INFO]: Promt successfully:\n", res_chunks)
    return res_chunks


def parser(files: List[str]):
    logger.info("Starting document parsing", extra={"files": files})
    existing_records = []
    if Path("files_chunks.json").exists():
        raw = Path("files_chunks.json").read_text(encoding="utf-8").strip()
        if raw:
            try:
                data = json.loads(raw)
                if isinstance(data, list):
                    existing_records = data
                else:
                    raise ValueError("JSON is not an array")
            except Exception:
                backup = Path("files_chunks.json").with_name(
                    f"files_chunks_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                backup.write_text(raw, encoding="utf-8")
                existing_records = []

    # 2) Считаем следующий id
    max_id = 0
    for item in existing_records:
        if isinstance(item, dict) and "id" in item:
            try:
                max_id = max(max_id, int(item["id"]))
            except Exception:
                pass
    next_id = max_id + 1

    # 3) Генерируем новые записи и добавляем
    new_records = []

    logger.info("Loading models for parsing")
    emb = get_embedding_model()
    text_docs = get_text_chunker()
    logger.info("Models loaded, starting document processing")
    
    for file_name in files:
        file_path = DOC_DIR / file_name
        logger.info("Processing file", extra={"file_name": file_name, "file_path": str(file_path)})

        docs = text_docs.load_pdf_documents(file_path)
        logger.info("PDF loaded, splitting into chunks", extra={"pages_count": len(docs)})
        chunks = text_docs.splitting(docs)
        logger.info("Chunks created, starting vectorization", extra={"chunks_count": len(chunks)})
        vectors = emb.vectorize_text(chunks)
        logger.info("Vectorization completed")

        for i, chunk in enumerate(chunks):
            vec = vectors["emb"][i]
            if hasattr(vec, "tolist"):
                vec = vec.tolist()

            new_records.append(
                {
                    "id": next_id,
                    "source": chunk.metadata.get("source", str(file_path)),
                    "embeddings": vec,
                    "content": chunk.page_content,
                }
            )
            next_id += 1

    existing_records.extend(new_records)

    # 4) Сохраняем обратно (валидный JSON-массив)
    Path("files_chunks.json").write_text(
        json.dumps(existing_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info(
        "Chunks added to JSON",
        extra={
            "new_chunks": len(new_records),
            "total_records": len(existing_records)
        }
    )

    logger.info("Starting Milvus data push")
    push_milv()
    logger.info("Document parsing completed successfully", extra={"files": files})


def push_milv(name_db="rag_db", collec="docs"):
    json_path = Path("files_chunks.json")
    rows = json.loads(json_path.read_text(encoding="utf-8"))

    milvus = MilvusSingleton(host="standalone", port="19530")
    milvus.setup_database(name_db)

    DIM = int(np.asarray(rows[0]["embeddings"]).size)
    milvus.create_collection(collec, size_vec=DIM, drop_if_exists=True)

    max_bytes = 40 * 1024 * 1024

    ids, sources, embs, contents = [], [], [], []
    batch_bytes = 0
    total = 0

    def send():
        nonlocal ids, sources, embs, contents, batch_bytes, total
        if not ids:
            return
        milvus.insert_data(
            collec,
            {"id": ids, "source": sources, "embeddings": embs, "content": contents},
            flush=False
        )
        total += len(ids)
        ids, sources, embs, contents = [], [], [], []
        batch_bytes = 0

    for r in rows:
        src = str(r.get("source", ""))
        txt = str(r.get("content", ""))

        emb = np.asarray(r["embeddings"], dtype=np.float32).reshape(-1).tolist()

        row_bytes = (len(emb) * 4) + len(src.encode("utf-8")) + len(txt.encode("utf-8")) + 256

        if ids and (batch_bytes + row_bytes > max_bytes):
            send()

        ids.append(int(r["id"]))
        sources.append(src)
        embs.append(emb)
        contents.append(txt)
        batch_bytes += row_bytes

    send()

    col = milvus.get_collection(collec)
    col.flush()
    col.load()

    return total
