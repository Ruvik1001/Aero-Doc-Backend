import logging
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class TextChunker:
    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def load_pdf_documents(self, pdf_path: Path) -> List[Document]:
        return PDFPlumberLoader(str(pdf_path)).load()

    def splitting(self, docs: List[Document]) -> List[Document]:
        if not docs:
            logger.warning("Empty documents list provided to splitting")
            return []
        
        # Логируем информацию о документах перед разбиением
        total_text_length = sum(len(doc.page_content) for doc in docs)
        logger.info(
            "Splitting documents into chunks",
            extra={
                "docs_count": len(docs),
                "total_text_length": total_text_length,
                "chunk_size": self.splitter._chunk_size,
                "chunk_overlap": self.splitter._chunk_overlap
            }
        )
        
        chunks = self.splitter.split_documents(docs)
        
        logger.info(
            "Documents split into chunks",
            extra={
                "input_docs": len(docs),
                "output_chunks": len(chunks)
            }
        )
        
        return chunks