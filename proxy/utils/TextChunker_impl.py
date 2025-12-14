from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunker:
    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def load_pdf_documents(self, pdf_path: Path) -> List[Document]:
        return PDFPlumberLoader(str(pdf_path)).load()

    def splitting(self, docs: List[Document]) -> List[Document]:
        return self.splitter.split_documents(docs)