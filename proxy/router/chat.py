import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from starlette.responses import FileResponse

from proxy.schema.chat import Chat, ChatResponse, FileDownload

from dotenv import load_dotenv

load_dotenv()

DOC_DIR = Path(os.getenv("DOC_DIR"))

router = APIRouter()

@router.post("/q")
async def getAnswer(request: Chat) -> ChatResponse:
    return ChatResponse(
        request = request.request,
        response = "Верояно, ...",
        onTextBased = ["Самолёт, ...[1]", "Вертолёт, ...[2]"],
    )

@router.post("/doc")
async def downloadDoc(doc: FileDownload) -> FileResponse:
    if not doc.docName or ".." in doc.docName or "/" in doc.docName or "\\" in doc.docName:
        raise HTTPException(status_code=400, detail="Invalid file name")

    file_path = DOC_DIR / (doc.docName + ".pdf")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Document not found")

    return FileResponse(
        path=str(file_path),
        media_type="application/pdf",
        filename=doc.docName,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Cache-Control": "public, max-age=3600"
        }
    )
