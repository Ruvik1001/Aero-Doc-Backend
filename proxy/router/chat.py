import os
import logging
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from starlette.responses import FileResponse

from proxy.utils.giga import giga_answer
from proxy.utils.search import poisk, parser

from proxy.schema.chat import Chat, ChatResponse, FileDownload, FileUploadResponse

from dotenv import load_dotenv

load_dotenv()

DOC_DIR = Path(os.getenv("DOC_DIR"))
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/q")
async def getAnswer(request: Chat) -> ChatResponse:
    logger.info(
        "Received question request",
        extra={
            "query": request.request,
            "endpoint": "/q"
        }
    )
    
    try:
        logger.info("Starting search for relevant fragments")
        fragments = poisk(query=request.request)
        logger.info(
            "Search completed",
            extra={
                "fragments_count": len(fragments) if fragments else 0
            }
        )

        if not fragments or len(fragments) < 3:
            logger.warning(
                "Insufficient fragments found",
                extra={
                    "fragments_count": len(fragments) if fragments else 0,
                    "query": request.request
                }
            )
            return ChatResponse(
                request = request.request,
                response = "Не смогли найти информацию в нашей базе, пожалуйста, переформулируйте ваш вопрос.",
                onTextBased = fragments,
            )

        logger.info("Generating answer using GigaChat")
        response = giga_answer(query=request.request, fragments=fragments)
        logger.info(
            "Answer generated successfully",
            extra={
                "response_length": len(response) if response else 0
            }
        )

        return ChatResponse(
            request = request.request,
            response = response,
            onTextBased = fragments,
        )
    except Exception as e:
        logger.error(
            "Error processing question request",
            extra={
                "query": request.request,
                "error": str(e)
            },
            exc_info=True
        )
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/doc")
async def downloadDoc(doc: FileDownload) -> FileResponse:
    logger.info(
        "Received document download request",
        extra={
            "doc_name": doc.docName,
            "endpoint": "/doc"
        }
    )
    
    try:
        if not doc.docName or ".." in doc.docName or "/" in doc.docName or "\\" in doc.docName:
            logger.warning(
                "Invalid file name provided",
                extra={
                    "doc_name": doc.docName
                }
            )
            raise HTTPException(status_code=400, detail="Invalid file name")

        if (".pdf" not in doc.docName) and (".PDF" not in doc.docName):
            doc.docName += ".pdf"

        file_path = DOC_DIR / doc.docName
        logger.debug(
            "Constructed file path",
            extra={
                "file_path": str(file_path),
                "doc_dir": str(DOC_DIR)
            }
        )

        if not file_path.exists():
            logger.warning(
                "Document not found",
                extra={
                    "file_path": str(file_path),
                    "doc_name": doc.docName
                }
            )
            raise HTTPException(status_code=404, detail="Document not found")

        logger.info(
            "Serving document",
            extra={
                "file_path": str(file_path),
                "doc_name": doc.docName
            }
        )
        
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Error serving document",
            extra={
                "doc_name": doc.docName,
                "error": str(e)
            },
            exc_info=True
        )
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/upload", response_model=FileUploadResponse)
async def uploadDoc(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)) -> FileUploadResponse:
    logger.info(
        "Received document upload request",
        extra={
            "files_count": len(files),
            "endpoint": "/upload"
        }
    )
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    try:
        safe_filenames = []
        saved_files = []
        
        # Создаем директорию, если её нет
        DOC_DIR.mkdir(parents=True, exist_ok=True)
        
        # Обрабатываем каждый файл
        for file in files:
            # Валидация типа файла
            if file.content_type != "application/pdf":
                logger.warning(
                    "Invalid file type",
                    extra={
                        "file_name": file.filename,
                        "content_type": file.content_type
                    }
                )
                raise HTTPException(
                    status_code=400, 
                    detail=f"Only PDF files are allowed. File '{file.filename}' has type '{file.content_type}'"
                )
            
            # Валидация имени файла
            if not file.filename:
                logger.warning("Empty filename provided")
                raise HTTPException(status_code=400, detail="Filename is required")
            
            # Очистка имени файла от опасных символов
            safe_filename = file.filename.replace("..", "").replace("/", "").replace("\\", "")
            if not safe_filename or safe_filename != file.filename:
                logger.warning(
                    "Invalid characters in filename",
                    extra={
                        "original_filename": file.filename,
                        "safe_filename": safe_filename
                    }
                )
                raise HTTPException(status_code=400, detail=f"Invalid file name: {file.filename}")
            
            # Убеждаемся, что файл имеет расширение .pdf
            if not safe_filename.lower().endswith('.pdf'):
                safe_filename = safe_filename.rsplit('.', 1)[0] + '.pdf'
            
            file_path = DOC_DIR / safe_filename
            
            # Проверка размера файла
            file_content = await file.read()
            file_size = len(file_content)
            
            if file_size > MAX_FILE_SIZE:
                logger.warning(
                    "File size exceeds limit",
                    extra={
                        "file_name": safe_filename,
                        "file_size": file_size,
                        "max_size": MAX_FILE_SIZE
                    }
                )
                raise HTTPException(
                    status_code=400,
                    detail=f"File '{safe_filename}' size ({file_size / (1024*1024):.2f} MB) exceeds maximum allowed size of {MAX_FILE_SIZE / (1024*1024):.0f} MB"
                )
            
            if file_size == 0:
                logger.warning("Empty file uploaded", extra={"file_name": safe_filename})
                raise HTTPException(status_code=400, detail=f"File '{safe_filename}' is empty")
            
            # Сохраняем файл
            with open(file_path, "wb") as f:
                f.write(file_content)
            
            safe_filenames.append(safe_filename)
            saved_files.append({
                "filename": safe_filename,
                "file_path": str(file_path),
                "file_size": file_size
            })
        
        # Обработку файлов (парсинг и векторизация) выполняем в фоне
        # передаем весь список файлов в парсер
        background_tasks.add_task(parser, safe_filenames)
        
        total_size = sum(f["file_size"] for f in saved_files)
        logger.info(
            "Documents uploaded successfully, processing started in background",
            extra={
                "files_count": len(safe_filenames),
                "total_size": total_size,
                "files": safe_filenames
            }
        )
        
        return FileUploadResponse(
            success=True,
            message=f"Successfully uploaded {len(safe_filenames)} document(s). Processing started in background.",
            filename=", ".join(safe_filenames) if len(safe_filenames) <= 3 else f"{len(safe_filenames)} files",
            file_path=str(DOC_DIR)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Error uploading document",
            extra={
                "file_name": file.filename if file else None,
                "error": str(e)
            },
            exc_info=True
        )
        raise HTTPException(status_code=500, detail="Internal server error")
