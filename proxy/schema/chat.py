from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional, List

class Chat(BaseModel):
    request: str

class ChatResponse(Chat):
    response: str
    onTextBased: List[str]

class FileDownload(BaseModel):
    docName: str

class FileUploadResponse(BaseModel):
    success: bool
    message: str
    filename: str
    file_path: str