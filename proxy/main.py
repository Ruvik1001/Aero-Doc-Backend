import time

from fastapi import FastAPI, Request
from dotenv import load_dotenv
import os

from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

load_dotenv()

import logging
from pythonjsonlogger import jsonlogger

from fastapi.middleware.cors import CORSMiddleware

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Удаляем все существующие обработчики
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Создаем форматтер для JSON
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s %(service)s %(method)s %(endpoint)s %(status_code)s %(duration_sec)s',
        json_ensure_ascii=False,
        rename_fields={
            "asctime": "timestamp",
            "levelname": "level",
            "name": "logger"
        }
    )

    # Обработчик для вывода в консоль (Docker будет собирать эти логи)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        logger = logging.getLogger(__name__)
        
        # Логируем входящий запрос
        logger.info(
            "Incoming request",
            extra={
                "service": "request_manager_service",
                "method": request.method,
                "endpoint": str(request.url.path),
                "client_ip": request.client.host if request.client else None,
            }
        )
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Логируем успешный ответ
            logger.info(
                "Request completed",
                extra={
                    "service": "request_manager_service",
                    "method": request.method,
                    "endpoint": str(request.url.path),
                    "status_code": response.status_code,
                    "duration_sec": round(duration, 3),
                }
            )
            
            return response
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                "Request failed",
                extra={
                    "service": "request_manager_service",
                    "method": request.method,
                    "endpoint": str(request.url.path),
                    "status_code": 500,
                    "duration_sec": round(duration, 3),
                    "error": str(e),
                },
                exc_info=True
            )
            raise


def create_app() -> FastAPI:
    setup_logging()

    app = FastAPI(
        title="Request manager Service",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    app.add_middleware(LoggingMiddleware)
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


app = create_app()

from proxy.router import (
    health,
    chat
)

app.include_router(
    health.router,
    prefix="/api/v1/health",
    tags=["Health Monitoring"]
)

app.include_router(
    chat.router,
    prefix="/api/v1/chat",
    tags=["Chat"]
)