import torch
import logging
import time
import os
import threading
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class TextEmbedding:
    def __init__(self):
        model_name = 'intfloat/multilingual-e5-large-instruct'
        # Автоматически определяем устройство: используем GPU если доступен, иначе CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        start_time = time.time()
        logger.info(
            "Starting model loading",
            extra={
                "model_name": model_name,
                "device": device
            }
        )
        try:
            # Устанавливаем переменные окружения для оптимизации загрузки
            os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
            
            logger.info("Preparing model loading parameters...")
            
            # Оптимизация: используем локальный кэш и оптимизируем загрузку
            # Для CPU используем float32, для CUDA - float16
            model_kwargs = {
                "dtype": torch.float16 if device == "cuda" else torch.float32,
                "trust_remote_code": False,
            }
            
            logger.info("Initializing SentenceTransformer...")
            logger.info("This may take several minutes for large models on CPU...")
            logger.info("Model is loading from cache, please wait...")
            
            # Запускаем периодическое логирование в отдельном потоке
            self._loading = True
            self._load_error = None
            
            def log_progress():
                elapsed = 0
                while self._loading:
                    time.sleep(30)  # Логируем каждые 30 секунд
                    if self._loading:
                        elapsed += 30
                        logger.info(
                            f"Model still loading... elapsed: {elapsed}s ({elapsed//60}m {elapsed%60}s)",
                            extra={
                                "elapsed_seconds": elapsed,
                                "elapsed_minutes": round(elapsed / 60, 1)
                            }
                        )
            
            progress_thread = threading.Thread(target=log_progress, daemon=True)
            progress_thread.start()
            
            # Загружаем модель с явными параметрами
            try:
                self.embedding_model = SentenceTransformer(
                    model_name,
                    device=device,
                    model_kwargs=model_kwargs
                )
            finally:
                self._loading = False
            
            load_time = time.time() - start_time
            logger.info(
                "Model loaded successfully",
                extra={
                    "model_name": model_name,
                    "device": device,
                    "load_time_seconds": round(load_time, 2),
                    "load_time_minutes": round(load_time / 60, 2)
                }
            )
        except Exception as e:
            load_time = time.time() - start_time
            logger.error(
                "Failed to load model",
                extra={
                    "model_name": model_name,
                    "device": device,
                    "load_time_seconds": round(load_time, 2),
                    "error": str(e),
                    "error_type": type(e).__name__
                },
                exc_info=True
            )
            raise

    def vectorize_text(self, chunks):
        Data_db = {
            'id': [i for i in range(1, len(chunks) + 1)],
            'source': [chunk.metadata['source'] for chunk in chunks],
            'emb': [self.embedding_model.encode(chunk.page_content) for chunk in chunks],
            'content': [chunk.page_content for chunk in chunks]
        }
        return Data_db

    def model_emb(self):
        return self.embedding_model