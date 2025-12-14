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
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Отключаем параллелизм токенизатора
            
            logger.info("Preparing model loading parameters...")
            logger.info(f"PyTorch version: {torch.__version__}")
            logger.info(f"Device: {device}")
            
            # Запускаем периодическое логирование в отдельном потоке
            self._loading = True
            
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
            
            logger.info("Initializing SentenceTransformer...")
            logger.info("Loading model (this may take 2-3 minutes on CPU)...")
            
            # Упрощенная загрузка без дополнительных параметров
            # SentenceTransformer сам определит оптимальные настройки
            try:
                # Для CPU не передаем model_kwargs с dtype, чтобы избежать проблем
                if device == "cuda":
                    self.embedding_model = SentenceTransformer(
                        model_name,
                        device=device,
                        model_kwargs={"torch_dtype": torch.float16}
                    )
                else:
                    # Для CPU загружаем без дополнительных параметров
                    self.embedding_model = SentenceTransformer(
                        model_name,
                        device=device
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