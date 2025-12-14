#!/usr/bin/env python3
"""
Скрипт для загрузки всех PDF файлов из папки td в контейнер через API /upload
"""
import os
import sys
import argparse
from pathlib import Path
import requests
from typing import List
import time

# Конфигурация по умолчанию
DEFAULT_API_URL = "http://127.0.0.1:10000/api/v1/chat/upload"
DEFAULT_FOLDER_PATH = Path("td")
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
TIMEOUT = 300  # 5 минут на загрузку одного файла


def get_pdf_files(folder: Path) -> List[Path]:
    """Получить список всех PDF файлов в папке"""
    if not folder.exists():
        print(f"❌ Папка {folder} не существует!")
        return []
    
    pdf_files = list(folder.glob("*.pdf"))
    print(f"📁 Найдено {len(pdf_files)} PDF файлов в папке {folder}")
    return pdf_files


def upload_files(file_paths: List[Path], api_url: str) -> tuple[int, int, int]:
    """Загрузить все файлы одним запросом через API"""
    # Фильтруем файлы по размеру
    valid_files = []
    skipped = 0
    
    for file_path in file_paths:
        file_name = file_path.name
        file_size = file_path.stat().st_size
        
        if file_size > MAX_FILE_SIZE:
            print(f"⚠️  Пропущен {file_name}: размер {file_size / (1024*1024):.2f} MB превышает лимит {MAX_FILE_SIZE / (1024*1024):.0f} MB")
            skipped += 1
            continue
        
        if file_size == 0:
            print(f"⚠️  Пропущен {file_name}: файл пустой")
            skipped += 1
            continue
        
        valid_files.append(file_path)
    
    if not valid_files:
        print("❌ Нет файлов для загрузки после фильтрации")
        return 0, 0, skipped
    
    total_size = sum(f.stat().st_size for f in valid_files)
    print(f"📤 Загрузка {len(valid_files)} файлов (общий размер: {total_size / (1024*1024):.2f} MB)...", end=" ", flush=True)
    
    try:
        # Подготавливаем все файлы для отправки
        # В FastAPI для List[UploadFile] все файлы должны отправляться с одним ключом
        files_data = []
        for file_path in valid_files:
            file_name = file_path.name
            # Ключ должен совпадать с именем параметра в эндпоинте (files)
            files_data.append(('files', (file_name, open(file_path, 'rb'), 'application/pdf')))
        
        try:
            response = requests.post(
                api_url,
                files=files_data,
                timeout=TIMEOUT * len(valid_files)  # Увеличиваем таймаут пропорционально количеству файлов
            )
        finally:
            # Закрываем все открытые файлы
            for _, (_, file_obj, _) in files_data:
                file_obj.close()
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print(f"✅ Успешно загружено {len(valid_files)} файлов")
                return len(valid_files), 0, skipped
            else:
                print(f"❌ Ошибка: {result.get('message', 'Unknown error')}")
                return 0, len(valid_files), skipped
        else:
            error_msg = response.json().get('detail', f'HTTP {response.status_code}') if response.headers.get('content-type', '').startswith('application/json') else f'HTTP {response.status_code}'
            print(f"❌ Ошибка: {error_msg}")
            return 0, len(valid_files), skipped
            
    except requests.exceptions.Timeout:
        print(f"❌ Таймаут (превышено {TIMEOUT * len(valid_files)} секунд)")
        return 0, len(valid_files), skipped
    except requests.exceptions.ConnectionError:
        print(f"❌ Ошибка подключения к {api_url}")
        print("   Убедитесь, что контейнер запущен и доступен на порту 10000")
        return 0, len(valid_files), skipped
    except Exception as e:
        print(f"❌ Ошибка: {str(e)}")
        return 0, len(valid_files), skipped


def main():
    """Основная функция"""
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(
        description="Загрузка PDF файлов в контейнер через API /upload"
    )
    parser.add_argument(
        "--folder",
        "-f",
        type=str,
        default=str(DEFAULT_FOLDER_PATH),
        help=f"Путь к папке с PDF файлами (по умолчанию: {DEFAULT_FOLDER_PATH})"
    )
    parser.add_argument(
        "--api-url",
        "-u",
        type=str,
        default=DEFAULT_API_URL,
        help=f"URL API эндпоинта (по умолчанию: {DEFAULT_API_URL})"
    )
    args = parser.parse_args()
    
    folder_path = Path(args.folder)
    api_url = args.api_url
    
    print("=" * 60)
    print("🚀 Скрипт загрузки PDF файлов в контейнер")
    print("=" * 60)
    print(f"📂 Папка: {folder_path.absolute()}")
    print(f"🌐 API: {api_url}")
    print("=" * 60)
    print()
    
    # Получаем список файлов
    pdf_files = get_pdf_files(folder_path)
    
    if not pdf_files:
        print("❌ PDF файлы не найдены!")
        sys.exit(1)
    
    print()
    print(f"📊 Начинаем загрузку {len(pdf_files)} файлов одним запросом...")
    print()
    
    # Статистика
    start_time = time.time()
    
    # Загружаем все файлы одним запросом
    successful, failed, skipped = upload_files(pdf_files, api_url)
    
    # Итоговая статистика
    elapsed_time = time.time() - start_time
    print()
    print("=" * 60)
    print("📊 Итоговая статистика:")
    print(f"   ✅ Успешно загружено: {successful}")
    print(f"   ❌ Ошибок: {failed}")
    print(f"   ⚠️  Пропущено: {skipped}")
    print(f"   ⏱️  Время выполнения: {elapsed_time:.2f} секунд ({elapsed_time/60:.2f} минут)")
    print("=" * 60)
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

