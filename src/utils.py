"""
utils.py
--------
Вспомогательные функции: настройка логирования и запись результатов в CSV.
"""

import csv
import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(log_dir: str | Path, name: str = "parkvision") -> logging.Logger:
    """
    Настраивает логгер — пишет одновременно в консоль и в файл.

    Args:
        log_dir: папка куда сохранять лог-файл
        name: имя логгера

    Returns:
        Настроенный Logger.

    Пример использования:
        logger = setup_logger("output/logs")
        logger.info("Система запущена")
        logger.warning("Номер не распознан")
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Хэндлер для консоли
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Хэндлер для файла
    log_file = log_dir / f"parkvision_{_today()}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def setup_csv(output_dir: str | Path) -> Path:
    """
    Создаёт CSV-файл для записи распознанных номеров.
    Если файл уже существует — не перезаписывает, а дописывает в конец.

    Args:
        output_dir: папка для сохранения CSV

    Returns:
        Путь к созданному CSV-файлу.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"plates_{_today()}.csv"

    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "plate", "confidence", "source"])

    return csv_path


def write_plate(
    csv_path: str | Path,
    plate_text: str,
    confidence: float,
    source: str,
) -> None:
    """
    Дописывает одну строку с распознанным номером в CSV-файл.

    Args:
        csv_path: путь к CSV-файлу (от setup_csv)
        plate_text: распознанный номер, например "А123ВС456"
        confidence: уверенность распознавания (0.0 — 1.0)
        source: источник кадра (имя файла или RTSP-адрес)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    confidence_pct = round(confidence * 100, 1)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, plate_text, f"{confidence_pct}%", source])


def _today() -> str:
    """Возвращает текущую дату в формате YYYY-MM-DD для имён файлов."""
    return datetime.now().strftime("%Y-%m-%d")