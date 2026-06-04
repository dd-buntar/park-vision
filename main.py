"""
main.py
-------
Точка входа в систему ParkVision.

Примеры запуска:
    # Обработка видеофайла:
    python main.py --source video.mp4

    # Подключение к IP-камере по RTSP:
    python main.py --source rtsp://192.168.1.1:554/stream

    # Указать другую папку для результатов:
    python main.py --source video.mp4 --output results/

    # Запуск только на CPU (если нет GPU):
    python main.py --source video.mp4 --no-gpu
"""

import argparse
import sys
from pathlib import Path

from src.detector import Detector
from src.processor import Processor
from src.recognizer import Recognizer
from src.utils import setup_csv, setup_logger

# Путь к весам модели по умолчанию
DEFAULT_MODEL_PATH = "models/best.pt"

# Папки для результатов по умолчанию
DEFAULT_OUTPUT_DIR = "output/screenshots"
DEFAULT_LOG_DIR = "output/logs"


def parse_args() -> argparse.Namespace:
    """Парсит аргументы командной строки."""
    parser = argparse.ArgumentParser(
        prog="parkvision",
        description="ParkVision — система распознавания автомобильных номеров.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Источник видео: путь к файлу (.mp4, .avi, .mov) или RTSP-адрес камеры.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Путь к файлу весов YOLOv8 (по умолчанию: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Папка для сохранения скриншотов (по умолчанию: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.45,
        help="Порог уверенности детекции от 0.0 до 1.0 (по умолчанию: 0.45).",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Запустить на CPU вместо GPU.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Настраиваем логирование
    logger = setup_logger(DEFAULT_LOG_DIR)
    logger.info("=" * 50)
    logger.info("ParkVision запущен")
    logger.info(f"Источник: {args.source}")
    logger.info(f"Модель:   {args.model}")
    logger.info(f"Устройство: {'CPU' if args.no_gpu else 'GPU'}")
    logger.info("=" * 50)

    # Проверяем что файл весов существует
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(
            f"Файл весов не найден: {model_path}\n"
            "Скачайте веса и положите в папку models/.\n"
            "Инструкция: см. README.md, раздел 'Установка модели'."
        )
        sys.exit(1)

    # Проверяем источник — если это файл, он должен существовать
    source = args.source
    if not source.startswith("rtsp://") and not Path(source).exists():
        logger.error(f"Видеофайл не найден: {source}")
        sys.exit(1)

    device = "cpu" if args.no_gpu else "cuda"

    # Инициализируем компоненты
    logger.info("Загрузка модели детекции...")
    detector = Detector(
        model_path=model_path,
        confidence=args.confidence,
        device=device,
    )

    logger.info("Загрузка модели распознавания...")
    recognizer = Recognizer(gpu=not args.no_gpu)

    csv_path = setup_csv("output/logs")

    processor = Processor(
        detector=detector,
        recognizer=recognizer,
        csv_path=csv_path,
        output_dir=args.output,
        logger=logger,
    )

    # Запускаем обработку
    processor.process(source)


if __name__ == "__main__":
    main()