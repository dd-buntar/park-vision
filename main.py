"""
main.py
-------
Точка входа в систему ParkVision.

Примеры запуска:
    # Видеофайл (только уникальные номера — режим по умолчанию):
    python main.py --source video.mp4

    # Видеофайл (все кадры с номерами):
    python main.py --source video.mp4 --save-all

    # Подключение к IP-камере по RTSP:
    python main.py --source rtsp://192.168.1.1:554/stream

    # Одно изображение:
    python main.py --source photo.jpg

    # Папка с изображениями:
    python main.py --folder images/

    # Указать другую папку для результатов:
    python main.py --source video.mp4 --output results/

    # Запуск только на CPU (если нет GPU):
    python main.py --source video.mp4 --no-gpu
"""

import argparse
import sys
from pathlib import Path

from src.detector import Detector
from src.processor import Processor, IMAGE_EXTENSIONS
from src.recognizer import Recognizer
from src.utils import setup_csv, setup_logger

# Путь к весам моделей по умолчанию
DEFAULT_YOLO_MODEL = "models/best.pt"
DEFAULT_LPRNET_WEIGHTS = "models/LPRNet_Ep_BEST_model.ckpt"
DEFAULT_STN_WEIGHTS = "models/SpatialTransformer_Ep_BEST_model.ckpt"

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
        default=None,
        help=(
            "Источник: путь к видеофайлу (.mp4, .avi, .mov), "
            "изображению (.jpg, .png) или RTSP-адрес камеры."
        ),
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Путь к папке с изображениями для пакетной обработки.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_YOLO_MODEL,
        help=f"Путь к файлу весов YOLOv8 (по умолчанию: {DEFAULT_YOLO_MODEL}).",
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
        "--save-all",
        action="store_true",
        help=(
            "Сохранять скриншот и запись в CSV для каждого кадра с номером. "
            "По умолчанию сохраняются только уникальные номера."
        ),
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Запустить на CPU вместо GPU.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Проверяем что указан хотя бы один источник
    if not args.source and not args.folder:
        print("Ошибка: укажите --source или --folder.")
        print("Пример: python main.py --source video.mp4")
        print("        python main.py --folder images/")
        sys.exit(1)

    # Настраиваем логирование
    logger = setup_logger(DEFAULT_LOG_DIR)
    logger.info("=" * 50)
    logger.info("ParkVision запущен")
    if args.source:
        logger.info(f"Источник:  {args.source}")
    if args.folder:
        logger.info(f"Папка:     {args.folder}")
    logger.info(f"Модель:    {args.model}")
    logger.info(f"Устройство: {'CPU' if args.no_gpu else 'GPU'}")
    logger.info(f"Режим:     {'все кадры' if args.save_all else 'только уникальные номера'}")
    logger.info("=" * 50)

    # Проверяем что файл весов YOLO существует
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(
            f"Файл весов не найден: {model_path}\n"
            "Скачайте веса и положите в папку models/.\n"
            "Инструкция: см. README.md, раздел 'Установка модели'."
        )
        sys.exit(1)

    device = "cpu" if args.no_gpu else "cuda"

    # Инициализируем компоненты
    logger.info("Загрузка модели детекции (YOLOv8)...")
    detector = Detector(
        model_path=model_path,
        confidence=args.confidence,
        device=device,
    )

    logger.info("Загрузка модели распознавания (LPRNet)...")
    recognizer = Recognizer(
        lprnet_weights=DEFAULT_LPRNET_WEIGHTS,
        stn_weights=DEFAULT_STN_WEIGHTS,
        gpu=not args.no_gpu,
    )

    csv_path = setup_csv(DEFAULT_LOG_DIR)

    processor = Processor(
        detector=detector,
        recognizer=recognizer,
        csv_path=csv_path,
        output_dir=args.output,
        save_all=args.save_all,
        logger=logger,
    )

    # Запускаем нужный режим
    if args.folder:
        folder_path = Path(args.folder)
        if not folder_path.exists():
            logger.error(f"Папка не найдена: {folder_path}")
            sys.exit(1)
        processor.process_folder(args.folder)

    elif args.source:
        source = args.source
        source_path = Path(source)

        if source_path.suffix.lower() in IMAGE_EXTENSIONS:
            if not source_path.exists():
                logger.error(f"Файл не найден: {source}")
                sys.exit(1)
            processor.process_image(source)

        else:
            if not source.startswith("rtsp://") and not source_path.exists():
                logger.error(f"Файл не найден: {source}")
                sys.exit(1)
            processor.process(source)


if __name__ == "__main__":
    main()