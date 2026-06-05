"""
processor.py
------------
Основной модуль обработки видеопотока, изображений и папок.
Читает кадры из файла, RTSP-потока или папки с изображениями,
запускает детекцию и распознавание, рисует результаты на кадре,
сохраняет скриншоты и пишет CSV-лог.

Для корректного отображения кириллицы используется PIL вместо cv2.putText.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.detector import Detector, FrameDetections, Detection
from src.recognizer import Recognizer, RecognitionResult
from src.utils import write_plate

# Цвета рамок в формате BGR
COLOR_CAR = (0, 0, 255)      # красный — машина
COLOR_PLATE = (0, 255, 0)    # зелёный — номерной знак
COLOR_TEXT = (255, 255, 255) # белый — текст

# Минимальная уверенность распознавания для записи в лог
MIN_CONFIDENCE_TO_LOG = 0.5

# Каждый N-й кадр подаётся на детекцию (остальные пропускаются).
DETECTION_EVERY_N_FRAMES = 3

# Поддерживаемые форматы изображений
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# Размер шрифта для подписей
FONT_SIZE = 18

# Шрифты которые пробуем загрузить по очереди (Windows/Linux)
FONT_CANDIDATES = [
    "arial.ttf",
    "Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _load_font(size: int = FONT_SIZE) -> ImageFont.FreeTypeFont:
    """Загружает шрифт поддерживающий кириллицу."""
    for font_path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(font_path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


class Processor:
    """
    Обрабатывает видеопоток, одиночные изображения и папки с изображениями:
    детектирует номера, распознаёт текст, визуализирует и логирует результаты.

    Пример использования:
        processor = Processor(detector, recognizer, csv_path, output_dir)

        # Видеофайл:
        processor.process("video.mp4")

        # RTSP-поток:
        processor.process("rtsp://192.168.1.1:554/stream")

        # Одно изображение:
        processor.process_image("photo.jpg")

        # Папка с изображениями:
        processor.process_folder("images/")
    """

    def __init__(
        self,
        detector: Detector,
        recognizer: Recognizer,
        csv_path: str | Path,
        output_dir: str | Path,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Args:
            detector: инициализированный модуль детекции
            recognizer: инициализированный модуль распознавания
            csv_path: путь к CSV-файлу для записи номеров
            output_dir: папка для сохранения скриншотов
            logger: логгер (если None — создаётся стандартный)
        """
        self.detector = detector
        self.recognizer = recognizer
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger("parkvision")
        self.font = _load_font()

    def process(self, source: str) -> None:
        """
        Запускает обработку видеопотока до его окончания или нажатия Q.

        Args:
            source: путь к видеофайлу или RTSP-адрес камеры
        """
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            self.logger.error(f"Не удалось открыть источник: {source}")
            return

        self.logger.info(f"Запуск обработки: {source}")
        frame_count = 0
        plates_found = 0

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    self.logger.info("Поток завершён.")
                    break

                frame_count += 1

                if frame_count % DETECTION_EVERY_N_FRAMES != 0:
                    continue

                detections = self.detector.detect(frame)

                if not detections.plates:
                    continue

                annotated = self._annotate_frame(frame, detections, source)
                plates_found += len(detections.plates)
                self._save_screenshot(annotated, frame_count)

                try:
                    cv2.imshow("ParkVision", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.logger.info("Остановлено пользователем.")
                        break
                except cv2.error:
                    pass

        finally:
            cap.release()
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
            self.logger.info(
                f"Обработка завершена. Кадров: {frame_count}, "
                f"номеров найдено: {plates_found}."
            )

    def process_image(self, image_path: str | Path) -> None:
        """
        Обрабатывает одно изображение.

        Args:
            image_path: путь к изображению
        """
        image_path = Path(image_path)
        frame = cv2.imread(str(image_path))

        if frame is None:
            self.logger.error(f"Не удалось открыть изображение: {image_path}")
            return

        detections = self.detector.detect(frame)

        if not detections.plates:
            self.logger.debug(f"Номера не найдены: {image_path.name}")
            return

        annotated = self._annotate_frame(frame, detections, str(image_path))

        stem = image_path.stem
        frame_number = int(stem) if stem.isdigit() else abs(hash(stem)) % 999999
        self._save_screenshot(annotated, frame_number)

        self.logger.debug(f"Обработано: {image_path.name}")

    def process_folder(self, folder_path: str | Path) -> None:
        """
        Обрабатывает все изображения в папке.

        Args:
            folder_path: путь к папке с изображениями
        """
        folder_path = Path(folder_path)

        if not folder_path.exists():
            self.logger.error(f"Папка не найдена: {folder_path}")
            return

        images = [
            f for f in sorted(folder_path.iterdir())
            if f.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if not images:
            self.logger.error(
                f"Изображения не найдены в папке: {folder_path}\n"
                f"Поддерживаемые форматы: {', '.join(IMAGE_EXTENSIONS)}"
            )
            return

        self.logger.info(f"Найдено изображений: {len(images)}")

        for i, image_path in enumerate(images, start=1):
            self.process_image(image_path)
            if i % 10 == 0:
                self.logger.info(f"Прогресс: {i}/{len(images)}")

        self.logger.info(f"Готово. Обработано {len(images)} изображений.")

    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: FrameDetections,
        source: str,
    ) -> np.ndarray:
        """
        Рисует рамки и подписи на кадре, пишет номера в CSV.

        Args:
            frame: исходный кадр
            detections: результаты детекции
            source: источник (для записи в CSV)

        Returns:
            Кадр с нанесёнными аннотациями.
        """
        annotated = frame.copy()

        for car in detections.cars:
            annotated = self._draw_box(annotated, car, COLOR_CAR, "car")

        for plate in detections.plates:
            result = self.recognizer.recognize(frame, plate)
            label = self._build_label(result)
            annotated = self._draw_box(annotated, plate, COLOR_PLATE, label)

            if (
                result is not None
                and result.is_valid
                and result.confidence >= MIN_CONFIDENCE_TO_LOG
            ):
                write_plate(
                    csv_path=self.csv_path,
                    plate_text=result.plate_text,
                    confidence=result.confidence,
                    source=source,
                )
                self.logger.info(
                    f"Номер: {result.plate_text} "
                    f"(уверенность: {result.confidence:.0%})"
                )

        return annotated

    def _draw_box(
        self,
        frame: np.ndarray,
        detection: Detection,
        color: tuple[int, int, int],
        label: str,
    ) -> np.ndarray:
        """
        Рисует прямоугольник и подпись на кадре.
        Использует PIL для текста чтобы корректно отображать кириллицу.

        Args:
            frame: кадр в формате BGR
            detection: детекция с координатами рамки
            color: цвет рамки в формате BGR
            label: текст подписи

        Returns:
            Кадр с нанесёнными аннотациями.
        """
        # Рисуем рамку через OpenCV
        cv2.rectangle(
            frame,
            (detection.x1, detection.y1),
            (detection.x2, detection.y2),
            color,
            thickness=2,
        )

        # Переводим в PIL для отрисовки текста с кириллицей
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # Размер текста для фона под подпись
        bbox = draw.textbbox((0, 0), label, font=self.font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        padding = 4

        # Фон под текст — того же цвета что рамка
        # PIL использует RGB, переворачиваем BGR
        rgb_color = (color[2], color[1], color[0])
        bg_x1 = detection.x1
        bg_y1 = max(0, detection.y1 - text_h - padding * 2)
        bg_x2 = detection.x1 + text_w + padding * 2
        bg_y2 = detection.y1
        draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=rgb_color)

        # Текст поверх фона
        draw.text(
            (detection.x1 + padding, bg_y1 + padding // 2),
            label,
            font=self.font,
            fill=(255, 255, 255),  # белый
        )

        # Переводим обратно в BGR для OpenCV
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def _build_label(self, result: RecognitionResult | None) -> str:
        """
        Формирует текстовую подпись для номерного знака.

        Args:
            result: результат распознавания

        Returns:
            Строка вида "А123ВС456 (87%)" или "?" если не распознан.
        """
        if result is None or not result.plate_text:
            return "?"

        confidence_str = f"{result.confidence:.0%}"

        if result.is_valid:
            return f"{result.plate_text} ({confidence_str})"

        return f"? {result.plate_text} ({confidence_str})"

    def _save_screenshot(
        self,
        frame: np.ndarray,
        frame_number: int,
    ) -> None:
        """
        Сохраняет аннотированный кадр в папку output_dir.

        Args:
            frame: аннотированный кадр
            frame_number: номер кадра (используется в имени файла)
        """
        filename = self.output_dir / f"frame_{frame_number:06d}.jpg"
        cv2.imwrite(str(filename), frame)
        self.logger.debug(f"Скриншот сохранён: {filename}")