"""
processor.py
------------
Основной модуль обработки видеопотока.
Читает кадры из файла или RTSP-потока, запускает детекцию и распознавание,
рисует результаты на кадре, сохраняет скриншоты и пишет CSV-лог.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

from src.detector import Detector, FrameDetections, Detection
from src.recognizer import Recognizer, RecognitionResult
from src.utils import write_plate

COLOR_CAR = (0, 0, 255)
COLOR_PLATE = (0, 255, 0)
COLOR_TEXT = (255, 255, 255)

MIN_CONFIDENCE_TO_LOG = 0.5
DETECTION_EVERY_N_FRAMES = 3


class Processor:
    """
    Обрабатывает видеопоток: детектирует машины и номера,
    распознаёт текст, визуализирует и логирует результаты.

    Пример использования:
        processor = Processor(detector, recognizer, csv_path, output_dir)

        # Видеофайл:
        processor.process("video.mp4")

        # RTSP-поток:
        processor.process("rtsp://192.168.1.1:554/stream")
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
                    pass  # headless-окружение, окно не поддерживается

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
            source: источник видео (для записи в CSV)

        Returns:
            Кадр с нанесёнными аннотациями.
        """
        annotated = frame.copy()

        for car in detections.cars:
            self._draw_box(annotated, car, COLOR_CAR, "car")

        for plate in detections.plates:
            result = self.recognizer.recognize(frame, plate)
            label = self._build_label(result)
            self._draw_box(annotated, plate, COLOR_PLATE, label)

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
    ) -> None:
        """
        Рисует прямоугольник и подпись на кадре.

        Args:
            frame: кадр (изменяется на месте)
            detection: детекция с координатами рамки
            color: цвет рамки в формате BGR
            label: текст подписи
        """
        cv2.rectangle(
            frame,
            (detection.x1, detection.y1),
            (detection.x2, detection.y2),
            color,
            thickness=2,
        )

        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            frame,
            (detection.x1, detection.y1 - text_h - 8),
            (detection.x1 + text_w + 4, detection.y1),
            color,
            thickness=-1,  # -1 означает заливку
        )

        cv2.putText(
            frame,
            label,
            (detection.x1 + 2, detection.y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLOR_TEXT,
            thickness=2,
        )

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