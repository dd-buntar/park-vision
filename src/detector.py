"""
detector.py
-----------
Модуль детекции транспортных средств и номерных знаков.
Использует предобученную модель YOLOv8.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO


# Индексы классов в модели
#TODO Уточнить классы в документации весов модели
CLASS_CAR = 0
CLASS_PLATE = 1

# Минимальная уверенность модели для принятия детекции
DEFAULT_CONFIDENCE = 0.45


@dataclass
class Detection:
    """Результат одной детекции — рамка и метаданные."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def contains(self, other: "Detection") -> bool:
        """Проверяет, находится ли другая рамка внутри этой."""
        return (
            self.x1 <= other.x1
            and self.y1 <= other.y1
            and self.x2 >= other.x2
            and self.y2 >= other.y2
        )


@dataclass
class FrameDetections:
    """Все детекции на одном кадре."""
    cars: list[Detection]
    plates: list[Detection]


class Detector:
    """
    Обёртка над YOLOv8 для детекции машин и номерных знаков.

    Пример использования:
        detector = Detector("models/best.pt")
        result = detector.detect(frame)
        for plate in result.plates:
            print(plate.x1, plate.y1, plate.x2, plate.y2)
    """

    def __init__(
        self,
        model_path: str | Path,
        confidence: float = DEFAULT_CONFIDENCE,
        device: str = "cuda",
    ) -> None:
        """
        Args:
            model_path: путь к файлу весов (.pt)
            confidence: порог уверенности (0.0 — 1.0)
            device: 'cuda' для GPU, 'cpu' для процессора
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Файл весов не найден: {model_path}\n"
                "Скачайте веса и положите в папку models/. "
                "Инструкция в README.md."
            )

        self.confidence = confidence
        self.device = device
        self.model = YOLO(str(model_path))
        self.model.to(device)

    def detect(self, frame: np.ndarray) -> FrameDetections:
        """
        Запускает детекцию на одном кадре.

        Args:
            frame: кадр в формате BGR (numpy array, как из OpenCV)

        Returns:
            FrameDetections с отфильтрованными машинами и номерами.
            Номера без машины отбрасываются.
        """
        results = self.model(frame, conf=self.confidence, verbose=False)[0]

        cars: list[Detection] = []
        plates: list[Detection] = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])

            detection = Detection(x1, y1, x2, y2, confidence, class_id)

            if class_id == CLASS_CAR:
                cars.append(detection)
            elif class_id == CLASS_PLATE:
                plates.append(detection)

        # Оставляем только номера, которые находятся внутри рамки машины.
        # Это убирает ложные срабатывания на вывески, таблички и т.д.
        valid_plates = self._filter_plates(cars, plates)

        return FrameDetections(cars=cars, plates=valid_plates)

    def _filter_plates(
        self,
        cars: list[Detection],
        plates: list[Detection],
    ) -> list[Detection]:
        """
        Отбрасывает номера, которые не принадлежат ни одной машине.

        Args:
            cars: список детекций машин
            plates: список детекций номерных знаков

        Returns:
            Отфильтрованный список номерных знаков.
        """
        if not cars:
            # Если машин не найдено — возвращаем все номера как есть.
            # Это полезно при тестировании на фото одного номера крупным планом.
            return plates

        valid = []
        for plate in plates:
            for car in cars:
                if car.contains(plate):
                    valid.append(plate)
                    break

        return valid