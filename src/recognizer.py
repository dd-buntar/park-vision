"""
recognizer.py
-------------
Модуль распознавания текста с номерных знаков.
Использует EasyOCR для извлечения символов и валидирует
результат по формату российского номера.
"""

import re
from dataclasses import dataclass

import cv2
import easyocr
import numpy as np

from src.detector import Detection


ALLOWED_CHARS = "АВЕКМНОРСТУХ"

RU_PLATE_PATTERN = re.compile(
    rf"^[{ALLOWED_CHARS}]{{1}}\d{{3}}[{ALLOWED_CHARS}]{{2}}\d{{2,3}}$"
)

CHAR_SUBSTITUTIONS = {
    "A": "А", "B": "В", "E": "Е", "K": "К",
    "M": "М", "H": "Н", "O": "О", "P": "Р",
    "C": "С", "T": "Т", "Y": "У", "X": "Х"
}


@dataclass
class RecognitionResult:
    """Результат распознавания одного номерного знака."""
    raw_text: str
    plate_text: str
    is_valid: bool
    confidence: float


class Recognizer:
    """
    Обёртка над EasyOCR для распознавания российских номерных знаков.

    Пример использования:
        recognizer = Recognizer()
        result = recognizer.recognize(frame, plate_detection)
        if result.is_valid:
            print(result.plate_text)
    """

    def __init__(self, gpu: bool = True) -> None:
        """
        Args:
            gpu: использовать ли GPU для распознавания.
                 Передайте False если GPU недоступен.
        """
        # EasyOCR при первом запуске скачивает языковые модели (~100MB).
        # Последующие запуски используют кэш.
        self.reader = easyocr.Reader(
            lang_list=["ru", "en"],
            gpu=gpu,
            verbose=False,
        )

    def recognize(
        self,
        frame: np.ndarray,
        plate: Detection,
    ) -> RecognitionResult | None:
        """
        Распознаёт текст номерного знака на кадре.

        Args:
            frame: полный кадр в формате BGR (numpy array)
            plate: детекция номерного знака с координатами

        Returns:
            RecognitionResult или None если вырезать номер не удалось.
        """
        plate_img = self._crop_plate(frame, plate)
        if plate_img is None:
            return None

        plate_img = self._preprocess(plate_img)
        ocr_results = self.reader.readtext(plate_img)

        if not ocr_results:
            return RecognitionResult(
                raw_text="",
                plate_text="",
                is_valid=False,
                confidence=0.0,
            )

        raw_text, confidence = self._merge_ocr_results(ocr_results)
        plate_text = self._postprocess(raw_text)
        is_valid = bool(RU_PLATE_PATTERN.match(plate_text))

        return RecognitionResult(
            raw_text=raw_text,
            plate_text=plate_text,
            is_valid=is_valid,
            confidence=confidence,
        )

    def _crop_plate(
        self,
        frame: np.ndarray,
        plate: Detection,
    ) -> np.ndarray | None:
        """
        Вырезает область номерного знака из кадра.

        Args:
            frame: полный кадр
            plate: детекция с координатами рамки

        Returns:
            Вырезанное изображение или None если координаты некорректны.
        """
        h, w = frame.shape[:2]

        x1 = max(0, plate.x1)
        y1 = max(0, plate.y1)
        x2 = min(w, plate.x2)
        y2 = min(h, plate.y2)

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2]

    def _preprocess(self, plate_img: np.ndarray) -> np.ndarray:
        """
        Подготавливает изображение номера перед подачей в OCR.
        Увеличивает, переводит в серый и улучшает контраст.

        Args:
            plate_img: вырезанное изображение номерного знака

        Returns:
            Обработанное изображение.
        """
        plate_img = cv2.resize(
            plate_img,
            None,
            fx=2.0,
            fy=2.0,
            interpolation=cv2.INTER_CUBIC,
        )

        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        return enhanced

    def _merge_ocr_results(
        self,
        ocr_results: list,
    ) -> tuple[str, float]:
        """
        Объединяет несколько текстовых блоков от EasyOCR в одну строку.
        EasyOCR может разбить номер на несколько частей.

        Args:
            ocr_results: список результатов от reader.readtext()

        Returns:
            Кортеж (объединённый текст, средняя уверенность).
        """
        texts = []
        confidences = []

        for _, text, confidence in ocr_results:
            texts.append(text.strip())
            confidences.append(confidence)

        merged = "".join(texts)
        avg_confidence = sum(confidences) / len(confidences)

        return merged, avg_confidence

    def _postprocess(self, text: str) -> str:
        """
        Очищает и нормализует распознанный текст.

        Шаги:
        1. Убираем пробелы, дефисы и прочий мусор
        2. Приводим к верхнему регистру
        3. Заменяем визуально похожие символы на кириллицу

        Args:
            text: сырой текст от EasyOCR

        Returns:
            Нормализованная строка.
        """
        text = re.sub(r"[^А-ЯA-Z0-9а-яa-z]", "", text)
        text = text.upper()

        result = ""
        for char in text:
            result += CHAR_SUBSTITUTIONS.get(char, char)

        return result