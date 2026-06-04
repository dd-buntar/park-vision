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


# Разрешённые кириллические буквы в российских номерах
# (только те, у которых есть графический аналог в латинице)
ALLOWED_CHARS = "АВЕКМНОРСТУХ"

# Regex-шаблон российского номера: А123ВС456 или А123ВС45
RU_PLATE_PATTERN = re.compile(
    rf"^[{ALLOWED_CHARS}]{{1}}\d{{3}}[{ALLOWED_CHARS}]{{2}}\d{{2,3}}$"
)

# Замены для позиций где должна быть БУКВА.
# Цифры и латиница → кириллица.
LETTER_SUBSTITUTIONS = {
    # Латиница → кириллица
    "A": "А", "B": "В", "E": "Е", "K": "К",
    "M": "М", "H": "Н", "O": "О", "P": "Р",
    "C": "С", "T": "Т", "Y": "У", "X": "Х",
    # Цифры → похожие буквы
    "0": "О", "1": "Т", "3": "З", "4": "А",
    "6": "Б", "8": "В",
    # Знак рубля → Р
    "₽": "Р",
}

# Замены для позиций где должна быть ЦИФРА.
# Буквы → похожие цифры.
DIGIT_SUBSTITUTIONS = {
    # Кириллица → цифры
    "О": "0", "З": "3", "А": "4", "В": "8",
    # Латиница → цифры
    "O": "0", "Z": "3", "B": "8", "S": "5",
    "I": "1", "L": "1", "G": "6", "T": "7",
}


@dataclass
class RecognitionResult:
    """Результат распознавания одного номерного знака."""
    raw_text: str        # текст как вернул EasyOCR, до обработки
    plate_text: str      # текст после очистки и замен
    is_valid: bool       # соответствует ли формату российского номера
    confidence: float    # средняя уверенность EasyOCR (0.0 — 1.0)


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
        # Увеличиваем — маленькие номера OCR читает хуже
        plate_img = cv2.resize(
            plate_img,
            None,
            fx=2.0,
            fy=2.0,
            interpolation=cv2.INTER_CUBIC,
        )

        # Переводим в оттенки серого
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        # CLAHE — адаптивное выравнивание гистограммы.
        # Улучшает читаемость при неравномерном освещении.
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

        Применяет умную позиционную постобработку — разные таблицы замен
        для букв и цифр в зависимости от позиции в номере.

        Структура российского номера: Б 999 ББ 999
        Позиции: 0=буква, 1-3=цифры, 4-5=буквы, 6-8=цифры региона

        Args:
            text: сырой текст от EasyOCR

        Returns:
            Нормализованная строка.
        """
        # Убираем всё кроме букв, цифр и знака рубля (₽ путают с Р)
        text = re.sub(r"[^А-ЯA-Z0-9а-яa-zЁё₽]", "", text)

        # Приводим к верхнему регистру
        text = text.upper()

        # Если строка слишком короткая — возвращаем как есть
        if len(text) < 6:
            return text

        # Позиции букв и цифр в российском номере
        # Б 9 9 9 Б Б 9 9 [9]
        # 0 1 2 3 4 5 6 7 [8]
        LETTER_POSITIONS = {0, 4, 5}
        DIGIT_POSITIONS  = {1, 2, 3, 6, 7, 8}

        result = ""
        for i, char in enumerate(text):
            if i in LETTER_POSITIONS:
                result += LETTER_SUBSTITUTIONS.get(char, char)
            elif i in DIGIT_POSITIONS:
                result += DIGIT_SUBSTITUTIONS.get(char, char)
            else:
                result += char

        return result