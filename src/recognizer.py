"""
recognizer.py
-------------
Модуль распознавания текста с номерных знаков.
Использует LPRNet — специализированную нейросеть для номерных знаков.
Значительно точнее EasyOCR для этой задачи.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from src.detector import Detection
from src.lprnet.lprnet_model import LPRNet
from src.lprnet.stn_model import SpatialTransformer
from src.lprnet.decoder import GreedyDecoder


# Символы которые знает модель (латиница + цифры)
# Последний символ '-' — blank для CTC декодера
CHARS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'E', 'K', 'M', 'H', 'O', 'P', 'C', 'T',
    'Y', 'X', '-'
]

# Параметры LPRNet
LPR_MAX_LEN = 9
OUT_INDICES = (2, 6, 13, 22)
DROPOUT = 0.0

# Входной размер изображения для LPRNet
INPUT_SIZE = (94, 24)  # ширина x высота

# Замена латиницы на кириллицу для российских номеров
LATIN_TO_CYRILLIC = {
    'A': 'А', 'B': 'В', 'E': 'Е', 'K': 'К',
    'M': 'М', 'H': 'Н', 'O': 'О', 'P': 'Р',
    'C': 'С', 'T': 'Т', 'Y': 'У', 'X': 'Х',
}

# Разрешённые кириллические буквы в российских номерах
ALLOWED_CHARS = "АВЕКМНОРСТУХ"

RU_PLATE_PATTERN = re.compile(
    rf"^[{ALLOWED_CHARS}]{{1}}\d{{3}}[{ALLOWED_CHARS}]{{2}}\d{{2,3}}$"
)


@dataclass
class RecognitionResult:
    """Результат распознавания одного номерного знака."""
    raw_text: str        # текст как вернул LPRNet (латиница)
    plate_text: str      # текст после конвертации в кириллицу
    is_valid: bool       # соответствует ли формату российского номера
    confidence: float    # уверенность (0.0 — 1.0)


class Recognizer:
    """
    Обёртка над LPRNet для распознавания российских номерных знаков.

    Пример использования:
        recognizer = Recognizer(
            lprnet_weights="models/LPRNet_Ep_BEST_model.ckpt",
            stn_weights="models/SpatialTransformer_Ep_BEST_model.ckpt",
        )
        result = recognizer.recognize(frame, plate_detection)
        if result and result.is_valid:
            print(result.plate_text)
    """

    def __init__(
        self,
        lprnet_weights: str | Path = "models/LPRNet_Ep_BEST_model.ckpt",
        stn_weights: str | Path = "models/SpatialTransformer_Ep_BEST_model.ckpt",
        gpu: bool = True,
    ) -> None:
        """
        Args:
            lprnet_weights: путь к весам LPRNet
            stn_weights: путь к весам SpatialTransformer
            gpu: использовать ли GPU
        """
        self.device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")
        self.decoder = GreedyDecoder()

        # Загружаем SpatialTransformer
        stn_weights = Path(stn_weights)
        if not stn_weights.exists():
            raise FileNotFoundError(
                f"Веса STN не найдены: {stn_weights}\n"
                "Скачайте файл SpatialTransformer_Ep_BEST_model.ckpt в папку models/."
            )
        self.stn = SpatialTransformer()
        self._load_weights(self.stn, stn_weights)
        self.stn.to(self.device)
        self.stn.eval()

        # Загружаем LPRNet
        lprnet_weights = Path(lprnet_weights)
        if not lprnet_weights.exists():
            raise FileNotFoundError(
                f"Веса LPRNet не найдены: {lprnet_weights}\n"
                "Скачайте файл LPRNet_Ep_BEST_model.ckpt в папку models/."
            )
        self.lprnet = LPRNet(
            class_num=len(CHARS),
            dropout_prob=DROPOUT,
            out_indices=OUT_INDICES,
        )
        self._load_weights(self.lprnet, lprnet_weights)
        self.lprnet.to(self.device)
        self.lprnet.eval()

    def _load_weights(self, model: torch.nn.Module, weights_path: Path) -> None:
        """Загружает веса модели из файла .ckpt."""
        checkpoint = torch.load(str(weights_path), map_location=self.device)
        # Веса могут быть сохранены напрямую или в обёртке с ключом net_state_dict
        if isinstance(checkpoint, dict) and "net_state_dict" in checkpoint:
            state_dict = checkpoint["net_state_dict"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)

    def recognize(
        self,
        frame: np.ndarray,
        plate: Detection,
    ) -> RecognitionResult | None:
        """
        Распознаёт текст номерного знака на кадре.

        Args:
            frame: полный кадр в формате BGR
            plate: детекция номерного знака с координатами

        Returns:
            RecognitionResult или None если вырезать номер не удалось.
        """
        plate_img = self._crop_plate(frame, plate)
        if plate_img is None:
            return None

        tensor = self._preprocess(plate_img)

        with torch.no_grad():
            # Пропускаем через STN для выравнивания перспективы
            tensor = self.stn(tensor)
            # Распознаём символы
            output = self.lprnet(tensor)

        preds = output.cpu().detach().numpy()
        labels, _ = self.decoder.decode(preds, CHARS)

        if not labels:
            return None

        raw_text = labels[0]
        plate_text = self._to_cyrillic(raw_text)
        is_valid = bool(RU_PLATE_PATTERN.match(plate_text))
        confidence = self._estimate_confidence(preds[0])

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
        """Вырезает область номерного знака из кадра."""
        h, w = frame.shape[:2]
        x1 = max(0, plate.x1)
        y1 = max(0, plate.y1)
        x2 = min(w, plate.x2)
        y2 = min(h, plate.y2)

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2]

    def _preprocess(self, plate_img: np.ndarray) -> torch.Tensor:
        """
        Подготавливает изображение номера для LPRNet.
        Масштабирует до 94x24, нормализует и переводит в тензор.
        """
        img = cv2.resize(plate_img, INPUT_SIZE)
        img = img.astype(np.float32)
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))
        tensor = torch.from_numpy(img).unsqueeze(0)
        return tensor.to(self.device)

    def _to_cyrillic(self, text: str) -> str:
        """
        Конвертирует латинские буквы в кириллицу.
        LPRNet возвращает латиницу, нам нужна кириллица.
        """
        return "".join(LATIN_TO_CYRILLIC.get(c, c) for c in text)

    def _estimate_confidence(self, preds: np.ndarray) -> float:
        """
        Оценивает уверенность распознавания.
        Берём среднее максимальных вероятностей по всем позициям.

        Args:
            preds: выход LPRNet shape [len(CHARS), seq_len]

        Returns:
            Уверенность от 0.0 до 1.0
        """
        exp_preds = np.exp(preds - np.max(preds, axis=0, keepdims=True))
        probs = exp_preds / exp_preds.sum(axis=0, keepdims=True)
        max_probs = np.max(probs, axis=0)
        return float(np.mean(max_probs))