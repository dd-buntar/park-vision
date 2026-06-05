"""
test_processor.py
-----------------
Тесты для модуля processor.py.
Используем unittest.mock чтобы не загружать реальные модели.

Запуск:
    pytest tests/test_processor.py -v
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.detector import Detection, FrameDetections
from src.processor import Processor
from src.recognizer import RecognitionResult


# ---------------------------------------------------------------------------
# Фабрики тестовых объектов
# ---------------------------------------------------------------------------

def make_detection(x1: int, y1: int, x2: int, y2: int, class_id: int = 0) -> Detection:
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=0.9, class_id=class_id)


def make_processor(tmp_path: Path) -> Processor:
    """
    Создаёт Processor с фиктивными детектором и распознавателем.

    tmp_path — это встроенная фикстура pytest: временная папка,
    которая создаётся для каждого теста и удаляется после.
    """
    fake_detector = MagicMock()
    fake_recognizer = MagicMock()
    csv_path = tmp_path / "plates.csv"
    output_dir = tmp_path / "output"

    return Processor(
        detector=fake_detector,
        recognizer=fake_recognizer,
        csv_path=csv_path,
        output_dir=output_dir,
        save_all=True,
    )


# ---------------------------------------------------------------------------
# Тесты _build_label()
# ---------------------------------------------------------------------------

class TestBuildLabel:
    """Проверяем формирование текстовой подписи над номером."""

    def setup_method(self):
        self.processor = Processor.__new__(Processor)

    def test_valid_plate_shows_text_and_confidence(self):
        """Валидный номер — показывает текст и уверенность."""
        result = RecognitionResult(
            raw_text="A123BC456",
            plate_text="А123ВС456",
            is_valid=True,
            confidence=0.87,
        )
        label = self.processor._build_label(result)
        assert "А123ВС456" in label
        assert "87%" in label

    def test_invalid_plate_shows_question_mark(self):
        """Невалидный номер — показывает знак вопроса."""
        result = RecognitionResult(
            raw_text="XYZ",
            plate_text="ХУZ",
            is_valid=False,
            confidence=0.4,
        )
        label = self.processor._build_label(result)
        assert label.startswith("?")

    def test_none_result_returns_question_mark(self):
        """Если распознавание вернуло None — показывает знак вопроса."""
        assert self.processor._build_label(None) == "?"

    def test_empty_plate_text_returns_question_mark(self):
        """Пустой текст номера — показывает знак вопроса."""
        result = RecognitionResult(
            raw_text="",
            plate_text="",
            is_valid=False,
            confidence=0.0,
        )
        assert self.processor._build_label(result) == "?"


# ---------------------------------------------------------------------------
# Тесты _save_screenshot()
# ---------------------------------------------------------------------------

class TestSaveScreenshot:
    """Проверяем сохранение скриншотов."""

    def test_screenshot_file_created(self, tmp_path: Path):
        """После вызова файл должен появиться в output_dir."""
        processor = make_processor(tmp_path)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)  # пустой чёрный кадр

        processor._save_screenshot(frame, frame_number=42)

        expected_file = processor.output_dir / "frame_000042.jpg"
        assert expected_file.exists()

    def test_screenshot_filename_has_leading_zeros(self, tmp_path: Path):
        """Имя файла должно содержать ведущие нули для правильной сортировки."""
        processor = make_processor(tmp_path)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        processor._save_screenshot(frame, frame_number=1)

        expected_file = processor.output_dir / "frame_000001.jpg"
        assert expected_file.exists()


# ---------------------------------------------------------------------------
# Тесты _annotate_frame() — здесь используем моки полноценно
# ---------------------------------------------------------------------------

class TestAnnotateFrame:
    """
    Проверяем логику аннотирования кадра.
    Детектор и распознаватель заменены моками.
    """

    def test_valid_plate_written_to_csv(self, tmp_path: Path):
        """
        Если номер валидный и уверенность высокая — он должен попасть в CSV.
        Используем patch чтобы перехватить вызов write_plate.
        """
        processor = make_processor(tmp_path)

        # Настраиваем мок распознавателя — он вернёт валидный номер
        processor.recognizer.recognize.return_value = RecognitionResult(
            raw_text="A123BC456",
            plate_text="А123ВС456",
            is_valid=True,
            confidence=0.9,
        )

        plate = make_detection(50, 100, 200, 140, class_id=1)
        detections = FrameDetections(cars=[], plates=[plate])
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # patch перехватывает write_plate и считает сколько раз она вызвана
        with patch("src.processor.write_plate") as mock_write:
            processor._annotate_frame(frame, detections, source="test.mp4", frame_number=0)
            mock_write.assert_called_once()

    def test_invalid_plate_not_written_to_csv(self, tmp_path: Path):
        """Невалидный номер — не должен попасть в CSV."""
        processor = make_processor(tmp_path)

        processor.recognizer.recognize.return_value = RecognitionResult(
            raw_text="XYZ",
            plate_text="ХУZ",
            is_valid=False,
            confidence=0.9,
        )

        plate = make_detection(50, 100, 200, 140, class_id=1)
        detections = FrameDetections(cars=[], plates=[plate])
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("src.processor.write_plate") as mock_write:
            processor._annotate_frame(frame, detections, source="test.mp4", frame_number=0)
            mock_write.assert_not_called()

    def test_low_confidence_plate_not_written_to_csv(self, tmp_path: Path):
        """Валидный номер но низкая уверенность — не должен попасть в CSV."""
        processor = make_processor(tmp_path)

        processor.recognizer.recognize.return_value = RecognitionResult(
            raw_text="A123BC456",
            plate_text="А123ВС456",
            is_valid=True,
            confidence=0.3,  # ниже порога MIN_CONFIDENCE_TO_LOG = 0.5
        )

        plate = make_detection(50, 100, 200, 140, class_id=1)
        detections = FrameDetections(cars=[], plates=[plate])
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("src.processor.write_plate") as mock_write:
            processor._annotate_frame(frame, detections, source="test.mp4", frame_number=0)
            mock_write.assert_not_called()