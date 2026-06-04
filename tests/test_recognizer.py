"""
test_recognizer.py
------------------
Юнит-тесты для логики модуля recognizer.py.
Тестируется постобработка текста и валидация номера — без запуска EasyOCR.

Запуск:
    pytest tests/test_recognizer.py -v
"""

import pytest
from src.recognizer import Recognizer, RU_PLATE_PATTERN


def make_recognizer() -> Recognizer:
    """Создаёт Recognizer, минуя __init__ (без загрузки EasyOCR)."""
    return Recognizer.__new__(Recognizer)


# ---------------------------------------------------------------------------
# Тесты _postprocess()
# ---------------------------------------------------------------------------

class TestPostprocess:
    """Проверяем умную позиционную постобработку текста."""

    def setup_method(self):
        self.recognizer = make_recognizer()

    def test_correct_full_plate(self):
        """Полный номер с латиницей — правильно конвертируется."""
        assert self.recognizer._postprocess("A123BC456") == "А123ВС456"

    def test_digit_one_at_letter_position(self):
        """Цифра 1 на позиции буквы — заменяется на Т."""
        assert self.recognizer._postprocess("1505YH36") == "Т505УН36"

    def test_ruble_sign_replaced(self):
        """Знак рубля на позиции буквы — заменяется на Р."""
        assert self.recognizer._postprocess("₽986YX36") == "Р986УХ36"

    def test_letter_o_at_digit_position(self):
        """Буква О на позиции цифры — заменяется на 0."""
        assert self.recognizer._postprocess("АО23ВС45") == "А023ВС45"

    def test_empty_string(self):
        """Пустая строка остаётся пустой."""
        assert self.recognizer._postprocess("") == ""

    def test_short_string_returned_as_is(self):
        """Короткая строка — возвращается без позиционной обработки."""
        result = self.recognizer._postprocess("АВС")
        assert len(result) <= 3


# ---------------------------------------------------------------------------
# Тесты валидации по regex
# ---------------------------------------------------------------------------

class TestPlatePattern:
    """Проверяем regex-шаблон российского номера."""

    def test_valid_plate_3_digit_region(self):
        """Стандартный номер с трёхзначным регионом — валидный."""
        assert RU_PLATE_PATTERN.match("А123ВС456")

    def test_valid_plate_2_digit_region(self):
        """Стандартный номер с двузначным регионом — валидный."""
        assert RU_PLATE_PATTERN.match("А123ВС45")

    def test_invalid_wrong_structure(self):
        """Неправильная структура — не валидный."""
        assert not RU_PLATE_PATTERN.match("123АВС456")

    def test_invalid_latin_letters(self):
        """Латинские буквы вместо кириллицы — не валидный."""
        assert not RU_PLATE_PATTERN.match("A123BC456")

    def test_invalid_empty(self):
        """Пустая строка — не валидный."""
        assert not RU_PLATE_PATTERN.match("")

    def test_valid_all_allowed_letters(self):
        """Номер с буквами У и Х — валидный."""
        assert RU_PLATE_PATTERN.match("У999ХХ116")

    def test_invalid_missing_region(self):
        """Номер без кода региона — не валидный."""
        assert not RU_PLATE_PATTERN.match("А123ВС")

    def test_valid_repeated_letters(self):
        """Номер с повторяющимися буквами — валидный."""
        assert RU_PLATE_PATTERN.match("М777ММ177")