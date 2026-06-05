"""
test_recognizer.py
------------------
Юнит-тесты для модуля recognizer.py.
Тестируется конвертация латиницы в кириллицу и валидация номера.

Запуск:
    pytest tests/test_recognizer.py -v
"""

import pytest
from src.recognizer import Recognizer, RU_PLATE_PATTERN


def make_recognizer() -> Recognizer:
    """Создаёт Recognizer, минуя __init__ (без загрузки моделей)."""
    return Recognizer.__new__(Recognizer)


# ---------------------------------------------------------------------------
# Тесты _to_cyrillic()
# ---------------------------------------------------------------------------

class TestToCyrillic:
    """Проверяем конвертацию латиницы в кириллицу."""

    def setup_method(self):
        self.recognizer = make_recognizer()

    def test_full_latin_plate(self):
        """Полный номер с латиницей — правильно конвертируется."""
        assert self.recognizer._to_cyrillic("A123BC") == "А123ВС"

    def test_digits_unchanged(self):
        """Цифры не меняются."""
        assert self.recognizer._to_cyrillic("123") == "123"

    def test_mixed_latin_and_digits(self):
        """Латиница конвертируется, цифры остаются."""
        assert self.recognizer._to_cyrillic("X888XX01") == "Х888ХХ01"

    def test_empty_string(self):
        """Пустая строка остаётся пустой."""
        assert self.recognizer._to_cyrillic("") == ""

    def test_all_allowed_letters(self):
        """Все разрешённые латинские буквы конвертируются в кириллицу."""
        assert self.recognizer._to_cyrillic("ABEKMHOPCTYX") == "АВЕКМНОРСТУХ"


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