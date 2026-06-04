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
    """Проверяем очистку и нормализацию текста."""

    def setup_method(self):
        self.recognizer = make_recognizer()

    def test_latin_to_cyrillic(self):
        """Латинские буквы заменяются на кириллические аналоги."""
        assert self.recognizer._postprocess("A123BC") == "А123ВС"

    def test_removes_spaces_and_dashes(self):
        """Пробелы и дефисы убираются."""
        assert self.recognizer._postprocess("А 123 ВС-456") == "А123ВС456"

    def test_lowercase_to_uppercase(self):
        """Строчные буквы приводятся к верхнему регистру."""
        assert self.recognizer._postprocess("а123вс456") == "А123ВС456"

    def test_mixed_latin_and_cyrillic(self):
        """Смесь латиницы и кириллицы — латиница заменяется."""
        assert self.recognizer._postprocess("A123ВС456") == "А123ВС456"

    def test_empty_string(self):
        """Пустая строка остаётся пустой."""
        assert self.recognizer._postprocess("") == ""

    def test_y_to_cyrillic_u(self):
        """Латинская Y заменяется на кириллическую У."""
        assert self.recognizer._postprocess("Y123ВС456") == "У123ВС456"

    def test_garbage_characters_removed(self):
        """Мусорные символы убираются."""
        assert self.recognizer._postprocess("А!123#ВС@456") == "А123ВС456"


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