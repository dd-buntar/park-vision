"""
test_detector.py
----------------
Юнит-тесты для логики модуля detector.py.
Тестируется только геометрическая логика — без загрузки модели и весов.

Запуск:
    pytest tests/test_detector.py -v
"""

from src.detector import Detection, Detector


def make_detection(x1: int, y1: int, x2: int, y2: int, class_id: int = 0) -> Detection:
    """Вспомогательная фабрика — создаёт Detection с тестовыми данными."""
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=0.9, class_id=class_id)


class TestDetectionContains:
    """Проверяем метод contains() — входит ли одна рамка в другую."""

    def test_plate_fully_inside_car(self):
        """Номер полностью внутри машины — True."""
        car = make_detection(0, 0, 200, 150)
        plate = make_detection(50, 100, 150, 140)
        assert car.contains(plate) is True

    def test_plate_outside_car(self):
        """Номер полностью за пределами машины — False."""
        car = make_detection(0, 0, 100, 100)
        plate = make_detection(150, 150, 250, 200)
        assert car.contains(plate) is False

    def test_plate_partially_overlaps_car(self):
        """Номер частично выходит за рамку машины — False."""
        car = make_detection(0, 0, 100, 100)
        plate = make_detection(80, 80, 180, 120)
        assert car.contains(plate) is False


class TestDetectionDimensions:
    """Проверяем вычисляемые свойства width и height."""

    def test_width_and_height(self):
        d = make_detection(10, 20, 110, 80)
        assert d.width == 100
        assert d.height == 60


class TestFilterPlates:
    """
    Тестируем логику фильтрации номеров.
    Используем Detector.__new__ чтобы не загружать модель из файла.
    """

    def setup_method(self):
        """Создаём экземпляр Detector, минуя __init__ (без загрузки модели)."""
        self.detector = Detector.__new__(Detector)

    def test_plate_inside_car_is_kept(self):
        """Номер внутри машины — остаётся."""
        car = make_detection(0, 0, 300, 200, class_id=0)
        plate = make_detection(50, 150, 200, 190, class_id=1)
        result = self.detector._filter_plates([car], [plate])
        assert result == [plate]

    def test_plate_outside_car_is_removed(self):
        """Номер вне машины — отбрасывается."""
        car = make_detection(0, 0, 100, 100, class_id=0)
        plate = make_detection(200, 200, 300, 250, class_id=1)
        result = self.detector._filter_plates([car], [plate])
        assert result == []

    def test_no_cars_returns_all_plates(self):
        """Если машин нет — возвращаются все номера (режим тестирования)."""
        plate1 = make_detection(0, 0, 100, 50, class_id=1)
        plate2 = make_detection(200, 0, 300, 50, class_id=1)
        result = self.detector._filter_plates([], [plate1, plate2])
        assert len(result) == 2

    def test_stray_plate_among_valid_ones(self):
        """Один номер внутри машины, один посторонний — остаётся только валидный."""
        car = make_detection(0, 0, 200, 150, class_id=0)
        valid_plate = make_detection(30, 100, 150, 140, class_id=1)
        stray_plate = make_detection(500, 500, 600, 550, class_id=1)
        result = self.detector._filter_plates([car], [valid_plate, stray_plate])
        assert result == [valid_plate]