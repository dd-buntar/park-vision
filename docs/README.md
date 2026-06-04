# Обучение модели

Модель YOLOv8n дообучена на датасете российских номерных знаков
[Russian license plates](https://universe.roboflow.com/testcarplate/russian-license-plates-classification-by-this-type)
с Roboflow Universe.

## Процесс обучения
- База: yolov8n.pt
- Эпох: 30
- Размер изображения: 640×640
- Датасет: российские номерные знаки

## Файлы
- `training.ipynb` — ноутбук для воспроизведения обучения в Google Colab
- `graphs.png` — графики loss и метрик по эпохам
- `cars.png` — примеры предсказаний на валидационной выборке

## Запуск обучения

### Шаг 1 — Регистрация на Roboflow
1. Перейдите на [roboflow.com](https://roboflow.com) и создайте бесплатный аккаунт
2. После входа перейдите в настройки аккаунта:
   `https://app.roboflow.com/settings/api`
3. Скопируйте **Private API Key**

### Шаг 2 — Добавление ключа в Google Colab
1. Откройте `training.ipynb` в [Google Colab](https://colab.research.google.com)
2. В левой панели нажми иконку 🔑 **Secrets**
3. Нажмите **"Add new secret"**
4. В поле **Name** введите: `ROBOFLOW_API_KEY`
5. В поле **Value** вставьте скопированный ключ
6. Включите переключатель **"Notebook access"**

### Шаг 3 — Запуск обучения
1. В меню Colab выберите **Runtime → Change runtime type → T4 GPU**
2. Запускайте ячейки по порядку сверху вниз
3. Обучение занимает около 20–40 минут
4. После завершения ячейка 9 автоматически скачает файл `best.pt`

Скачанный файл `best.pt` находится в папке `models/` проекта ParkVision.