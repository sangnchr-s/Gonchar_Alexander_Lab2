# Лабораторная работа 2. Модель цены аренды квартиры

Вариант 05. Целевая переменная `rent_rub` – месячная арендная плата в рублях.

## Окружение

Из корня репозитория создайте окружение Conda с Python 3.12.16 и запустите JupyterLab.

```bash
conda env create -f environment.yml
conda activate lab2_gonchar
python -m jupyterlab
```

При запуске JupyterLab из другого окружения зарегистрируйте ядро из активного `lab2_gonchar` и выберите **Python (lab2_gonchar)** в меню Jupyter.

```bash
python -m ipykernel install --user --name lab2_gonchar --display-name "Python (lab2_gonchar)"
```

## Данные и файлы проекта

Перед подготовкой данных создайте `data/raw/` и положите туда выданные `rent_v05.csv` и `data_description_v05.md`. Все файлы в `data/` хранятся локально.

В JupyterLab начните с `notebooks/01_preparation.ipynb` и выполняйте ячейки сверху вниз.

- `task/TASK.md` – условие лабораторной.
- `environment.yml` – описание окружения Conda с закреплёнными версиями пакетов.
- `notebooks/01_preparation.ipynb` – ноутбук подготовки данных. Первая ячейка кода настраивает пути проекта.
- `data/interim/` – каталог для тестовой выборки в исходном формате.
- `data/processed/` – каталог для подготовленных обучающей и валидационной выборок.
- `modules/` – общие модули Python. Файл `modules/__init__.py` обозначает пакет `modules`, из которого ноутбуки и скрипты смогут импортировать общий код.
- `models/` – каталог для сохранённых пайплайнов и модели в формате `.joblib`.
- `inference.py` – скрипт в корне репозитория, который добавим на этапе 3. Он получит путь к сырому тесту через `--test_path`, загрузит сохранённую модель и запишет рядом с тестом прогнозы и метрики R², MAE и RMSE.
