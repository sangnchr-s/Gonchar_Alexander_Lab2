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

В JupyterLab сначала выполните `notebooks/01_preparation.ipynb`, затем `notebooks/02_modeling.ipynb`. В каждом ноутбуке запускайте ячейки сверху вниз.

- `task/TASK.md` – условие лабораторной.
- `environment.yml` – описание окружения Conda с закреплёнными версиями пакетов.
- `notebooks/01_preparation.ipynb` – загрузка и разбиение выгрузки, аудит train, обучение пайплайна на train и преобразование validation.
- `notebooks/02_modeling.ipynb` – обучение четырёх регрессоров и сравнение MAE, RMSE и R² на validation. Критерий выбора модели – минимальная MAE в рублях.
- `data/interim/rent_v05_test.csv` – сырой тест в исходном формате. Создаётся при выполнении ноутбука подготовки.
- `data/processed/train.csv` и `data/processed/validation.csv` – подготовленные признаки и целевой столбец `rent_rub` в рублях. При обучении модели отделите этот столбец от признаков.
- `modules/__init__.py` – общий для разбиения и обучения моделей `RANDOM_STATE = 42`.
- `modules/preprocessing.py` – правила обработки сырых признаков, включая названия районов, планировку студий, возраст дома и проверку площадей.
- `models/preprocessor.joblib` – обученный пайплайн предобработки. Метод `transform` принимает исходные признаки объявления и возвращает числовые признаки для модели.
- `inference.py` – скрипт в корне репозитория, который добавим на этапе 3. Он получит путь к сырому тесту через `--test_path`, загрузит сохранённую модель и запишет рядом с тестом прогнозы и метрики R², MAE и RMSE.
