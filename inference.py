"""Оценка сохранённой модели на сырой тестовой выборке."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


MODEL_PATH = Path(__file__).resolve().parent / "models" / "rent_model.joblib"
TARGET = "rent_rub"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test_path", type=Path, required=True, help="Путь к сырому тестовому CSV")
    args = parser.parse_args()
    test_path = args.test_path
    test = pd.read_csv(test_path, dtype=str, keep_default_na=False, encoding="utf-8")

    if TARGET not in test:
        parser.error(f"В тесте требуется столбец {TARGET}.")
    if "predict" in test:
        parser.error("В сыром тесте уже есть столбец predict. Укажите исходный файл.")
    actual = pd.to_numeric(test[TARGET], errors="coerce")
    if len(test) < 2 or not np.isfinite(actual).all():
        parser.error(f"Для оценки нужны минимум две строки с числовым {TARGET} в каждой.")

    model = joblib.load(MODEL_PATH)
    prediction = model.predict(test.drop(columns=TARGET))
    r2 = r2_score(actual, prediction)
    mae = mean_absolute_error(actual, prediction)
    rmse = root_mean_squared_error(actual, prediction)
    metrics = f"R2={r2:.8f}|MAE={mae:.8f}|RMSE={rmse:.8f}"

    predicts_path = test_path.with_name(f"{test_path.stem}_predicts.csv")
    metrics_path = test_path.with_name(f"{test_path.stem}_metrics.txt")
    test.assign(predict=prediction).to_csv(predicts_path, index=False, encoding="utf-8")
    metrics_path.write_text(metrics + "\n", encoding="utf-8")
    print(metrics)


if __name__ == "__main__":
    main()
