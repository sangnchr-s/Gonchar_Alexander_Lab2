"""Подготовка признаков для объявлений об аренде, вариант 05."""

import numpy as np
import pandas as pd


MEASUREMENTS = [
    "year_built", "floors_total", "floor", "total_area", "living_area",
    "kitchen_area", "ceiling_height", "balcony", "metro_distance_m",
    "distance_to_center_km", "description_length", "photos_count", "max_tenants",
    "elevator", "appliances", "pets_allowed",
]
CATEGORIES = ["district", "house_type", "renovation", "furniture", "parking", "seller_type"]
AREA_RATIO_LIMIT = 4.0
METRO_DISTANCE_LIMIT_M = 15_000
DISTRICT_ALIASES = {
    "cоветский": "советский", "cеверный": "северный",
    "ж/д": "железнодорожный", "ж/д район": "железнодорожный",
    "пром.": "промышленный", "центр.": "центральный",
    "ленинскийй": "ленинский", "кировскии": "кировский",
    "перво-майский": "первомайский",
}


def prepare_features(raw, *, correct_total_area=False, correct_metro_distance=False):
    """Преобразовать одно или несколько сырых объявлений по одинаковым правилам."""
    features = raw[MEASUREMENTS].apply(pd.to_numeric, errors="coerce").astype(float)
    for column in CATEGORIES:
        values = raw[column].astype("str").str.strip().str.lower()
        features[column] = values.replace("", np.nan)
    features["district"] = features["district"].str.removesuffix(" р-н").replace(DISTRICT_ALIASES)

    rooms = raw["rooms"].astype("str").str.strip().str.lower()
    studio = rooms.eq("студия")
    features["is_studio"] = studio.astype(int)
    features["rooms_count"] = pd.to_numeric(rooms.mask(studio, "0"), errors="coerce")
    features.loc[studio & features["kitchen_area"].isna(), "kitchen_area"] = 0

    date = pd.to_datetime(raw["publication_date"], format="%d.%m.%Y %H:%M", errors="coerce")
    age = date.dt.year - features.pop("year_built")
    features["building_age"] = age.where(age >= 0)
    features.loc[features["floor"] > features["floors_total"], "floor"] = np.nan

    total = features["total_area"].where(features["total_area"] > 0)
    living = features["living_area"]
    if correct_total_area:
        area_ratio = total / living.where(living > 0)
        total = total.mask(area_ratio.gt(AREA_RATIO_LIMIT), total / 10)
    kitchen = features["kitchen_area"]
    features["area_conflict"] = ((living > total) | (kitchen > total) |
                                 (living + kitchen > total)).astype(int)
    features["living_area"] = living.mask((living > total) | (living < 0))
    features["kitchen_area"] = kitchen.mask((kitchen > total) | (kitchen < 0))

    metro = features.pop("metro_distance_m")
    metro = metro.where(metro >= 0)
    if correct_metro_distance:
        metro = metro.mask(metro.gt(METRO_DISTANCE_LIMIT_M), metro / 10)
    features["log_metro_distance"] = np.log1p(metro)
    features["log_total_area"] = np.log1p(total)
    return features.drop(columns="total_area")
