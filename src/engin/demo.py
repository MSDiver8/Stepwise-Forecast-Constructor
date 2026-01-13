import json
from importlib import resources

def load_demo_data():
    """
    Загружает демонстрационные данные,
    поставляемые вместе с пакетом.
    """
    with resources.files("engin").joinpath(
        "data/data_for_test.json"
    ).open("r", encoding="utf-8") as f:
        return json.load(f)