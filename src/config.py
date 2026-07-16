from pathlib import Path

import yaml

CONFIG_DIR = Path("configs")


def load_yaml(filename: str) -> dict:
    with open(CONFIG_DIR / filename, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)