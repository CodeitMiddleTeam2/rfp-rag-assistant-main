# src/prompt/loader.py

import yaml
from pathlib import Path


def load_prompt(name: str, version: str | None = None) -> str:
    base_dir = Path(__file__).resolve().parent
    path = base_dir / "prompts.yaml"

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    for item in data["prompts"]:
        if item["name"] == name and (version is None or item["version"] == version):
            return item["template"]

    raise ValueError(f"Prompt not found: {name}:{version}")
