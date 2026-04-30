"""Download the AI4I 2020 dataset via kagglehub and stage it at data/ai4i2020.csv."""

from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub

DATASET = "stephanmatzka/predictive-maintenance-dataset-ai4i-2020"
CSV_NAME = "ai4i2020.csv"
LOCAL_PATH = Path(__file__).resolve().parents[1] / "data" / CSV_NAME


def main() -> None:
    cache_dir = Path(kagglehub.dataset_download(DATASET))
    cache_csv = next(cache_dir.rglob(CSV_NAME))

    LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cache_csv, LOCAL_PATH)

    print(f"kagglehub cache: {cache_csv}")
    print(f"local path:      {LOCAL_PATH}")


if __name__ == "__main__":
    main()
