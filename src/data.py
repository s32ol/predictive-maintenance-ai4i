"""Data loading for the AI4I 2020 predictive maintenance dataset."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "ai4i2020.csv"

# Canonical AI4I 2020 schema (UCI ID 601 / Matzka 2020)
SCHEMA = {
    "UDI": "int64",
    "Product ID": "object",
    "Type": "object",
    "Air temperature [K]": "float64",
    "Process temperature [K]": "float64",
    "Rotational speed [rpm]": "int64",
    "Torque [Nm]": "float64",
    "Tool wear [min]": "int64",
    "Machine failure": "int64",
    "TWF": "int64",
    "HDF": "int64",
    "PWF": "int64",
    "OSF": "int64",
    "RNF": "int64",
}


def load(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the AI4I 2020 dataset and validate schema."""
    df = pd.read_csv(path)
    missing = set(SCHEMA) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df[list(SCHEMA)]
