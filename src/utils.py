# src/utils.py

import pandas as pd

TARGET_LABELS = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

def add_label_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label"] = df["target"].map(TARGET_LABELS)
    return df
