# src/visualize.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from load_data import load_iris_data
from utils import add_label_column

def plot_pairplot(df: pd.DataFrame) -> None:
    """
    Pairplot d. Features
    """
    sns.set(style="whitegrid")
    sns.pairplot(df, hue="label", palette="Set2", diag_kind="kde")
    plt.suptitle("Iris-Datensatz – Pairplot", y=1.02)
    plt.show()

def plot_feature_distributions(df: pd.DataFrame) -> None:
    """
    Histogramme für alle Features.
    """
    sns.set(style="whitegrid")
    features = df.columns[:-1]  
    for feature in features:
        plt.figure(figsize=(6, 4))
        sns.histplot(data=df, x=feature, hue="label", kde=True, palette="Set2", bins=20)
        plt.title(f"Verteilung von: {feature}")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    df = load_iris_data()
    df = add_label_column(df)
    plot_pairplot(df)
    plot_feature_distributions(df)
