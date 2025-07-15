# src/load_data.py

import pandas as pd
from skLearn.datasets import load_iris

def load_iris_data():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    return df

if __name__ == "__main__":
    df = load_iris_data()
    print("Datenvorschau:")
    print(df.head())
    print("\nZielklassen:", df['target'].unique())


