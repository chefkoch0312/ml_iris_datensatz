# src/model.py

import joblib
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from load_data import load_iris_data
from utils import add_label_column

def save_model(model, path: str = "models/iris_model.pkl") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Modell gespeichert unter: {path}")

def load_model(path: str = "models/iris_model.pkl") -> LogisticRegression:
    model = joblib.load(path)
    print(f"Modell geladen von: {path}")
    return model

def train_model(df: pd.DataFrame) -> LogisticRegression:
    X = df.drop(columns=["target", "label"])
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Model Accuracy:", round(acc, 3))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=df["label"].unique()))

    return model

if __name__ == "__main__":
    df = load_iris_data()
    df = add_label_column(df)
    model = train_model(df)
    save_model(model)