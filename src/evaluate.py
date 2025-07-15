# src/evaluate.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from model import load_model
from load_data import load_iris_data
from utils import add_label_column, TARGET_LABELS

def evaluate_model():
    df = load_iris_data()
    df = add_label_column(df)
    
    X = df.drop(columns=["target", "label"])
    y = df["target"]
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = load_model()
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", round(acc, 3))
    
    print("\nClassification Report:")
    target_names = [TARGET_LABELS[label] for label in sorted(set(y_test))]
    print(classification_report(y_test, y_pred, target_names=target_names))

if __name__ == "__main__":
    evaluate_model()
