# src/predict.py

import numpy as np
import pandas as pd
from model import load_model
from utils import TARGET_LABELS

FEATURE_NAMES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]

def predict_flower(features: list[float]) -> str:
    if len(features) != 4:
        raise ValueError("Genau vier numerische Eingabewerte werden benötigt.")
    
    model = load_model()

    input_df = pd.DataFrame([features], columns=FEATURE_NAMES)

    prediction = model.predict(input_df)[0]
    label = TARGET_LABELS[prediction]
    return label

if __name__ == "__main__":
    # Bsp: Setosa
    test_input = [5.1, 3.5, 1.4, 0.2]
    result = predict_flower(test_input)
    print("Vorhersage für Eingabe", test_input, "→", result)