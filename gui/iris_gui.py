# gui/iris_gui.py

import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton,
    QLineEdit, QFormLayout, QMessageBox
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from load_data import load_iris_data
from utils import add_label_column
from model import train_model, save_model
from predict import predict_flower

class IrisApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Iris-Klassifikation mit ML")
        self.setMinimumWidth(400)

        layout = QVBoxLayout()
        form_layout = QFormLayout()

        self.inputs = []
        labels = ["Sepal Length (cm)", "Sepal Width (cm)", "Petal Length (cm)", "Petal Width (cm)"]
        for label in labels:
            line_edit = QLineEdit()
            line_edit.setPlaceholderText("z. B. 5.1")
            self.inputs.append(line_edit)
            form_layout.addRow(label + ":", line_edit)

        self.result_label = QLabel("Vorhersage: (noch keine)")
        self.result_label.setStyleSheet("font-weight: bold; margin-top: 10px;")

        self.predict_btn = QPushButton("Vorhersage")
        self.train_btn = QPushButton("Modell trainieren")

        self.predict_btn.clicked.connect(self.make_prediction)
        self.train_btn.clicked.connect(self.train_model)

        self.reset_btn = QPushButton("Eingaben zurücksetzen")
        self.reset_btn.clicked.connect(self.reset_inputs)

        layout.addLayout(form_layout)
        layout.addWidget(self.predict_btn)
        layout.addWidget(self.train_btn)
        layout.addWidget(self.reset_btn)
        layout.addWidget(self.result_label)
        self.setLayout(layout)

    def make_prediction(self):
        try:
            # Eingabewerte prüfen
            features = []
            for inp in self.inputs:
                value = inp.text().strip()
                if not value:
                    raise ValueError("Ein oder mehrere Felder sind leer.")
                num = float(value)
                if num < 0 or num > 10:
                    raise ValueError("Wert außerhalb des gültigen Bereichs (0–10).")
                features.append(num)

            result = predict_flower(features)
            self.result_label.setText(f"Vorhersage: {result}")
            self.result_label.setStyleSheet("color: green; font-weight: bold;")
        
        except ValueError as e:
            QMessageBox.warning(self, "Ungültige Eingabe", str(e))
            self.result_label.setText("Vorhersage: –")
            self.result_label.setStyleSheet("color: red; font-weight: bold;")

    def train_model(self):
        df = load_iris_data()         
        df = add_label_column(df)     
        model = train_model(df)       
        save_model(model)             
        QMessageBox.information(self, "Erfolg", "Modell wurde neu trainiert und gespeichert.")

    def reset_inputs(self):
        for inp in self.inputs:
            inp.clear()
        self.result_label.setText("Vorhersage: (noch keine)")
        self.result_label.setStyleSheet("color: black;")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = IrisApp()
    window.show()
    sys.exit(app.exec_())
