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

        layout.addLayout(form_layout)
        layout.addWidget(self.predict_btn)
        layout.addWidget(self.train_btn)
        layout.addWidget(self.result_label)
        self.setLayout(layout)

    def make_prediction(self):
        try:
            features = [float(inp.text()) for inp in self.inputs]
            result = predict_flower(features)
            self.result_label.setText(f"Vorhersage: {result}")
        except ValueError:
            QMessageBox.warning(self, "Fehler", "Bitte vier gültige numerische Werte eingeben.")

    def train_model(self):
        df = load_iris_data()         
        df = add_label_column(df)     
        model = train_model(df)       
        save_model(model)             
        QMessageBox.information(self, "Erfolg", "Modell wurde neu trainiert und gespeichert.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = IrisApp()
    window.show()
    sys.exit(app.exec_())
