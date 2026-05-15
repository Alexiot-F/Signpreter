
"""
gesture_classifier.py
---------------------
Word-based gesture classifier using trained ML model (.pkl)
"""

import numpy as np
import os
import joblib


class GestureClassifier:
    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        model_path = os.path.join(os.path.dirname(__file__), "model", "gesture_model.pkl")

        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                print("[INFO] Loaded word-based ML model.")
            except Exception as e:
                print(f"[ERROR] Failed to load model: {e}")
        else:
            print("[ERROR] Model not found!")

    def predict(self, features):
        """
        Predict word from feature vector (63 values)
        Returns (word, confidence)
        """
        if self.model is None:
            return "", 0.0

        try:
            pred = self.model.predict([features])[0]
            return pred, 0.95
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            return "", 0.0
