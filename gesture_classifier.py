"""
gesture_classifier.py
---------------------
Letter-based gesture classifier using trained ML model (.pkl)
"""

import numpy as np
import os
import joblib


class GestureClassifier:
    def __init__(self):
        self.model = None

        
        LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["SPACE", "DEL", "SPEAK"]
        self._load_model()

    def _load_model(self):
        model_path = os.path.join(
            os.path.dirname(__file__),
            "model",
            "gesture_model.pkl"
        )

        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                print("[INFO] Loaded letter-based ML model.")
            except Exception as e:
                print(f"[ERROR] Failed to load model: {e}")
        else:
            print("[ERROR] Model not found!")

    def predict(self, features):
        """
        Predict letter from feature vector (63 values)
        Returns (label, confidence)
        """

        if self.model is None:
            return None, 0.0

        try:
            features = np.array(features).reshape(1, -1)

            # ✅ Get prediction
            pred = self.model.predict(features)[0]

            # ✅ Try getting confidence (if model supports it)
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(features)[0]
                confidence = np.max(probs)
            else:
                confidence = 0.9  # fallback

            # ✅ Confidence threshold (prevents noise)
            if confidence > 0.80:
                return pred, confidence
            else:
                return None, confidence

        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            return None, 0.0