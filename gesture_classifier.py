import numpy as np
import os
import joblib


class GestureClassifier:
    def __init__(self):
        self.model = None
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
                print("[INFO] Loaded ML model.")
            except Exception as e:
                print(f"[ERROR] Failed to load model: {e}")
        else:
            print("[ERROR] Model not found!")

    def predict(self, features):
        """
        Always returns (label, confidence)
        No filtering here — handled in main.py
        """

        if self.model is None:
            return None, 0.0

        try:
            features = np.array(features, dtype=np.float32)

            # Ensure correct shape
            if features.ndim != 1:
                return None, 0.0

            features = features.reshape(1, -1)

            # Predict class
            pred = self.model.predict(features)[0]

            # Predict confidence
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(features)[0]
                confidence = float(np.max(probs))
            else:
                # fallback confidence (not ideal but safe)
                confidence = 0.5

            return pred, confidence

        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            return None, 0.0