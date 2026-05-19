import numpy as np
import csv
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

DATA_PATH = "dataset/data.csv"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

X = []
y = []

# ✅ Load CSV data (FIXED FORMAT)
with open(DATA_PATH, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) < 64:
            continue  # skip broken rows

        # 🔥 IMPORTANT CHANGE:
        features = list(map(float, row[:-1]))   # first 63 values
        label = row[-1]                         # last value = label

        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("[INFO] Dataset loaded:", X.shape)

# ✅ Normalize data (VERY IMPORTANT)
X = X / np.max(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ Better model config
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42
)

model.fit(X_train, y_train)

# Accuracy
acc = model.score(X_test, y_test)
print(f"[INFO] Accuracy: {acc*100:.2f}%")

# Save model
model_path = os.path.join(MODEL_DIR, "gesture_model.pkl")
joblib.dump(model, model_path)

print(f"[INFO] Model saved to {model_path}")