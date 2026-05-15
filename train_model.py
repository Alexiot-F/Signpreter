
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

# Load CSV data
with open(DATA_PATH, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        label = row[0]
        features = list(map(float, row[1:]))
        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("[INFO] Dataset loaded:", X.shape)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Accuracy
acc = model.score(X_test, y_test)
print(f"[INFO] Accuracy: {acc*100:.2f}%")

# Save model
joblib.dump(model, os.path.join(MODEL_DIR, "gesture_model.pkl"))

print("[INFO] Model saved to model/gesture_model.pkl")
