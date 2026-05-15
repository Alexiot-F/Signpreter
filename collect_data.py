"""
collect_data.py
----------------
Collects hand landmark data for training a custom WORD-based gesture model.
"""

import cv2
import os
import sys
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hand_tracker import HandTracker


# ---------------- CONFIG ---------------- #
WORDS = ["HELLO", "YES", "NO", "STOP", "CALL", "FUCK YOU"]
SAMPLES_PER_CLASS = 150

KEY_MAP = {
    ord('1'): "HELLO",
    ord('2'): "YES",
    ord('3'): "NO",
    ord('4'): "STOP",
    ord('5'): "CALL",
    ord('6'): "FUCK YOU"
}
# ---------------------------------------- #

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)


def collect():
    tracker = HandTracker()
    cap = cv2.VideoCapture(0)

    current_label = None
    recording = False
    sample_count = 0
    collected = {w: 0 for w in WORDS}

    csv_path = os.path.join(DATASET_DIR, "data.csv")

    # Load existing data
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0] in collected:
                    collected[row[0]] += 1
        print(f"[INFO] Existing samples: {sum(collected.values())}")

    print("\n=== WORD DATA COLLECTION ===")
    print("Press 1-6 to record words")
    print("Press 'q' to quit\n")

    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        lm, bbox = tracker.detect(rgb)

        if lm is not None:
            tracker.draw_landmarks(frame, lm)
            features = tracker.extract_features(lm)

            if recording and current_label:
                if sample_count < SAMPLES_PER_CLASS:
                    with open(csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([current_label] + features.tolist())

                    sample_count += 1
                    collected[current_label] += 1
                else:
                    recording = False
                    print(f"[INFO] Done collecting '{current_label}'")

        # -------- KEY HANDLING -------- #
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key in KEY_MAP:
            current_label = KEY_MAP[key]
            sample_count = 0
            recording = True
            print(f"[INFO] Recording '{current_label}'...")

        # -------- UI TOP -------- #
        cv2.rectangle(frame, (0, 0), (640, 50), (20, 20, 20), -1)

        if recording:
            status = f"Recording: {current_label} [{sample_count}/{SAMPLES_PER_CLASS}]"
            color = (0, 255, 0)
        else:
            status = "Press 1-6 to record words"
            color = (200, 200, 200)

        cv2.putText(frame, status, (10, 35), font, 0.7, color, 2)

        # -------- UI BOTTOM -------- #
        cv2.rectangle(frame, (0, 430), (640, 480), (20, 20, 20), -1)
        done = sum(1 for v in collected.values() if v >= SAMPLES_PER_CLASS)

        cv2.putText(frame,
                    f"Completed: {done}/{len(WORDS)} words | q=quit",
                    (10, 465), font, 0.55, (180, 180, 180), 1)

        cv2.imshow("Word Data Collection", frame)

    # -------- CLEAN EXIT -------- #
    cap.release()
    cv2.destroyAllWindows()

    print("\n[INFO] Collection complete.")
    print(f"[INFO] Data saved to: {csv_path}")


if __name__ == "__main__":
    collect()