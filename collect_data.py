"""
collect_data.py
----------------
Collects hand landmark data for training a LETTER-based gesture model.
"""

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"  # Fix Qt Wayland issue (Linux)

import cv2
import sys
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hand_tracker import HandTracker


# ---------------- CONFIG ---------------- #
LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["SPACE", "DEL", "SPEAK"]
SAMPLES_PER_CLASS = 100   # keep low for testing, increase later
# ---------------------------------------- #

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)


def collect():
    tracker = HandTracker()
    cap = cv2.VideoCapture(0)

    current_label = None
    recording = False
    sample_count = 0
    stability_counter = 0

    data_buffer = []  # ✅ for fast batch writing
    collected = {label: 0 for label in LABELS}

    csv_path = os.path.join(DATASET_DIR, "data.csv")

    # ✅ Load existing dataset
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[-1] in collected:
                    collected[row[-1]] += 1
        print(f"[INFO] Existing samples: {sum(collected.values())}")

    print("\n=== LETTER DATA COLLECTION ===")
    print("Press A-Z to record letters")
    print("Press 'q' to quit\n")

    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        lm, _ = tracker.detect(rgb)

        if lm is not None:
            tracker.draw_landmarks(frame, lm)
            features = tracker.extract_features(lm)

            # ✅ FAST + STABLE COLLECTION
            if recording and current_label:
                stability_counter += 1

                if stability_counter > 2 and sample_count < SAMPLES_PER_CLASS:
                    data_buffer.append(features.tolist() + [current_label])

                    sample_count += 1
                    collected[current_label] += 1
                    stability_counter = 0

                if sample_count >= SAMPLES_PER_CLASS:
                    recording = False
                    print(f"[INFO] Done collecting '{current_label}'")

        # ✅ Batch write (fast)
        if len(data_buffer) >= 20:
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(data_buffer)
            data_buffer.clear()

        # -------- KEY HANDLING -------- #
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key != 255:
            try:
                char = chr(key).upper()
                if char in LABELS:
                    current_label = char
                    sample_count = 0
                    stability_counter = 0
                    recording = True
                    print(f"[INFO] Recording '{current_label}'...")
            except:
                pass

        # -------- UI TOP -------- #
        cv2.rectangle(frame, (0, 0), (640, 50), (20, 20, 20), -1)

        if recording:
            status = f"Recording: {current_label} [{sample_count}/{SAMPLES_PER_CLASS}]"
            color = (0, 255, 0)
        else:
            status = "Press A-Z to record letters"
            color = (200, 200, 200)

        cv2.putText(frame, status, (10, 35), font, 0.7, color, 2)

        # -------- UI BOTTOM -------- #
        cv2.rectangle(frame, (0, 430), (640, 480), (20, 20, 20), -1)

        done = sum(1 for v in collected.values() if v >= SAMPLES_PER_CLASS)

        cv2.putText(frame,
                    f"Completed: {done}/{len(LABELS)} labels | q=quit",
                    (10, 465), font, 0.55, (180, 180, 180), 1)

        cv2.imshow("Letter Data Collection", frame)

    # ✅ Save remaining data
    if data_buffer:
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(data_buffer)

    cap.release()
    cv2.destroyAllWindows()

    print("\n[INFO] Collection complete.")
    print(f"[INFO] Data saved to: {csv_path}")


if __name__ == "__main__":
    collect()