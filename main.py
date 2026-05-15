
"""
Word-Based Sign Language to Speech System
----------------------------------------
Gesture → Word → Speech (Stable + Minimal)
"""

import cv2
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gesture_classifier import GestureClassifier
from text_speech import TextSpeechEngine
from hand_tracker import HandTracker
from display import DisplayEngine


def main():
    print("\n[INFO] Starting system...")

    # Init components
    hand_tracker = HandTracker()
    classifier = GestureClassifier()
    tts_engine = TextSpeechEngine()
    display_engine = DisplayEngine()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Webcam not found")
        return

    print("[INFO] Press 'q' to quit\n")

    # --- State ---
    current_word = ""
    last_word = ""
    stable_count = 0

    STABLE_THRESHOLD = 10
    COOLDOWN_FRAMES = 20
    cooldown_counter = 0

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # FPS
        curr_time = time.time()
        fps = 1 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time

        # Detect hand
        landmarks, _ = hand_tracker.detect(rgb)

        predicted_word = ""
        confidence = 0.0

        if landmarks is not None:
            hand_tracker.draw_landmarks(frame, landmarks)

            features = hand_tracker.extract_features(landmarks)
            predicted_word, confidence = classifier.predict(features)

            # Stability logic
            if predicted_word == last_word and predicted_word != "":
                stable_count += 1
            else:
                stable_count = 0
                last_word = predicted_word

            # Cooldown handling
            if cooldown_counter > 0:
                cooldown_counter -= 1

            # Accept stable gesture
            if stable_count >= STABLE_THRESHOLD and cooldown_counter == 0:
                current_word = predicted_word

                print(f"[DETECTED] {current_word}")

                # Speak once
                tts_engine.speak_async(current_word)

                cooldown_counter = COOLDOWN_FRAMES
                stable_count = 0

        # Draw UI
        frame = display_engine.draw(
            frame=frame,
            predicted_word=predicted_word,
            confidence=confidence,
            current_text=current_word,
            fps=fps,
            hold_progress=0,
            cooldown=cooldown_counter > 0
        )

        cv2.imshow("Gesture Speech System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    tts_engine.cleanup()

    print("[INFO] System closed.")


if __name__ == "__main__":
    main()
