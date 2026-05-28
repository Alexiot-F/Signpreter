import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import time
from hand_tracker import HandTracker
from gesture_classifier import GestureClassifier
from text_speech import speak


def main():
    tracker = HandTracker()
    classifier = GestureClassifier()

    cap = cv2.VideoCapture(0)

    sentence = ""

    # -------- CONTROL VARIABLES -------- #
    last_label = None
    last_time = 0

    stable_label = None
    stable_count = 0

    CONF_THRESHOLD = 0.3    
    STABLE_THRESHOLD = 3      
    COOLDOWN_TIME = 1.0    

    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        lm, _ = tracker.detect(rgb)

        label = None
        confidence = 0

        if lm is not None:
            tracker.draw_landmarks(frame, lm)
            features = tracker.extract_features(lm)

            label, confidence = classifier.predict(features)

        # -------- STABILITY CHECK -------- #
        if label == stable_label:
            stable_count += 1
        else:
            stable_label = label
            stable_count = 1

        # -------- ACCEPT GESTURE -------- #
        current_time = time.time()

        if (
            stable_label
            and stable_count >= STABLE_THRESHOLD
            and confidence >= CONF_THRESHOLD
            and (current_time - last_time) > COOLDOWN_TIME
        ):
            print("ACCEPTED:", stable_label, confidence)

            if stable_label == "SPACE":
                sentence += " "
            elif stable_label == "DEL":
                sentence = sentence[:-1]
            elif stable_label == "SPEAK":
                speak(sentence)
            else:
                sentence += stable_label

            last_time = current_time
            last_label = stable_label

        # -------- KEYBOARD CONTROLS -------- #
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            speak(sentence)
        elif key == ord('c'):
            sentence = ""
        elif key == 8:
            sentence = sentence[:-1]
        elif key == 32:
            sentence += " "

        # -------- UI -------- #
        cv2.rectangle(frame, (0, 0), (640, 60), (30, 30, 30), -1)

        display_label = label if label else "-"
        cv2.putText(frame, f"Letter: {display_label}", (10, 40),
                    font, 1, (0, 255, 0), 2)

        cv2.rectangle(frame, (0, 60), (640, 140), (10, 10, 10), -1)

        cv2.putText(frame, f"{sentence}", (10, 110),
                    font, 1.2, (255, 255, 255), 2)

        cv2.rectangle(frame, (0, 400), (640, 480), (20, 20, 20), -1)

        cv2.putText(frame,
                    "Q=Quit | S=Speak | C=Clear | BACKSPACE=Delete | SPACE=Space",
                    (10, 440), font, 0.5, (200, 200, 200), 1)

        cv2.putText(frame,
                    f"Confidence: {confidence:.2f}",
                    (10, 465), font, 0.5, (150, 150, 150), 1)

        cv2.imshow("Signpreter", frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()