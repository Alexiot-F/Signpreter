import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
from hand_tracker import HandTracker
from gesture_classifier import GestureClassifier
from text_speech import speak


def main():
    tracker = HandTracker()
    classifier = GestureClassifier()

    cap = cv2.VideoCapture(0)

    sentence = ""
    last_label = None
    cooldown = 0

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

        # -------- GESTURE INPUT -------- #
        if label and label != last_label and cooldown == 0:
            if label == "SPACE":
                sentence += " "
            elif label == "DEL":
                sentence = sentence[:-1]
            elif label == "SPEAK":
                speak(sentence)
            else:
                sentence += label

            cooldown = 10  # prevent spam

        last_label = label

        if cooldown > 0:
            cooldown -= 1

        # -------- KEYBOARD CONTROLS -------- #
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('s'):  # speak
            speak(sentence)

        elif key == ord('c'):  # clear
            sentence = ""

        elif key == 8:  # backspace
            sentence = sentence[:-1]

        elif key == 32:  # space
            sentence += " "

        # -------- UI -------- #

        # Top bar
        cv2.rectangle(frame, (0, 0), (640, 60), (30, 30, 30), -1)

        display_label = label if label else "-"
        cv2.putText(frame, f"Letter: {display_label}", (10, 40),
                    font, 1, (0, 255, 0), 2)

        # Middle (sentence)
        cv2.rectangle(frame, (0, 60), (640, 140), (10, 10, 10), -1)

        cv2.putText(frame, f"{sentence}", (10, 110),
                    font, 1.2, (255, 255, 255), 2)

        # Bottom controls
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