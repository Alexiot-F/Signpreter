
"""
display.py (Aesthetic UI)
------------------------
Modern, clean, user-friendly UI for word-based gesture system
"""

import cv2


# Colors (BGR)
COLOR_BG      = (15, 15, 15)
COLOR_PANEL   = (25, 25, 25)
COLOR_ACCENT  = (0, 200, 120)
COLOR_TEXT    = (240, 240, 240)
COLOR_SUBTEXT = (160, 160, 160)
COLOR_WARN    = (0, 200, 255)


def draw_panel(frame, x, y, w, h, color, alpha=0.7):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


class DisplayEngine:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.bold = cv2.FONT_HERSHEY_DUPLEX

    def draw(self, frame, predicted_word, confidence, current_text,
             fps, hold_progress, cooldown):

        h, w = frame.shape[:2]

        # ---- TOP BAR ----
        draw_panel(frame, 0, 0, w, 60, COLOR_BG, 0.9)
        cv2.putText(frame, "Gesture → Speech",
                    (15, 38), self.bold, 0.7, COLOR_ACCENT, 2)

        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (w - 120, 35), self.font, 0.6, COLOR_SUBTEXT, 1)

        # ---- CENTER WORD DISPLAY ----
        if predicted_word:
            color = COLOR_ACCENT if not cooldown else COLOR_WARN

            text_size = cv2.getTextSize(predicted_word, self.bold, 2.5, 5)[0]
            x = (w - text_size[0]) // 2
            y = h // 2

            cv2.putText(frame, predicted_word,
                        (x, y),
                        self.bold, 2.5, color, 5)

            # Confidence
            conf_text = f"{int(confidence * 100)}%"
            cv2.putText(frame, conf_text,
                        (x, y + 50),
                        self.font, 0.7, COLOR_SUBTEXT, 2)
        else:
            cv2.putText(frame, "SHOW GESTURE",
                        (w // 2 - 160, h // 2),
                        self.font, 1.0, COLOR_SUBTEXT, 2)

        # ---- STATUS (small but useful) ----
        status = "COOLDOWN" if cooldown else ("DETECTING" if predicted_word else "WAITING")
        status_color = COLOR_WARN if cooldown else COLOR_ACCENT

        cv2.putText(frame, status,
                    (15, h - 110),
                    self.font, 0.6, status_color, 2)

        # ---- BOTTOM PANEL ----
        draw_panel(frame, 0, h - 80, w, 80, COLOR_PANEL, 0.9)

        cv2.putText(frame, "Output:",
                    (15, h - 50),
                    self.font, 0.6, COLOR_SUBTEXT, 1)

        display_text = current_text if current_text else "_"

        cv2.putText(frame, display_text,
                    (15, h - 15),
                    self.bold, 1.0, COLOR_TEXT, 2)

        return frame

    def draw_guide_box(self, frame):
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        size = 150

        cv2.rectangle(frame,
                      (cx - size, cy - size),
                      (cx + size, cy + size),
                      COLOR_ACCENT, 1)

        return frame

