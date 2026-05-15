"""
display.py
----------
Handles all OpenCV UI rendering for the sign language system.
Draws overlays, info panels, progress bars, and text onto the video frame.
"""

import cv2
import numpy as np


# Color palette (BGR)
COLOR_GREEN   = (0, 255, 100)
COLOR_YELLOW  = (0, 220, 255)
COLOR_CYAN    = (255, 220, 0)
COLOR_RED     = (50, 50, 255)
COLOR_WHITE   = (255, 255, 255)
COLOR_BLACK   = (0, 0, 0)
COLOR_DARK    = (20, 20, 20)
COLOR_BLUE    = (255, 120, 50)
COLOR_ORANGE  = (30, 140, 255)


def draw_filled_rect(frame, x, y, w, h, color, alpha=0.6):
    """Draw a semi-transparent filled rectangle."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


class DisplayEngine:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_bold = cv2.FONT_HERSHEY_DUPLEX

    def draw(self, frame, predicted_letter, confidence, current_text,
             fps, hold_progress, cooldown):
        h, w = frame.shape[:2]

        # ---- Top bar ----
        draw_filled_rect(frame, 0, 0, w, 60, COLOR_DARK, alpha=0.8)
        cv2.putText(frame, "Sign Language Recognition System",
                    (10, 38), self.font_bold, 0.75, COLOR_GREEN, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (w - 120, 38), self.font, 0.65, COLOR_YELLOW, 1)

        # ---- Left panel: detected letter ----
        draw_filled_rect(frame, 5, 70, 160, 180, COLOR_DARK, alpha=0.75)

        if predicted_letter:
            letter_color = COLOR_GREEN if not cooldown else COLOR_YELLOW
            cv2.putText(frame, predicted_letter,
                        (30, 170), self.font_bold, 4.0, letter_color, 6)
            conf_pct = int(confidence * 100)
            cv2.putText(frame, f"Conf: {conf_pct}%",
                        (12, 235), self.font, 0.6, COLOR_WHITE, 1)
        else:
            cv2.putText(frame, "---",
                        (30, 170), self.font_bold, 3.0, (100, 100, 100), 4)
            cv2.putText(frame, "No hand",
                        (18, 235), self.font, 0.6, (150, 150, 150), 1)

        cv2.putText(frame, "Letter:",
                    (12, 90), self.font, 0.6, COLOR_CYAN, 1)

        # ---- Hold progress bar ----
        bar_x, bar_y, bar_w, bar_h = 5, 255, 160, 18
        draw_filled_rect(frame, bar_x, bar_y, bar_w, bar_h, (40, 40, 40), alpha=0.9)
        fill_w = int(bar_w * hold_progress)
        if fill_w > 0:
            bar_color = COLOR_ORANGE if hold_progress < 1.0 else COLOR_GREEN
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), COLOR_WHITE, 1)
        cv2.putText(frame, "Hold",
                    (bar_x + 55, bar_y + 13), self.font, 0.45, COLOR_WHITE, 1)

        # ---- Status indicator ----
        status_text = "COOLDOWN" if cooldown else ("DETECTING" if predicted_letter else "WAITING")
        status_color = COLOR_YELLOW if cooldown else (COLOR_GREEN if predicted_letter else (120, 120, 120))
        cv2.putText(frame, status_text,
                    (10, 295), self.font, 0.55, status_color, 1)

        # ---- Bottom text panel ----
        panel_h = 100
        draw_filled_rect(frame, 0, h - panel_h, w, panel_h, COLOR_DARK, alpha=0.85)
        cv2.putText(frame, "Recognized Text:",
                    (10, h - panel_h + 22), self.font, 0.6, COLOR_CYAN, 1)

        # Display current text (word wrap at ~55 chars)
        display_text = current_text if current_text else "_"
        max_len = 50
        if len(display_text) > max_len:
            display_text = "..." + display_text[-(max_len - 3):]

        cv2.putText(frame, display_text,
                    (10, h - panel_h + 55), self.font_bold, 1.0, COLOR_WHITE, 2)

        # Controls hint
        cv2.putText(frame, "[S] Speak   [C] Clear   [SPACE] Space   [Q] Quit",
                    (10, h - 12), self.font, 0.45, (160, 160, 160), 1)

        # ---- Hand detection bounding box hint ----
        cv2.putText(frame, "Show hand in center",
                    (w // 2 - 90, h - panel_h - 10), self.font, 0.5, (160, 160, 160), 1)

        return frame

    def draw_guide_box(self, frame):
        """Draw a guide box in the center of the frame."""
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        size = 160
        cv2.rectangle(frame,
                      (cx - size, cy - size),
                      (cx + size, cy + size),
                      COLOR_GREEN, 2)
        return frame