"""
hand_tracker.py
---------------
Handles hand detection and landmark extraction using MediaPipe.
Provides normalized feature vectors for the gesture classifier.
"""

import cv2
import mediapipe as mp
import numpy as np


class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )

        # Custom landmark drawing spec
        self.landmark_spec = self.mp_drawing.DrawingSpec(
            color=(0, 255, 128), thickness=2, circle_radius=4
        )
        self.connection_spec = self.mp_drawing.DrawingSpec(
            color=(255, 255, 0), thickness=2
        )

    def detect(self, rgb_frame):
        """
        Detect hands in the frame.
        Returns (hand_landmarks, bounding_box) or (None, None).
        """
        results = self.hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            hand_lm = results.multi_hand_landmarks[0]
            # Compute bounding box
            h, w, _ = rgb_frame.shape
            xs = [lm.x * w for lm in hand_lm.landmark]
            ys = [lm.y * h for lm in hand_lm.landmark]
            bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
            return hand_lm, bbox
        return None, None

    def draw_landmarks(self, bgr_frame, hand_landmarks):
        """Draw hand skeleton on the frame."""
        self.mp_drawing.draw_landmarks(
            bgr_frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.landmark_spec,
            self.connection_spec,
        )

    def extract_features(self, hand_landmarks):
        """
        Extract normalized landmark coordinates as a flat feature vector.
        Normalization: translate to wrist origin, scale by hand size.
        Returns numpy array of shape (63,) — 21 landmarks × (x, y, z).
        """
        landmarks = hand_landmarks.landmark
        # Wrist as origin (landmark 0)
        wrist_x = landmarks[0].x
        wrist_y = landmarks[0].y
        wrist_z = landmarks[0].z

        coords = []
        for lm in landmarks:
            coords.append(lm.x - wrist_x)
            coords.append(lm.y - wrist_y)
            coords.append(lm.z - wrist_z)

        features = np.array(coords, dtype=np.float32)

        # Normalize by the max absolute value (scale invariance)
        max_val = np.max(np.abs(features))
        if max_val > 0:
            features /= max_val

        return features