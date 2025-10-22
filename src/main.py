##!/usr/bin/env python3
# src/main.py
# Robust launcher for ArcPalm Stage-3 detection loop.
# - ensures src package import works
# - reads camera frames, runs MediaPipe hands
# - smooths landmarks and calls src.gestures.detect()
# - draws landmarks, labels, FPS and prints detections

import sys, os
# allow importing "src" package when running python src/main.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import cv2
import mediapipe as mp

# import your gesture utilities
# make sure src/gestures.py exists and defines detect() and Smoother
from src.gestures import detect, Smoother

# ---- CONFIG ----
CAM_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FLIP = True          # mirror for selfie view
SMOOTHER_ALPHA = 0.35

# ---- MediaPipe init ----
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---- helpers ----
def lm_to_xy_list(landmark_list):
    """Convert normalized MediaPipe landmark list to list of (x,y) tuples."""
    return [(lm.x, lm.y) for lm in landmark_list]

def draw_label(frame, text, pos=(10,60), color=(0,255,255)):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

# ---- smoother (from your gestures module) ----
smoother = Smoother(alpha=SMOOTHER_ALPHA)

# ---- camera ----
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_ANY)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    print(f"ERROR: Cannot open camera index {CAM_INDEX}. Try idx 0,1,2...")
    raise SystemExit

prev_time = time.time()
print("ArcPalm Stage-3 launcher running. Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            # temporary read failure, skip and continue
            time.sleep(0.02)
            continue

        if FLIP:
            frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape

        # prepare frame for MediaPipe
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        results = hands.process(img_rgb)
        img_rgb.flags.writeable = True

        gesture_shown = None
        conf_shown = 0.0

        # if hands detected, process each
        if results.multi_hand_landmarks and results.multi_handedness:
            # iterate over detected hands with handedness
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label  # "Left" or "Right"

                # convert to list of (x,y) normalized coords
                lm_list = lm_to_xy_list(hand_landmarks.landmark)

                # smooth positions (smoother expects list of (x,y) pairs)
                try:
                    lm_smoothed = smoother.smooth(lm_list)
                except Exception:
                    # fallback to raw if smoother fails
                    lm_smoothed = lm_list

                # run your gesture detector: expect (gesture_name, confidence)
                try:
                    gesture_name, confidence = detect(lm_smoothed)
                except Exception as e:
                    # detection function raised error — print and continue safely
                    print("detect() error:", repr(e))
                    gesture_name, confidence = (None, 0.0)

                # draw skeleton & landmarks using original (not-smoothed) mp landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

                # draw label near wrist
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                wx, wy = int(wrist.x * w), int(wrist.y * h)
                cv2.putText(frame, f"{label}", (wx + 8, wy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

                # if gesture recognized, annotate and print once
                if gesture_name:
                    gesture_shown = f"{label}: {gesture_name} ({confidence:.2f})"
                    conf_shown = confidence
                    # print to console (useful for logging)
                    print(f"[{label}] {gesture_name}  conf={confidence:.2f}")

        # compute and draw FPS
        now = time.time()
        fps = 1.0 / (now - prev_time) if (now - prev_time) > 1e-6 else 0.0
        prev_time = now
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # show detected gesture (if any)
        if gesture_shown:
            draw_label(frame, gesture_shown, pos=(10, 70))

        # display
        cv2.imshow("ArcPalm Stage-3 (q to quit)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Exiting.")