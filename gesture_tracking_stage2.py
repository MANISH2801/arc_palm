#!/usr/bin/env python3
"""
gesture_tracking_stage2.py
Stage 2 — Simple dynamic gesture tracking (temporal buffer + swipe detection).
Run: python gesture_tracking_stage2.py
Quit: press 'q'
"""

import time
import cv2
import mediapipe as mp
from collections import deque
import numpy as np

# ---------- CONFIG ----------
CAM_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

BUFFER_LEN = 12         # number of frames to keep for motion history
SWIPE_DIST_PX = 150     # min pixel travel required to consider a swipe
SWIPE_VEL_PX = 200      # optional velocity threshold (px/s)
DETECT_COOLDOWN = 0.6   # seconds to ignore repeated detections (debounce)

# ---------- mediapipe setup ----------
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

# ---------- helpers ----------
def norm_to_pixel(norm_landmark, frame_w, frame_h):
    return int(norm_landmark.x * frame_w), int(norm_landmark.y * frame_h)

def centroid_of_landmarks(landmarks, frame_w, frame_h):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    cx = int(np.mean(xs) * frame_w)
    cy = int(np.mean(ys) * frame_h)
    return cx, cy

# ---------- buffers ----------
# We'll keep a dict per hand-side ('Left'/'Right') mapping to deque of (ts, x, y)
history = {
    "Left": deque(maxlen=BUFFER_LEN),
    "Right": deque(maxlen=BUFFER_LEN)
}
last_detect_time = {
    "Left": 0.0,
    "Right": 0.0
}

# ---------- camera ----------
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_ANY)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
if not cap.isOpened():
    print("ERROR: Cannot open camera. Try a different CAM_INDEX.")
    raise SystemExit

prev_time = time.time()
print("Stage 2 running — press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Empty frame, retrying ...")
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        results = hands.process(img_rgb)
        img_rgb.flags.writeable = True

        now = time.time()

        # Draw stored trails
        for side, dq in history.items():
            pts = [(int(x), int(y)) for (_, x, y) in dq]
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i-1], pts[i], (200, 200, 0), 2, cv2.LINE_AA)

        # Process detected hands
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label  # "Left" or "Right"
                # use centroid as a robust anchor (alternative: use WRIST)
                cx, cy = centroid_of_landmarks(hand_landmarks.landmark, w, h)

                # store timestamped position
                history[label].append((now, cx, cy))

                # draw landmark skeleton
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_styles.get_default_hand_landmarks_style(),
                                          mp_styles.get_default_hand_connections_style())

                # draw latest centroid
                cv2.circle(frame, (cx, cy), 6, (0, 200, 0), -1)
                cv2.putText(frame, f"{label}", (cx+8, cy-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

                # Gesture detection (very simple swipe detector)
                dq = history[label]
                if len(dq) >= 3:
                    t0, x0, y0 = dq[0]
                    tN, xN, yN = dq[-1]
                    dt = max(tN - t0, 1e-6)
                    dx = xN - x0
                    dy = yN - y0
                    dist = np.hypot(dx, dy)
                    vx = dx / dt
                    vy = dy / dt

                    # Only trigger if cooldown passed
                    if now - last_detect_time[label] > DETECT_COOLDOWN:
                        # horizontal swipe
                        if abs(dx) > SWIPE_DIST_PX and abs(dx) > abs(dy):
                            if dx > 0:
                                gesture = "Swipe Right"
                            else:
                                gesture = "Swipe Left"
                            last_detect_time[label] = now
                            print(f"[{label}] {gesture}  dist={int(dx)} px  vel={int(vx)} px/s")
                            cv2.putText(frame, gesture, (50, 80 if label=="Left" else 120),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

                        # vertical swipe
                        elif abs(dy) > SWIPE_DIST_PX and abs(dy) > abs(dx):
                            if dy > 0:
                                gesture = "Swipe Down"
                            else:
                                gesture = "Swipe Up"
                            last_detect_time[label] = now
                            print(f"[{label}] {gesture}  dist={int(dy)} px  vel={int(vy)} px/s")
                            cv2.putText(frame, gesture, (50, 80 if label=="Left" else 120),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

        # FPS
        now2 = time.time()
        fps = 1.0 / (now2 - prev_time) if (now2 - prev_time) > 1e-6 else 0.0
        prev_time = now2
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("ArcPalm Stage2 - Motion Tracking (q to quit)", frame)
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