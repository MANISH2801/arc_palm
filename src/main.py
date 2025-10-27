#!/usr/bin/env python3
# src/main.py
# ArcPalm Stage-4 ready main: HUD + debounce + safe action calls

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import cv2
import mediapipe as mp

from src.gestures import detect, Smoother, perform_action

# ---- CONFIG ----
CAM_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FLIP = True
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
    return [(lm.x, lm.y) for lm in landmark_list]

def draw_label(frame, text, pos=(10,60), color=(0,255,255)):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

# ---- smoother ----
smoother = Smoother(alpha=SMOOTHER_ALPHA)

# ---- HUD / debounce state ----
prev_time = time.time()
p_time = 0.0
display_text = ""
display_timer = 0.0
display_ttl = 1.5  # seconds to show action popup
prev_gesture = None
prev_conf = 0.0

print("\n==============================")
print("   🚀  ArcPalm Stage-4 HUD Ready ")
print("   Press 'q' to quit anytime.")
print("==============================\n")

# ---- camera ----
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_ANY)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    print(f"ERROR: Cannot open camera index {CAM_INDEX}. Try idx 0,1,2...")
    raise SystemExit

print("ArcPalm Stage-4 launcher running. Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02)
            continue

        if FLIP:
            frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        results = hands.process(img_rgb)
        img_rgb.flags.writeable = True

        gesture_shown = None
        conf_shown = 0.0

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label

                lm_list = lm_to_xy_list(hand_landmarks.landmark)

                try:
                    lm_smoothed = smoother.smooth(lm_list)
                except Exception:
                    lm_smoothed = lm_list

                # detection + safe action firing with debounce (only when gesture changes)
                try:
                    gesture_name, confidence = detect(lm_smoothed)
                    # normalize base name (if detect returns "swipe_up:0.123")
                    base_name = gesture_name.split(":")[0] if gesture_name else None

                    # show gesture (but fire action only when changed and confident)
                    if base_name:
                        gesture_shown = f"{label}: {base_name}"
                        conf_shown = confidence

                    # fire action when it changed from previous and confidence high enough
                    if base_name and confidence > 0.75 and base_name != prev_gesture:
                        perform_action(gesture_name)  # gesture_name may include dist e.g. "swipe_up:0.123"
                        display_text = f"Action: {base_name}"
                        display_timer = time.time()
                        prev_gesture = base_name
                        prev_conf = confidence

                    # reset prev_gesture to None if no gesture for a short time (so repeated same gesture later can trigger again)
                    if not base_name:
                        # small cooldown before clearing to avoid flicker
                        if time.time() - display_timer > 0.6:
                            prev_gesture = None

                except Exception as e:
                    print("detect() error:", repr(e))
                    gesture_name, confidence = (None, 0.0)

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                wx, wy = int(wrist.x * w), int(wrist.y * h)
                cv2.putText(frame, f"{label}", (wx + 8, wy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

                if gesture_shown:
                    # small per-hand overlay
                    cv2.putText(frame, f"{gesture_shown} ({conf_shown:.2f})", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)

        # FPS HUD
        now = time.time()
        fps = 1.0 / (now - prev_time) if (now - prev_time) > 1e-6 else 0.0
        prev_time = now
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # show action popup briefly
        if display_text and (time.time() - display_timer < display_ttl):
            cv2.putText(frame, display_text, (int(w*0.5)-80, int(h*0.85)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 255), 2)
        else:
            # clear display_text only after ttl
            if time.time() - display_timer >= display_ttl:
                display_text = ""

        cv2.imshow("ArcPalm Stage-4 (q to quit)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("✅  Stage-4 session closed safely.\n")
    print("Exiting.")