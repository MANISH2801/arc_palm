# gesture_capture_stage1.py
# Simple MediaPipe + OpenCV camera demo for hand landmarks (copy-paste ready)

import time
import cv2
import mediapipe as mp

# ---- config ----
CAM_INDEX = 0          # change to 1 or 2 if your webcam is on a different index
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
MODEL_COMPLEXITY = 1   # 0,1,2 -> detection+landmark detail tradeoff
MIN_DETECTION_CONF = 0.5
MIN_TRACKING_CONF = 0.5

# ---- mediapipe init ----
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=MODEL_COMPLEXITY,
    min_detection_confidence=MIN_DETECTION_CONF,
    min_tracking_confidence=MIN_TRACKING_CONF
)

# ---- open camera ----
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_ANY)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    print(f"ERROR: Cannot open camera index {CAM_INDEX}. Try another index (0,1,2...).")
    raise SystemExit

# ---- FPS helper ----
prev_time = time.time()
fps = 0

print("Press 'q' to quit. If camera is blank, try CAM_INDEX=1 or 2.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("WARNING: Empty frame received. Re-check camera index or connection.")
            break

        # Flip for selfie view (optional)
        frame = cv2.flip(frame, 1)

        # Convert BGR -> RGB for mediapipe
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        results = hands.process(img_rgb)
        img_rgb.flags.writeable = True

        # Draw landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # draw landmarks + connections
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                # Optional: label left/right
                label = handedness.classification[0].label
                # get wrist coord for label position
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                h, w, _ = frame.shape
                cx, cy = int(wrist.x * w), int(wrist.y * h)
                cv2.putText(frame, label, (cx - 30, cy - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # FPS calc
        now = time.time()
        fps = 1.0 / (now - prev_time) if now != prev_time else fps
        prev_time = now
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show frame
        cv2.imshow("ArcPalm - Hand Capture (q to quit)", frame)

        # Quit if q pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Camera closed, exiting.")