# src/main.py  (very small bootstrap!)
import cv2, mediapipe as mp, time
from src.gestures import detect, Smoother

mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
smoother = Smoother(0.35)
with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6) as hands:
    while True:
        ok, img = cap.read()
        if not ok:
            break
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = hands.process(img_rgb)
        if res.multi_hand_landmarks:
            # take first hand for dev
            lm = [(p.x, p.y) for p in res.multi_hand_landmarks[0].landmark]
            lm = smoother.smooth(lm)
            gesture, conf = detect(lm)
            if gesture:
                cv2.putText(img, f"{gesture} {conf:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("ArcPalm Stage3 Ready", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()