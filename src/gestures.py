# src/gestures.py
"""
ArcPalm – proportional gesture control
Gestures:
  • Swipe Up    → Volume Up  (distance-scaled)
  • Swipe Down  → Volume Down
  • Swipe Right → Brightness Up  (distance-scaled)
  • Swipe Left  → Brightness Down
  • Pinch       → Mute / Un-mute
"""

import os
import math
import time
import shutil
from collections import deque
from typing import List, Tuple, Optional

# --------------------------------------------------
# Landmark helpers
# --------------------------------------------------
def _xy(l):
    if isinstance(l, (list, tuple)):
        return float(l[0]), float(l[1])
    return float(l.x), float(l.y)

def get_point(landmarks, idx):
    return _xy(landmarks[idx])

def hand_size(landmarks):
    x0, y0 = get_point(landmarks, 0)
    x12, y12 = get_point(landmarks, 12)
    return math.hypot(x0 - x12, y0 - y12) + 1e-6

def norm_dist(a, b, landmarks):
    ax, ay = get_point(landmarks, a)
    bx, by = get_point(landmarks, b)
    return math.hypot(ax - bx, ay - by) / hand_size(landmarks)

# --------------------------------------------------
# Motion tracking
# --------------------------------------------------
_motion_x = deque(maxlen=6)
_motion_y = deque(maxlen=6)

def _movement_distance():
    if len(_motion_x) < 2:
        return 0.0
    dx = _motion_x[-1] - _motion_x[0]
    dy = _motion_y[-1] - _motion_y[0]
    return math.hypot(dx, dy)

# --------------------------------------------------
# Gesture detection
# --------------------------------------------------
def simple_pinch(landmarks, thresh=0.22):
    try:
        d = norm_dist(4, 8, landmarks)
        return d < thresh
    except Exception:
        return False

def detect_swipes(landmarks):
    """Return (gesture_name, travel_distance)"""
    if not landmarks:
        return None, 0.0
    wx, wy = get_point(landmarks, 0)
    _motion_x.append(wx)
    _motion_y.append(wy)
    if len(_motion_x) < _motion_x.maxlen:
        return None, 0.0

    dx = _motion_x[-1] - _motion_x[0]
    dy = _motion_y[-1] - _motion_y[0]
    dist = _movement_distance()
    gesture = None

    # thresholds
    if abs(dx) > abs(dy):
        if dx > 0.08:
            gesture = "swipe_right"
        elif dx < -0.08:
            gesture = "swipe_left"
    else:
        if dy < -0.08:
            gesture = "swipe_up"
        elif dy > 0.08:
            gesture = "swipe_down"

    if gesture:
        _motion_x.clear(); _motion_y.clear()
        return gesture, dist
    return None, 0.0

def detect(landmarks):
    """Return (gesture_name, confidence)."""
    if not landmarks or len(landmarks) < 9:
        return (None, 0.0)

    if simple_pinch(landmarks):
        return ("pinch", 0.95)

    swipe, dist = detect_swipes(landmarks)
    if swipe:
        return (f"{swipe}:{dist:.3f}", 0.9)

    return (None, 0.0)

# --------------------------------------------------
# Smoothing (for landmark jitter)
# --------------------------------------------------
class Smoother:
    def __init__(self, alpha=0.35):
        self.alpha = alpha
        self.prev = None
    def smooth(self, pts: List[Tuple[float,float]]):
        if self.prev is None:
            self.prev = [(float(x), float(y)) for x,y in pts]
            return pts
        out = []
        for (x,y),(px,py) in zip(pts, self.prev):
            out.append((
                self.alpha*x + (1-self.alpha)*px,
                self.alpha*y + (1-self.alpha)*py
            ))
        self.prev = out[:]
        return out

# --------------------------------------------------
# Action mapping
# --------------------------------------------------
_last_action_time = 0.0
_COOLDOWN = 0.6

def _can_fire():
    return time.time() - _last_action_time > _COOLDOWN
def _set_last():
    global _last_action_time; _last_action_time = time.time()

def perform_action(gesture):
    if not gesture or not _can_fire():
        return

    base = gesture.split(":")[0]
    try:
        dist = float(gesture.split(":")[1])
    except Exception:
        dist = 0.1

    # --- proportional scaling ---
    # long swipe (≈0.5 norm units) → up to 100% change
    scale = min(1.0, max(0.05, dist * 2.0))   # 0.05–1.0
    steps = int(scale * 10)                   # up to 10 key events
    bright_step = int(scale * 50)             # up to 50%

    if shutil.which("xdotool") is None:
        print(f"[perform_action] would perform {base} ({steps})")
        _set_last(); return

    try:
        if base == "swipe_up":
            os.system(f"for i in $(seq 1 {steps}); do xdotool key XF86AudioRaiseVolume; done")
        elif base == "swipe_down":
            os.system(f"for i in $(seq 1 {steps}); do xdotool key XF86AudioLowerVolume; done")
        elif base == "swipe_right":
            os.system(f"brightnessctl set +{bright_step}%")
        elif base == "swipe_left":
            os.system(f"brightnessctl set {max(10, 100 - bright_step)}%")
        elif base == "pinch":
            os.system("xdotool key XF86AudioMute")
        else:
            print("[perform_action] Unknown gesture:", base)
    except Exception as e:
        print("perform_action error:", e)

    os.system(f'notify-send "ArcPalm" "Gesture: {base} (scale {scale:.2f})"')
    _set_last()