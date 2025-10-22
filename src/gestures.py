# src/gestures.py
import math
import time

def hand_size(landmarks):
    # landmarks: list of (x,y) normalized coords
    # use wrist (0) to middle fingertip (12) distance as scale
    x0,y0 = landmarks[0]
    x12,y12 = landmarks[12]
    return math.hypot(x0-x12, y0-y12) + 1e-6

def norm_dist(a,b,landmarks):
    ax,ay = landmarks[a]; bx,by = landmarks[b]
    return math.hypot(ax-bx, ay-by) / hand_size(landmarks)

def simple_pinch(landmarks, thresh=0.12):
    # check thumb_tip (4) to index_tip (8)
    d = norm_dist(4,8,landmarks)
    return d < thresh

def detect(landmarks):
    """
    Return (gesture_name, confidence)
    Start simple: 'pinch', 'open', 'fist' or None
    """
    if simple_pinch(landmarks):
        return ("pinch", 0.9)
    # placeholder for other rules
    return (None, 0.0)
# smoothing & cooldown helpers
class Smoother:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.prev = None
    def smooth(self, pts):
        if self.prev is None:
            self.prev = pts
            return pts
        out = []
        for (x,y),(px,py) in zip(pts, self.prev):
            out.append((self.alpha*x + (1-self.alpha)*px,
                        self.alpha*y + (1-self.alpha)*py))
        self.prev = out
        return out

# cooldown usage example:
# last_action = 0
# if time.time()-last_action > cooldown: do_action(); last_action=time.time()