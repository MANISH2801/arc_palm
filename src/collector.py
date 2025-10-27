# src/collector.py
import time, csv, os, json
from pathlib import Path

OUT_DIR = Path("data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def save_sample(landmarks, label):
    t = int(time.time()*1000)
    fn = OUT_DIR / f"{label}_{t}.json"
    with open(fn, "w") as f:
        json.dump({"landmarks": landmarks, "label": label, "ts": t}, f)

# later integrate with MediaPipe capture loop to call save_sample(...)