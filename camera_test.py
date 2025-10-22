#!/usr/bin/env python3
"""
camera_test.py
Simple, robust OpenCV camera test.
Usage:
  python camera_test.py            # open default camera (index 0)
  python camera_test.py -d 1       # open camera index 1
  python camera_test.py -w 1280 -h 720
"""

import cv2
import time
import argparse
import signal
import sys

stop_flag = False
def handle_sigint(sig, frame):
    global stop_flag
    stop_flag = True

signal.signal(signal.SIGINT, handle_sigint)

def main():
    parser = argparse.ArgumentParser(description="OpenCV camera test")
    parser.add_argument('-d', '--device', type=int, default=0, help='camera device index (default 0)')
    parser.add_argument('-w', '--width',  type=int, default=1280, help='frame width')
    # parser.add_argument('-h', '--height', type=int, default=720, help='frame height')
    parser.add_argument('-f', '--flip', action='store_true', help='flip frame vertically (useful for some laptops)')
    args = parser.parse_args()

    device = args.device
    width = args.width
    # height = args.height

    print(f"Opening camera index {device} ...")
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)  # CAP_V4L2 works well on Linux; fallback below if fails

    # Fallback if CAP_V4L2 not available
    if not cap.isOpened():
        cap = cv2.VideoCapture(device)

    if not cap.isOpened():
        print("ERROR: Cannot open camera. Check device index and permissions.")
        print("Use `v4l2-ctl --list-devices` to see camera devices.")
        return 1

    # Try to set resolution (some cameras ignore these)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Read one frame to confirm
    ok, frame = cap.read()
    if not ok or frame is None:
        print("ERROR: Camera opened but frame could not be read.")
        cap.release()
        return 1

    print("Camera opened successfully. Press 'q' or Ctrl+C to quit.")

    # for FPS measurement
    prev = time.time()
    fps = 0
    frames = 0
    show_fps_every = 15

    window_name = "Camera Test (press q to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            if stop_flag:
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Warning: frame read failed, retrying ...")
                time.sleep(0.1)
                continue

            if args.flip:
                frame = cv2.flip(frame, 0)

            # draw a small overlay with FPS and resolution
            frames += 1
            if frames >= show_fps_every:
                now = time.time()
                fps = frames / (now - prev)
                prev = now
                frames = 0

            text = f"{frame.shape[1]}x{frame.shape[0]}  FPS: {fps:.1f}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow(window_name, frame)

            # key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except Exception as e:
        print("Exception:", e)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera closed, resources released.")

    return 0

if __name__ == "__main__":
    sys.exit(main())