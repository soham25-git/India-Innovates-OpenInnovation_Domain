#!/usr/bin/env python3
import cv2
import time
import os

def test_camera(index, w, h, backend_name, backend_api):
    print(f"\n--- Testing Index {index} | {w}x{h} | Backend: {backend_name} ---")
    if backend_name == "GSTREAMER_LIBCAMERA":
        pipeline = f"libcamerasrc ! video/x-raw, width={w}, height={h} ! videoconvert ! appsink"
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    elif backend_name == "GSTREAMER_V4L2":
        pipeline = f"v4l2src device=/dev/video{index} ! video/x-raw, width={w}, height={h} ! videoconvert ! appsink"
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(index, backend_api)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    if not cap.isOpened():
        print(f"❌ Could not open camera.")
        return False

    # Wait for a few frames
    for i in range(15):
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"✅ SUCCESS! Received frame: {frame.shape}")
            cv2.imwrite(f"diag_cam_{index}_{backend_name}_{w}x{h}.jpg", frame)
            cap.release()
            return True
        time.sleep(0.1)
    
    print(f"❌ Opened, but failed to read frames.")
    cap.release()
    return False

def main():
    print("🔍 KRISHI-EYE Camera Diagnostics")
    indices = [0, 1, 4] # Common RPi 5 camera indices
    resolutions = [(6560, 2464), (1280, 480), (640, 480)]
    backends = [
        ("GSTREAMER_LIBCAMERA", cv2.CAP_GSTREAMER),
        # ("GSTREAMER_V4L2", cv2.CAP_GSTREAMER),
        ("V4L2", cv2.CAP_V4L2)
    ]

    for idx in indices:
        for res in resolutions:
            for b_name, b_api in backends:
                if test_camera(idx, res[0], res[1], b_name, b_api):
                    print(f"\n🏆 WORKING CONFIG FOUND: Index {idx}, Res {res}, Backend {b_name}")
                    return

if __name__ == "__main__":
    main()
