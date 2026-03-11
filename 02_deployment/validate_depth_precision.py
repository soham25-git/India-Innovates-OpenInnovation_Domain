#!/usr/bin/env python3
"""
KRISHI-EYE: Single-Point Precision Depth (Monocular)
Model: SCDepthV3 (NPU Accelerated)
Feature: One-Touch Calibration for "Exact" Distance.
"""
import os
import cv2
import numpy as np
import time
import argparse
import subprocess
from hailo_platform import (HEF, VDevice, HailoSchedulingAlgorithm, FormatType)

# ============================================================
# Camera Capture Bridge
# ============================================================
class SubprocessVideoCapture:
    def __init__(self, camera_idx=0, w=640, h=480):
        self.w, self.h = w, h
        self.cmd = [
            "rpicam-vid", "--camera", str(camera_idx),
            "--width", str(w), "--height", str(h),
            "--codec", "yuv420", "--nopreview", "-t", "0", "--inline", "--flush", "-o", "-"
        ]
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=w*h*2)
        self.frame_size = int(w * h * 1.5)

    def read(self):
        raw = self.proc.stdout.read(self.frame_size)
        if len(raw) < self.frame_size: return False, None
        yuv = np.frombuffer(raw, dtype=np.uint8).reshape((int(self.h * 1.5), self.w))
        return True, cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

    def release(self):
        if hasattr(self, 'proc') and self.proc:
            self.proc.terminate()

# ============================================================
# SCDepthV3 Inference
# ============================================================
class HailoSCDepthInference:
    def __init__(self, hef_path, vdevice):
        self.hef = HEF(hef_path)
        self.infer_model = vdevice.create_infer_model(hef_path)
        self.infer_model.set_batch_size(1)
        for output in self.infer_model.outputs:
            output.set_format_type(FormatType.FLOAT32)
        input_info = self.hef.get_input_vstream_infos()[0]
        self.target_h, self.target_w = input_info.shape[0], input_info.shape[1]

    def run(self, frame):
        resized = cv2.resize(frame, (self.target_w, self.target_h))
        input_data = np.ascontiguousarray(np.expand_dims(resized, axis=0)).astype(np.uint8)
        output_buffers = {out.name: np.empty(self.infer_model.output(out.name).shape, dtype=np.float32) 
                         for out in self.hef.get_output_vstream_infos()}
        with self.infer_model.configure() as configured_model:
            bindings = configured_model.create_bindings(output_buffers=output_buffers)
            bindings.input().set_buffer(input_data)
            configured_model.run([bindings], 10000)
        return list(output_buffers.values())[0].squeeze()

# ============================================================
# Main Implementation
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hef", default="scdepthv3.hef", help="Path to scdepthv3 HEF")
    args = parser.parse_args()

    # --- Calibration Logic ---
    # We use a simple linear mapping: depth = abs(val * scale + bias)
    # The user can calibrate this live.
    scale = -1.45
    bias = -4.5
    calib_set = False

    cap = SubprocessVideoCapture(w=640, h=480)
    
    print("🚀 Initializing Monocular Precision Probe...")
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    
    try:
        with VDevice(params) as vdevice:
            depth_inf = HailoSCDepthInference(args.hef, vdevice)
            depth_buffer = []

            print("\n" + "="*50)
            print("PRECISION DEPTH ACTIVE")
            print("1. Point crosshair at an object exactly 0.5m away (use tape).")
            print("2. Press 'c' to lock calibration.")
            print("3. Press 'q' to exit.")
            print("="*50 + "\n")

            while True:
                ret, frame = cap.read()
                if not ret: continue

                t_start = time.time()
                raw_depth = depth_inf.run(frame)
                
                # Smoothing
                depth_buffer.append(raw_depth)
                if len(depth_buffer) > 10: depth_buffer.pop(0)
                avg_depth = np.mean(depth_buffer, axis=0)
                
                # Sample center
                h, w = avg_depth.shape
                oy, ox = h // 2, w // 2
                raw_val = avg_depth[oy, ox]
                
                # Calculate meters
                dist_m = abs(scale * raw_val + bias) if calib_set else 0.0

                # Heatmap for context
                d_min, d_max = avg_depth.min(), avg_depth.max()
                depth_vis = ((avg_depth - d_min) / (d_max - d_min) * 255).astype(np.uint8) if d_max > d_min else np.zeros_like(avg_depth).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                depth_display = cv2.resize(depth_color, (frame.shape[1], frame.shape[0]))

                # UI Overlay
                cx, cy = frame.shape[1] // 2, frame.shape[0] // 2
                cv2.drawMarker(frame, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 40, 3)
                
                # Big Number
                header_color = (0, 255, 0) if calib_set else (0, 0, 255)
                header_text = f"DEPTH: {dist_m:.3f}m" if calib_set else "PRESS 'C' TO CALIB (0.5m)"
                cv2.rectangle(frame, (0,0), (640, 80), (0,0,0), -1)
                cv2.putText(frame, header_text, (20, 55), 0, 1.5, header_color, 3)
                cv2.putText(frame, f"Raw: {raw_val:.3f}", (frame.shape[1]-150, 40), 0, 0.6, (255,255,255), 1)

                cv2.imshow("KRISHI-EYE | Precision Probe", np.hstack((frame, depth_display)))
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): break
                if key == ord('c'):
                    # Calibration Math:
                    # Target = 0.5m
                    # We have Raw Val (e.g. -3.4) and Raw Far (e.g. -6.3)
                    # For simplicity, we adjust the BIAS to make the current RAW exactly 0.5m
                    # 0.5 = abs(scale * raw_val + new_bias)
                    # bias = 0.5 - (scale * raw_val) -> absolute
                    bias = 0.5 - (scale * raw_val)
                    calib_set = True
                    print(f"✅ Calibrated! Bias adjusted to {bias:.3f} for exact 0.5m at center.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
