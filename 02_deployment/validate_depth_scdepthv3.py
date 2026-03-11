#!/usr/bin/env python3
"""
SCDepthV3 Implementation for Hailo-8L
Fixed 256x320 resolution for Hailo Model Zoo HEF.
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
# SCDepthV3 Inference Logic
# ============================================================
class HailoSCDepthInference:
    def __init__(self, hef_path, vdevice):
        self.hef = HEF(hef_path)
        self.infer_model = vdevice.create_infer_model(hef_path)
        self.infer_model.set_batch_size(1)
        
        for output in self.infer_model.outputs:
            output.set_format_type(FormatType.FLOAT32)
            
        # Get input requirements
        input_info = self.hef.get_input_vstream_infos()[0]
        self.target_h, self.target_w = input_info.shape[0], input_info.shape[1]
        print(f"✅ SCDepthV3 Loaded. Input Resolution: {self.target_w}x{self.target_h}")
        # Expected: 320x256 based on buffer error (245760 bytes)

    def run(self, frame):
        # Precise resize to match HEF
        resized = cv2.resize(frame, (self.target_w, self.target_h))
        input_data = np.ascontiguousarray(np.expand_dims(resized, axis=0)).astype(np.uint8)
        
        output_buffers = {
            out.name: np.empty(self.infer_model.output(out.name).shape, dtype=np.float32)
            for out in self.hef.get_output_vstream_infos()
        }
        
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
    parser.add_argument("--calib", default="stereo_calib", help="Calibration folder")
    args = parser.parse_args()

    # 1. Physical Constants
    baseline = 0.06032 
    focal_length = 1587.8
    try:
        mtxL = np.load(os.path.join(args.calib, "cameraMatrixL.npy"))
        focal_length = float(mtxL[0, 0])
    except: pass

    # 2. Camera & Hailo
    cap = SubprocessVideoCapture(w=640, h=480)
    
    print("🚀 Initializing Hailo-8L for SCDepthV3...")
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    
    try:
        with VDevice(params) as vdevice:
            depth_inf = HailoSCDepthInference(args.hef, vdevice)
            depth_buffer = []

            print("🎬 NPU scdepthv3 Active. Press 'q' to exit.")

            while True:
                ret, frame = cap.read()
                if not ret: continue

                t_start = time.time()
                
                # Inference
                raw_depth = depth_inf.run(frame)
                
                # Smoothing
                depth_buffer.append(raw_depth)
                if len(depth_buffer) > 5: depth_buffer.pop(0)
                avg_depth = np.mean(depth_buffer, axis=0)
                
                # Visualization: Robust Min-Max Scaling
                # Since the output is ~ -5.7, we use min-max to see the gradient regardless of absolute value
                depth_min, depth_max = avg_depth.min(), avg_depth.max()
                if depth_max > depth_min:
                    depth_normalized = (avg_depth - depth_min) / (depth_max - depth_min)
                else:
                    depth_normalized = np.zeros_like(avg_depth)
                
                depth_vis = (depth_normalized * 255).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                depth_display = cv2.resize(depth_color, (frame.shape[1], frame.shape[0]))
                
                # Calibration: SCDepthV3 often outputs log-depth or inverse-depth
                cx, cy = frame.shape[1] // 2, frame.shape[0] // 2
                oy, ox = int(cy * avg_depth.shape[0] / frame.shape[0]), int(cx * avg_depth.shape[1] / frame.shape[1])
                depth_val = avg_depth[oy, ox]
                
                # Calibration: SCDepthV3 often outputs log-depth or inverse-depth
                h, w = avg_depth.shape
                
                # Probes: Center, and 4 Quadrants
                probes = {
                    "CENTER": (h // 2, w // 2),
                    "CLOSEST": np.unravel_index(np.argmax(avg_depth), (h, w))
                }
                
                def to_meters(val):
                    # Linear mapping: -3.4 (Near) -> 0.35m, -6.6 (Far) -> 5.0m
                    return abs(-1.45 * val - 4.5)
                
                results = {k: to_meters(avg_depth[p[0], p[1]]) for k, p in probes.items()}
                
                # Overlay
                cx, cy = frame.shape[1] // 2, frame.shape[0] // 2
                cv2.drawMarker(frame, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                
                # Big UI Readout
                cv2.rectangle(frame, (5, 5), (280, 110), (0,0,0), -1)
                cv2.putText(frame, f"CENTER: {results['CENTER']:.2f}m", (15, 35), 0, 0.9, (0, 255, 0), 2)
                cv2.putText(frame, f"CLOSEST: {results['CLOSEST']:.2f}m", (15, 70), 0, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, f"RAW: {avg_depth.max():.2f}", (15, 100), 0, 0.5, (200, 200, 200), 1)
                
                fps = 1 / (time.time() - t_start)
                cv2.putText(frame, f"FPS: {fps:.1f}", (15, frame.shape[0]-20), 0, 0.6, (255, 255, 255), 1)

                # Combine feed and heatmap
                combined = np.hstack((frame, depth_display))
                cv2.imshow("KRISHI-EYE | SCDepthV3 Target Probe", combined)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
