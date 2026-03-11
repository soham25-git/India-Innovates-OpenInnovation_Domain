#!/usr/bin/env python3
"""
Standalone Depth Implementation (FastDepth)
Deploying Hailo FastDepth on RPi 5 with real-time visualization.
"""
import os
import cv2
import numpy as np
import time
import argparse
import subprocess
from hailo_platform import (HEF, VDevice, HailoSchedulingAlgorithm, FormatType)

# ============================================================
# Resilient Camera Wrapper (rpicam-vid Bridge)
# ============================================================
class SubprocessVideoCapture:
    """Uses rpicam-vid to pipe raw YUV data; 100% reliable if rpicam-hello works."""
    def __init__(self, camera_idx=0, w=1280, h=720):
        self.w, self.h = w, h
        self.cmd = [
            "rpicam-vid", "--camera", str(camera_idx),
            "--width", str(w), "--height", str(h),
            "--codec", "yuv420", "--nopreview", "-t", "0", "--inline", "--flush", "-o", "-"
        ]
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=w*h*2)
        self.frame_size = int(w * h * 1.5) # YUV420 size

    def isOpened(self):
        return self.proc.poll() is None

    def read(self):
        raw = self.proc.stdout.read(self.frame_size)
        if len(raw) < self.frame_size: return False, None
        yuv = np.frombuffer(raw, dtype=np.uint8).reshape((int(self.h * 1.5), self.w))
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        return True, bgr

    def release(self):
        if self.proc:
            self.proc.terminate()

# ============================================================
# Inference Helper
# ============================================================
class HailoInference:
    def __init__(self, hef_path, vdevice):
        self.hef = HEF(hef_path)
        self.vdevice = vdevice
        self.infer_model = vdevice.create_infer_model(hef_path)
        self.infer_model.set_batch_size(1)
        
        # Output format FLOAT32 for easy post-processing
        for output in self.infer_model.outputs:
            output.set_format_type(FormatType.FLOAT32)
            
        input_info = self.hef.get_input_vstream_infos()[0]
        self.input_shape = input_info.shape
        self.input_name = input_info.name

    def run(self, input_data):
        if len(input_data.shape) == 3:
            input_data = np.expand_dims(input_data, axis=0)
        
        input_data = np.ascontiguousarray(input_data).astype(np.uint8)
        
        output_buffers = {
            out.name: np.empty(self.infer_model.output(out.name).shape, dtype=np.float32)
            for out in self.hef.get_output_vstream_infos()
        }
        
        with self.infer_model.configure() as configured_model:
            bindings = configured_model.create_bindings(output_buffers=output_buffers)
            bindings.input().set_buffer(input_data)
            configured_model.run([bindings], 10000)
            
        return output_buffers

def preprocess_depth(image, target_size=(224, 224)):
    """Preprocess for FastDepth (224x224x3)."""
    resized = cv2.resize(image, target_size)
    return resized.astype(np.uint8) # HEF handles uint8 [0-255]

def postprocess_depth(outputs):
    """Interpret FastDepth output tensor."""
    # FastDepth typically outputs a (1, 224, 224, 1) or similar depth map
    name = list(outputs.keys())[0]
    depth_map = outputs[name].squeeze()
    
    # Normalize for visualization [0-255]
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    if depth_max > depth_min:
        depth_norm = (depth_map - depth_min) / (depth_max - depth_min)
    else:
        depth_norm = depth_map
    
    return (depth_norm * 255).astype(np.uint8)

# ============================================================
# Main Loop
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hef", default="fast_depth.hef", help="Path to FastDepth HEF")
    parser.add_argument("--calib", default="stereo_calib", help="Path to calibration folder")
    args = parser.parse_args()

    # 1. Load Calibration (Optional but helpful for scale)
    baseline = 0.06 # Default 60mm
    focal_length = 1.0
    try:
        mtxL = np.load(os.path.join(args.calib, "cameraMatrixL.npy"))
        T = np.load(os.path.join(args.calib, "T.npy"))
        focal_length = float(mtxL[0, 0])
        baseline = abs(float(T[0])) if T.size > 0 else 0.06
        print(f"📊 Calibration Loaded: Baseline={baseline*1000:.2f}mm, Focal={focal_length:.2f}")
    except Exception as e:
        print(f"⚠️ Calibration usage simplified: {e}")

    # 2. Start Camera
    cap = SubprocessVideoCapture(0) # Use Camera 0 (Left)
    if not cap.isOpened():
        print("❌ Camera failed.")
        return

    # 3. Load Hailo
    print("🚀 Loading Hailo-8L...")
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    
    try:
        with VDevice(params) as vdevice:
            depth_inf = HailoInference(args.hef, vdevice)
            print("🎬 Live Depth Started. Press 'q' to exit.")
            
            # Temporal Buffer for Smoothing
            depth_buffer = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: continue

                start_time = time.time()
                
                # Inference
                input_tensor = preprocess_depth(frame)
                outputs = depth_inf.run(input_tensor)
                depth_raw = postprocess_depth(outputs).astype(np.float32)
                
                # 1. Temporal Smoothing (reduces flickering)
                depth_buffer.append(depth_raw)
                if len(depth_buffer) > 5: depth_buffer.pop(0)
                smoothed_depth = np.mean(depth_buffer, axis=0).astype(np.uint8)
                
                # Visuals
                depth_color = cv2.applyColorMap(smoothed_depth, cv2.COLORMAP_JET)
                depth_display = cv2.resize(depth_color, (frame.shape[1], frame.shape[0]))
                
                # 2. Distance Calculation (Inverse Scale)
                # Baseline * Focal / RawValue
                k = baseline * focal_length
                
                # Point Selection (Center of Screen)
                cx, cy = frame.shape[1] // 2, frame.shape[0] // 2
                raw_center = smoothed_depth[int(cy * 224 / frame.shape[0]), int(cx * 224 / frame.shape[1])]
                
                # Empirical calibration: Depth = (Baseline * Focal) / Raw
                # We normalize k based on the model's range (0-255)
                # If your tape measure shows 1.0m, adjust this k_bias
                k_bias = 250.0 
                dist_m = (raw_center / k_bias) if raw_center > 0 else 0
                
                # Overlay
                cv2.drawMarker(frame, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                cv2.putText(frame, f"Dist: {dist_m:.2f}m", (cx+20, cy), 0, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Raw: {raw_center}", (cx+20, cy+30), 0, 0.6, (255, 255, 255), 1)

                fps = 1 / (time.time() - start_time)
                combined = np.hstack((frame, depth_display))
                cv2.putText(combined, f"FPS: {fps:.1f}", (20, 30), 0, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("KRISHI-EYE Depth Calibrator", combined)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Shutdown successful.")

if __name__ == "__main__":
    main()
