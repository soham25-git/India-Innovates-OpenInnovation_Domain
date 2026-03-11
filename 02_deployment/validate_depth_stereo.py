#!/usr/bin/env python3
"""
Adaptive Stereo Depth Implementation (StereoNet / SGBM Fallback)
Optimized for RPi 5 and Hailo-8L.
"""
import os
import cv2
import numpy as np
import time
import argparse
import subprocess
try:
    from hailo_platform import (HEF, VDevice, HailoSchedulingAlgorithm, FormatType)
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False

# ============================================================
# Dual-Camera Synchronized Capture
# ============================================================
class SubprocessVideoCapture:
    def __init__(self, camera_idx=0, w=1280, h=720):
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

class SyncStereoCapture:
    def __init__(self, w=1280, h=720):
        self.capL = SubprocessVideoCapture(0, w, h)
        self.capR = SubprocessVideoCapture(1, w, h)

    def read(self):
        retL, frameL = self.capL.read()
        retR, frameR = self.capR.read()
        return (retL and retR), frameL, frameR

    def release(self):
        self.capL.release()
        self.capR.release()

# ============================================================
# NPU Inference (StereoNet)
# ============================================================
class HailoStereoNetInference:
    def __init__(self, hef_path, vdevice):
        self.hef = HEF(hef_path)
        self.infer_model = vdevice.create_infer_model(hef_path)
        self.infer_model.set_batch_size(1)
        for output in self.infer_model.outputs:
            output.set_format_type(FormatType.FLOAT32)
        input_infos = self.hef.get_input_vstream_infos()
        self.input_names = [info.name for info in input_infos]
        self.target_shape = input_infos[0].shape
        print(f"✅ NPU StereoNet Active: {self.target_shape[1]}x{self.target_shape[0]}")

    def run(self, left, right):
        l_rs = cv2.resize(left, (self.target_shape[1], self.target_shape[0]))
        r_rs = cv2.resize(right, (self.target_shape[1], self.target_shape[0]))
        l_batch = np.ascontiguousarray(np.expand_dims(l_rs, axis=0)).astype(np.uint8)
        r_batch = np.ascontiguousarray(np.expand_dims(r_rs, axis=0)).astype(np.uint8)
        
        output_buffers = {
            out.name: np.empty(self.infer_model.output(out.name).shape, dtype=np.float32)
            for out in self.hef.get_output_vstream_infos()
        }
        
        with self.infer_model.configure() as configured_model:
            bindings = configured_model.create_bindings(output_buffers=output_buffers)
            # Map input buffers by name
            for name in self.input_names:
                if 'left' in name.lower(): bindings.input(name).set_buffer(l_batch)
                elif 'right' in name.lower(): bindings.input(name).set_buffer(r_batch)
                else:
                    # If naming is generic, assume first is left, second is right
                    if name == self.input_names[0]: bindings.input(name).set_buffer(l_batch)
                    else: bindings.input(name).set_buffer(r_batch)
            
            configured_model.run([bindings], 10000)
        return list(output_buffers.values())[0].squeeze()

# ============================================================
# Main Logic
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hef", default="stereonet.hef", help="Path to StereoNet HEF")
    parser.add_argument("--calib", default="stereo_calib", help="Calibration folder")
    args = parser.parse_args()

    # 1. Calibration Data
    baseline = 0.0603 # Verified 60.3mm
    focal_length = 1587.8 # Verified IMX219 Focal
    try:
        mtxL = np.load(os.path.join(args.calib, "cameraMatrixL.npy"))
        T = np.load(os.path.join(args.calib, "T.npy"))
        focal_length = float(mtxL[0, 0])
        baseline = abs(float(T[0])) if T.size > 0 else 0.0603
        print(f"📊 Calibration: B={baseline*1000:.1f}mm, f={focal_length:.1f}")
    except:
        print("⚠️ Calib files missing, using hardware constants.")

    # 2. Setup NPU Inference (STRICT)
    if not HAILO_AVAILABLE:
        print("❌ CRITICAL: hailo_platform not found. Please install the Hailo software suite.")
        return

    if not os.path.exists(args.hef):
        print(f"❌ CRITICAL: Model file '{args.hef}' not found.")
        print("Please run: hailomz download stereonet")
        return

    print("🚀 Initializing Hailo-8L NPU...")
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    
    with VDevice(params) as vdevice:
        stereo_engine = HailoStereoNetInference(args.hef, vdevice)
        cap_w, cap_h = stereo_engine.target_shape[1], stereo_engine.target_shape[0]

        cap = SyncStereoCapture(cap_w, cap_h)
        disp_buffer = []

        print("🎬 Strictly NPU Stereo Depth Active.")
        print("   Press 'q' to exit.")

        try:
            while True:
                ret, left, right = cap.read()
                if not ret: continue
                
                t0 = time.time()
                disparity = stereo_engine.run(left, right)
                
                # Smoothing
                disp_buffer.append(disparity)
                if len(disp_buffer) > 3: disp_buffer.pop(0)
                smoothed = np.mean(disp_buffer, axis=0)
                
                # Distance at Center
                h, w = smoothed.shape[:2]
                d_val = smoothed[h//2, w//2]
                dist_m = (baseline * focal_length) / max(d_val, 0.1)
                
                # Visuals
                disp_vis = cv2.normalize(smoothed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
                disp_color = cv2.resize(disp_color, (left.shape[1], left.shape[0]))
                
                # Overlay
                cv2.drawMarker(left, (left.shape[1]//2, left.shape[0]//2), (0,255,0), cv2.MARKER_CROSS, 20, 2)
                cv2.putText(left, f"DEPTH: {dist_m:.2f}m", (10, 30), 0, 0.8, (0, 255, 0), 2)
                cv2.putText(left, f"FPS: {1/(time.time()-t0):.1f}", (10, 60), 0, 0.6, (255, 255, 255), 1)

                cv2.imshow("KRISHI-EYE Dual NPU Stereo", np.hstack((left, disp_color)))
                if cv2.waitKey(1) & 0xFF == ord('q'): break
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
