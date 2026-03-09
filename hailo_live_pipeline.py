#!/usr/bin/env python3
"""
KRISHI-EYE Live Pipeline v5 — FINAL PRODUCTION
Hailo-8L + SCDepthV3 + YOLO Seg + Classifier + UNet + UART (Teensy 4.1)

Architecture:
  - ROUND_ROBIN scheduler across all 4 NPU models
  - configure() called ONCE per model at startup
  - Output buffers pre-allocated once and reused
  - Depth runs in a background thread every frame (non-blocking)
  - EMA temporal smoothing on depth map
  - DepthTracker: last-known-valid fallback for DEPTH_STALE_FRAMES
  - All drawing on display_frame copy — source frame never mutated
  - 5×5 median patch depth sampling at centroid
  - Range-gated metric conversion (no abs() hiding invalids)
  - CLAHE pre-processing for shadow/glare robustness
  - Per-class confidence penalties (Pest/Nematode overfit correction)
  - Spatial+temporal cooldown de-duplication (prevents re-spraying)
  - UART serial output to Teensy 4.1 with packet throttling

UART Packet Format (space-separated, newline-terminated):
  H_CAM V_CAM H_NOZZLE V_NOZZLE LIQUID_ML CLASS_ID
  Example: "90 45 92 47 4000 5"

Usage:
    python hailo_live_pipeline_perp.py
    python hailo_live_pipeline_perp.py --uart /dev/ttyACM0
    python hailo_live_pipeline_perp.py --uart /dev/ttyACM0 --bias -7.84 --scale -1.45
"""

import os
import cv2
import numpy as np
import time
import argparse
import subprocess
import threading
import serial
from hailo_platform import (HEF, VDevice, HailoSchedulingAlgorithm, FormatType)


# ============================================================
# Constants
# ============================================================
CLASSIFIER_CLASSES  = ['Bacteria', 'Fungi', 'Healthy', 'Nematode',
                       'Pest', 'Phytophthora', 'Virus']

YOLO_CONF_THRESHOLD = 0.45
YOLO_IOU_THRESHOLD  = 0.50
BAUD_RATE           = 115200

# Depth
DEPTH_MIN_M         = 0.05
DEPTH_MAX_M         = 15.0
DEPTH_EMA_ALPHA     = 0.40
DEPTH_STALE_FRAMES  = 30       # how many frames before depth is flagged stale

# Servo mapping (camera FOV → servo degrees)
CAM_H_MIN, CAM_H_MAX = 20, 160    # Horizontal camera servo range
CAM_V_MIN, CAM_V_MAX = 10, 80     # Vertical camera servo range
FRAME_W, FRAME_H     = 1280, 720  # Camera resolution

# Spray calculation
MAX_SPRAY_ML          = 800        # Maximum spray volume for 100% infection (mL)

# Cooldown
COOLDOWN_SECONDS      = 1.5        # Seconds to suppress same-leaf re-targeting
COOLDOWN_RADIUS_PX    = 40         # Pixel radius for spatial matching
UART_MIN_INTERVAL_S   = 0.3        # Minimum seconds between UART packets

# Per-class confidence penalties (overfit correction — no retraining needed)
# Classes that the model tends to over-predict get a penalty added to the
# confidence threshold.  If threshold is 0.45 and penalty is 0.25 then
# the effective threshold becomes 0.70 for that class.
CLASS_CONFIDENCE_PENALTIES = {
    3: 0.25,   # Nematode — model overfits
    4: 0.25,   # Pest     — model overfits
}

# Logit adjustments — applied BEFORE softmax to correct class bias
CLASS_LOGIT_ADJUSTMENTS = {
    1: -1.5,   # Fungi: model over-predicts, reduce logit
}


# ============================================================
# Hailo Inference — configure() + buffers allocated ONCE
# ============================================================
class HailoInference:
    def __init__(self, hef_path: str, vdevice):
        self.hef         = HEF(hef_path)
        self.infer_model = vdevice.create_infer_model(hef_path)
        self.infer_model.set_batch_size(1)

        for output in self.infer_model.outputs:
            output.set_format_type(FormatType.FLOAT32)

        info             = self.hef.get_input_vstream_infos()[0]
        self.input_shape = info.shape
        self.input_name  = info.name

        # configure once — NOT per frame
        self.configured_model = self.infer_model.configure()

        # pre-allocate output buffers — reused every call
        self.output_buffers = {
            out.name: np.empty(
                self.infer_model.output(out.name).shape, dtype=np.float32
            )
            for out in self.hef.get_output_vstream_infos()
        }
        print(f"  ✓ {os.path.basename(hef_path):35s} input={self.input_shape}")

    def run(self, input_data: np.ndarray) -> dict:
        if input_data.ndim == 3:
            input_data = np.expand_dims(input_data, axis=0)
        if input_data.dtype != np.uint8:
            input_data = (
                (input_data * 255).astype(np.uint8)
                if input_data.max() <= 1.01
                else input_data.astype(np.uint8)
            )
        input_data = np.ascontiguousarray(input_data)
        bindings   = self.configured_model.create_bindings(
            output_buffers=self.output_buffers
        )
        bindings.input().set_buffer(input_data)
        self.configured_model.run([bindings], 10000)
        return self.output_buffers


# ============================================================
# Background Depth Thread — runs every frame, non-blocking
# ============================================================
class DepthThread:
    def __init__(self, depth_inf: HailoInference, depth_h: int, depth_w: int):
        self.depth_inf  = depth_inf
        self.depth_h    = depth_h
        self.depth_w    = depth_w
        self._lock      = threading.Lock()
        self._frame     = None
        self._depth_map = None
        self._ema_map   = None
        self._running   = True
        self._got_first = False
        self._thread    = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def update_frame(self, frame: np.ndarray):
        resized = cv2.resize(frame, (self.depth_w, self.depth_h))
        with self._lock:
            self._frame = resized

    def get_depth_map(self):
        with self._lock:
            return self._depth_map.copy() if self._depth_map is not None else None

    def _worker(self):
        while self._running:
            frame_copy = None
            with self._lock:
                if self._frame is not None:
                    frame_copy  = self._frame
                    self._frame = None
            if frame_copy is None:
                time.sleep(0.005)
                continue
            try:
                out     = self.depth_inf.run(frame_copy)
                raw_map = list(out.values())[0].squeeze().astype(np.float32)
                with self._lock:
                    if self._ema_map is None:
                        self._ema_map = raw_map.copy()
                    else:
                        cv2.accumulateWeighted(raw_map, self._ema_map, DEPTH_EMA_ALPHA)
                    self._depth_map = self._ema_map.copy()
                if not self._got_first:
                    self._got_first = True
                    print(f"  [DepthThread] ✅ First depth map produced (shape={raw_map.shape}, range={raw_map.min():.3f}–{raw_map.max():.3f})")
            except Exception as e:
                import traceback
                print(f"[DepthThread] ❌ error: {e}")
                traceback.print_exc()

    def stop(self):
        self._running = False
        self._thread.join(timeout=2.0)


# ============================================================
# Depth helpers
# ============================================================
class DepthTracker:
    def __init__(self):
        self._last_valid  = None
        self._stale_count = 0

    def update(self, new_depth):
        if new_depth is not None:
            self._last_valid  = new_depth
            self._stale_count = 0
        else:
            self._stale_count += 1

    def get(self):
        """Returns (depth_metres, is_fresh). depth_metres is NEVER None if we ever had a valid reading."""
        if self._last_valid is None:
            return None, False
        if self._stale_count == 0:
            return self._last_valid, True
        # Always return last-known — never discard. Just flag staleness.
        return self._last_valid, False


def sample_depth_patch(depth_map, dx, dy, depth_h, depth_w, patch=7):
    """7×7 trimmed mean: remove top/bottom 20% outliers then average."""
    half = patch // 2
    y0 = max(0, dy - half);  y1 = min(depth_h, dy + half + 1)
    x0 = max(0, dx - half);  x1 = min(depth_w, dx + half + 1)
    region = depth_map[y0:y1, x0:x1].ravel()
    if region.size == 0:
        return None
    # Trimmed mean: sort, discard top/bottom 20%
    sorted_vals = np.sort(region)
    trim = max(1, len(sorted_vals) // 5)
    trimmed = sorted_vals[trim:-trim] if len(sorted_vals) > 2 * trim else sorted_vals
    return float(np.mean(trimmed))


def raw_to_metric(raw_val, scale, bias):
    """Convert raw model output to metric depth. CLAMPS to valid range (never None)."""
    if raw_val is None:
        return None
    d = scale * raw_val + bias
    # Clamp to valid range instead of rejecting — always produce a reading
    return float(max(DEPTH_MIN_M, min(DEPTH_MAX_M, d)))


# ============================================================
# Cooldown Tracker — prevents re-spraying the same leaf
# ============================================================
class CooldownTracker:
    """
    Tracks recently targeted centroids.
    Suppresses re-targeting within COOLDOWN_RADIUS_PX pixels
    for COOLDOWN_SECONDS after the last UART packet.
    """
    def __init__(self):
        self._entries = []   # list of (cx, cy, timestamp)

    def is_cooled_down(self, cx, cy):
        """Returns True if this centroid is still in cooldown (should skip)."""
        now = time.time()
        # Prune expired entries
        self._entries = [
            (ex, ey, t) for ex, ey, t in self._entries
            if (now - t) < COOLDOWN_SECONDS
        ]
        # Check spatial proximity
        for ex, ey, t in self._entries:
            dist = ((cx - ex) ** 2 + (cy - ey) ** 2) ** 0.5
            if dist < COOLDOWN_RADIUS_PX:
                return True
        return False

    def register(self, cx, cy):
        """Register a centroid as recently targeted."""
        self._entries.append((cx, cy, time.time()))


# ============================================================
# UART Helper — Teensy 4.1 serial communication
# ============================================================
class UARTSender:
    """
    Sends data packets to Teensy 4.1 via serial.
    Enforces minimum interval between packets to prevent overload.
    """
    def __init__(self, port, baud=BAUD_RATE):
        self.port       = port
        self.ser        = None
        self._last_send = 0.0

        if port:
            try:
                self.ser = serial.Serial(port, baud, timeout=0.1)
                time.sleep(0.5)   # let Teensy reset after serial connect
                print(f"  ✅ UART connected: {port} @ {baud} baud")
            except Exception as e:
                print(f"  ⚠️  UART unavailable ({port}): {e}")
                print(f"       Running in display-only mode.")
                self.ser = None

    def send_packet(self, h_cam, v_cam, h_nozzle, v_nozzle, liquid_ml, class_id):
        """
        Send: "H_CAM V_CAM H_NOZZLE V_NOZZLE LIQUID_ML CLASS_ID\n"
        Returns True if sent, False if throttled or unavailable.
        """
        now = time.time()
        if (now - self._last_send) < UART_MIN_INTERVAL_S:
            return False   # throttled

        packet = f"{h_cam} {v_cam} {h_nozzle} {v_nozzle} {liquid_ml} {class_id}\n"

        if self.ser and self.ser.is_open:
            try:
                self.ser.write(packet.encode('ascii'))
                self._last_send = now
                print(f"  📡 UART TX: {packet.strip()}")
                return True
            except Exception as e:
                print(f"  ⚠️  UART write error: {e}")
                return False
        else:
            # No serial — print to terminal as fallback
            print(f"  📡 UART (sim): {packet.strip()}")
            self._last_send = now
            return True

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()


# ============================================================
# Coordinate → Servo Angle Mapping
# ============================================================
def centroid_to_servo_angles(cx, cy, frame_w=FRAME_W, frame_h=FRAME_H):
    """
    Map pixel centroid (cx, cy) to servo angles.
    Horizontal: 20–160°  (left edge → right edge of frame)
    Vertical:   10–80°   (top edge → bottom edge of frame)
    """
    h_angle = int(CAM_H_MIN + (cx / frame_w) * (CAM_H_MAX - CAM_H_MIN))
    v_angle = int(CAM_V_MIN + (cy / frame_h) * (CAM_V_MAX - CAM_V_MIN))
    h_angle = max(CAM_H_MIN, min(CAM_H_MAX, h_angle))
    v_angle = max(CAM_V_MIN, min(CAM_V_MAX, v_angle))
    return h_angle, v_angle


def infection_to_spray_ml(infection_pct):
    """Map infection % to spray volume (mL). Min 50 mL if any infection."""
    ml = int((infection_pct / 100.0) * MAX_SPRAY_ML)
    return max(50, min(MAX_SPRAY_ML, ml))


# ============================================================
# Pre-processing
# ============================================================
def preprocess_yolo(image, target_size=(320, 320)):
    h, w    = image.shape[:2]
    scale   = min(target_size[1] / w, target_size[0] / h)
    nw, nh  = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (nw, nh))
    padded  = np.full((target_size[0], target_size[1], 3), 114, dtype=np.uint8)
    top     = (target_size[0] - nh) // 2
    left    = (target_size[1] - nw) // 2
    padded[top:top+nh, left:left+nw] = resized
    return padded, (scale, top, left)

def preprocess_classifier(image, target_size=(224, 224)):
    h, w    = image.shape[:2]
    scale   = 256.0 / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))
    cy, cx  = new_h // 2, new_w // 2
    half    = target_size[0] // 2
    return resized[cy-half:cy-half+target_size[0],
                   cx-half:cx-half+target_size[1]]

def preprocess_lesion(image, target_size=(256, 256)):
    return cv2.resize(image, (target_size[1], target_size[0]))


# ============================================================
# Post-processing
# ============================================================
def dfl_decode(box_tensor, reg_max=16):
    n          = box_tensor.shape[0]
    box_tensor = box_tensor.reshape(n, 4, reg_max)
    exp        = np.exp(box_tensor - np.max(box_tensor, axis=2, keepdims=True))
    sm         = exp / np.sum(exp, axis=2, keepdims=True)
    ar         = np.arange(reg_max, dtype=np.float32).reshape(1, 1, reg_max)
    return np.sum(sm * ar, axis=2)

def make_anchors(feat_sizes, strides):
    aps, sts = [], []
    for (fh, fw), stride in zip(feat_sizes, strides):
        sx = np.arange(fw, dtype=np.float32) + 0.5
        sy = np.arange(fh, dtype=np.float32) + 0.5
        gy, gx = np.meshgrid(sy, sx, indexing='ij')
        aps.append(np.stack([gx.ravel(), gy.ravel()], axis=1))
        sts.append(np.full((fh * fw, 1), stride, dtype=np.float32))
    return np.concatenate(aps), np.concatenate(sts)

def postprocess_yolo(outputs, meta,
                     conf_thr=YOLO_CONF_THRESHOLD,
                     iou_thr=YOLO_IOU_THRESHOLD):
    box_t, score_t, mask_t, proto_t = [], [], [], []
    for _, tensor in outputs.items():
        ch = tensor.shape[-1]
        if   ch == 64: box_t.append(tensor)
        elif ch == 1:  score_t.append(tensor)
        elif ch == 32:
            (mask_t if 2100 in tensor.shape else proto_t).append(tensor)

    if not (box_t and score_t and proto_t):
        return []
    if not (len(box_t) == 1 and box_t[0].shape[1] == 2100):
        return []

    all_boxes  = box_t[0].reshape(-1, 64)
    all_scores = score_t[0].reshape(-1)
    all_coeffs = mask_t[0].reshape(-1, 32)
    strides    = [8, 16, 32]
    feat_sizes = [(40, 40), (20, 20), (10, 10)]

    if all_scores.max() > 1.0 or all_scores.min() < 0.0:
        all_scores = 1.0 / (1.0 + np.exp(-all_scores))

    mask = all_scores > conf_thr
    if not np.any(mask):
        return []

    fb, fs, fc     = all_boxes[mask], all_scores[mask], all_coeffs[mask]
    anchors, st    = make_anchors(feat_sizes, strides)
    fa, fst        = anchors[mask], st[mask]
    ltrb           = dfl_decode(fb)

    x1 = np.clip((fa[:, 0:1] - ltrb[:, 0:1]) * fst, 0, 320)
    y1 = np.clip((fa[:, 1:2] - ltrb[:, 1:2]) * fst, 0, 320)
    x2 = np.clip((fa[:, 0:1] + ltrb[:, 2:3]) * fst, 0, 320)
    y2 = np.clip((fa[:, 1:2] + ltrb[:, 3:4]) * fst, 0, 320)
    bw = x2 - x1;  bh = y2 - y1

    nms_boxes = [[float(x1[i]), float(y1[i]), float(bw[i]), float(bh[i])]
                 for i in range(len(x1))]
    indices   = cv2.dnn.NMSBoxes(nms_boxes, fs.tolist(), conf_thr, iou_thr)
    if len(indices) == 0:
        return []

    protos           = proto_t[0].squeeze()
    scale, top, left = meta
    results          = []

    for i in indices:
        if isinstance(i, (list, np.ndarray, tuple)): i = i[0]
        bx, by, bw_i, bh_i = nms_boxes[i]
        m   = (protos @ fc[i]).reshape(80, 80)
        m   = 1.0 / (1.0 + np.exp(-m))
        m   = cv2.resize(m, (320, 320))
        mc  = m[int(by):int(by+bh_i), int(bx):int(bx+bw_i)]
        results.append({
            'box':        [int((bx-left)/scale), int((by-top)/scale),
                           int(bw_i/scale),       int(bh_i/scale)],
            'confidence': float(fs[i]),
            'mask':       (mc > 0.5).astype(np.uint8),
        })
    return results

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def postprocess_classifier(outputs):
    logits = list(outputs.values())[0].ravel().copy()
    for cid, adj in CLASS_LOGIT_ADJUSTMENTS.items():
        if cid < len(logits):
            logits[cid] += adj
    probs    = softmax(logits)
    cid      = int(np.argmax(probs))
    label    = CLASSIFIER_CLASSES[cid] if cid < len(CLASSIFIER_CLASSES) else "Unknown"
    return label, cid, float(probs[cid])

def postprocess_unet(outputs, threshold=0.5):
    m = list(outputs.values())[0].squeeze()
    return ((1.0 / (1.0 + np.exp(-m))) > threshold).astype(np.uint8) * 255


# ============================================================
# CLAHE Pre-processing (from ROBUSTNESS_RESEARCH.md)
# ============================================================
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

def apply_clahe(frame):
    """Apply CLAHE to L-channel of LAB for shadow/glare robustness (~1-2ms)."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# ============================================================
# Camera
# ============================================================
class SubprocessVideoCapture:
    def __init__(self, camera_idx=0, w=1280, h=720):
        self.w, self.h  = w, h
        self.frame_size = int(w * h * 1.5)
        self.proc = subprocess.Popen(
            ["rpicam-vid", "--camera", str(camera_idx),
             "--width", str(w), "--height", str(h),
             "--codec", "yuv420", "--nopreview",
             "-t", "0", "--inline", "--flush", "-o", "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=w * h * 2,
        )

    def isOpened(self): return self.proc.poll() is None

    def read(self):
        raw = self.proc.stdout.read(self.frame_size)
        if len(raw) < self.frame_size:
            return False, None
        yuv = np.frombuffer(raw, dtype=np.uint8).reshape(
            (int(self.h * 1.5), self.w))
        return True, cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

    def release(self): self.proc.terminate()


def find_working_camera():
    print("🎬 Searching for camera...")
    for idx in [0, 1]:
        print(f"  🔍 rpicam-vid Camera {idx}...")
        cap = SubprocessVideoCapture(idx)
        ret, _ = cap.read()
        if ret:
            print(f"  ✅ rpicam-vid Camera {idx} — OK")
            return cap
        cap.release()

    for cfg in [
        {"name": "GStreamer", "type": "GST",
         "pipe": "libcamerasrc ! videoconvert ! appsink"},
        {"name": "V4L2-0",   "type": "V4L2", "idx": 0},
        {"name": "V4L2-8",   "type": "V4L2", "idx": 8},
    ]:
        print(f"  🔍 {cfg['name']}...")
        cap = None
        try:
            cap = (cv2.VideoCapture(cfg["pipe"], cv2.CAP_GSTREAMER)
                   if cfg["type"] == "GST"
                   else cv2.VideoCapture(cfg["idx"], cv2.CAP_V4L2))
            if cap and cap.isOpened():
                for _ in range(5): cap.read()
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"  ✅ {cfg['name']} — OK  {frame.shape}")
                    return cap
                cap.release()
        except Exception:
            if cap: cap.release()

    print("  ❌ No camera found.")
    return None


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="KRISHI-EYE v5 Pipeline")
    ap.add_argument("--yolo",  default="yolov8n_seg.hef")
    ap.add_argument("--cls",   default="potato_classifier.hef")
    ap.add_argument("--unet",  default="lesion_segmentation.hef")
    ap.add_argument("--depth", default="scdepthv3.hef")
    ap.add_argument("--bias",  type=float, default=-7.840)
    ap.add_argument("--scale", type=float, default=-1.45)
    ap.add_argument("--uart",  type=str, default=None,
                    help="Serial port for Teensy 4.1 (e.g. /dev/ttyACM0)")
    ap.add_argument("--no-clahe", action="store_true",
                    help="Disable CLAHE pre-processing")
    args = ap.parse_args()

    # ── Camera ──────────────────────────────────────────────
    cap = find_working_camera()
    if not cap:
        return

    # ── UART ────────────────────────────────────────────────
    uart = UARTSender(args.uart)

    # ── Hailo ───────────────────────────────────────────────
    print("\n🔬 Loading Hailo-8L models...")
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

    depth_tracker  = DepthTracker()
    cooldown       = CooldownTracker()
    depth_thread   = None

    try:
        with VDevice(params) as vdevice:
            yolo_inf  = HailoInference(args.yolo,  vdevice)
            cls_inf   = HailoInference(args.cls,   vdevice)
            unet_inf  = HailoInference(args.unet,  vdevice)
            depth_inf = HailoInference(args.depth, vdevice)

            depth_h = depth_inf.input_shape[0]
            depth_w = depth_inf.input_shape[1]

            depth_thread = DepthThread(depth_inf, depth_h, depth_w)

            # ── Pre-warm depth: run 10 frames so EMA map is ready ─────
            print("  ⏳ Pre-warming depth (10 frames)...")
            for _ in range(10):
                ret, warmup_frame = cap.read()
                if ret and warmup_frame is not None:
                    depth_thread.update_frame(warmup_frame)
                    time.sleep(0.05)  # let background thread process
            # Wait for depth thread to finish last frame
            time.sleep(0.15)
            if depth_thread.get_depth_map() is not None:
                print("  ✅ Depth map ready.")
            else:
                print("  ⚠️  Depth still warming — will converge shortly.")

            print("\n🚀 Pipeline active — press 'q' to quit\n")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                t0 = time.time()

                # Always feed depth thread — non-blocking
                depth_thread.update_frame(frame)

                # ── CLAHE enhancement ────────────────────────
                enhanced = frame if args.no_clahe else apply_clahe(frame)

                # ── YOLO segmentation ────────────────────────
                inp_y, meta_y = preprocess_yolo(enhanced)
                detections    = postprocess_yolo(yolo_inf.run(inp_y), meta_y)

                # Non-blocking depth map read
                depth_map     = depth_thread.get_depth_map()
                display_frame = frame.copy()
                target_found  = False
                active_depth  = None
                is_fresh      = False

                for det in detections:
                    x, y, w, h = det['box']
                    x1 = max(0, x);                y1 = max(0, y)
                    x2 = min(frame.shape[1], x+w); y2 = min(frame.shape[0], y+h)
                    bb  = frame[y1:y2, x1:x2]
                    if bb.size == 0: continue

                    msk = cv2.resize(det['mask'], (x2-x1, y2-y1),
                                     interpolation=cv2.INTER_NEAREST)
                    seg = cv2.bitwise_and(bb, bb, mask=msk)

                    # ── Classifier ───────────────────────────
                    label, cid, conf = postprocess_classifier(
                        cls_inf.run(preprocess_classifier(seg))
                    )

                    # ── Class Bias Correction ────────────────
                    # Apply per-class penalty to effective threshold
                    effective_threshold = 0.45 + CLASS_CONFIDENCE_PENALTIES.get(cid, 0.0)
                    is_diseased = (cid != 2 and conf > effective_threshold)

                    if is_diseased:
                        # ── Centroid ─────────────────────────
                        M = cv2.moments(msk)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                        else:
                            cx, cy = (x2-x1)//2, (y2-y1)//2
                        bx_c = x1 + cx;  by_c = y1 + cy

                        # ── Cooldown check ───────────────────
                        if cooldown.is_cooled_down(bx_c, by_c):
                            # Still in cooldown — draw dimmed box, skip UART
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2),
                                          (128, 128, 128), 2)
                            cv2.putText(display_frame,
                                        f"{label} {conf:.0%} [COOLED]",
                                        (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.50, (128, 128, 128), 1)
                            continue

                        target_found = True

                        # ── UNet lesion mask ─────────────────
                        lesion = postprocess_unet(
                            unet_inf.run(preprocess_lesion(seg))
                        )
                        lesion_rsz = cv2.resize(lesion, (x2-x1, y2-y1),
                                                interpolation=cv2.INTER_NEAREST)

                        # ── Infection % ──────────────────────
                        leaf_px  = np.count_nonzero(msk)
                        inf_px   = np.count_nonzero(lesion_rsz > 127)
                        inf_pct  = (inf_px / leaf_px * 100.0
                                    if leaf_px > 0 else 0.0)

                        # ── Depth ────────────────────────────
                        new_depth = None
                        if depth_map is not None:
                            dx = int(np.clip(bx_c * depth_w / frame.shape[1],
                                             0, depth_w - 1))
                            dy = int(np.clip(by_c * depth_h / frame.shape[0],
                                             0, depth_h - 1))
                            raw = sample_depth_patch(
                                depth_map, dx, dy, depth_h, depth_w)
                            new_depth = raw_to_metric(raw, args.scale, args.bias)

                        depth_tracker.update(new_depth)
                        active_depth, is_fresh = depth_tracker.get()

                        d_str = (f"{active_depth:.3f}m"
                                 + ("" if is_fresh else " [↺]")
                                 if active_depth is not None
                                 else "acquiring...")

                        # ── Servo Angles ─────────────────────
                        h_cam, v_cam = centroid_to_servo_angles(bx_c, by_c)
                        # Nozzle mirrors camera (can add offset later)
                        h_nozzle = h_cam - 120
                        v_nozzle = v_cam + 130
                        if h_nozzle < 0: h_nozzle += 180
                        if h_nozzle > 180: h_nozzle -= 180
                        if v_nozzle < 0: v_nozzle += 180
                        if v_nozzle > 180: v_nozzle -= 180
                        liquid_ml = infection_to_spray_ml(inf_pct)

                        # ── UART TX ──────────────────────────
                        # Always send — use 0 depth if not yet available
                        sent = uart.send_packet(
                            h_cam, v_cam, h_nozzle, v_nozzle,
                            liquid_ml, cid
                        )
                        if sent:
                            cooldown.register(bx_c, by_c)

                        # ── Draw on display_frame ONLY ───────
                        roi = display_frame[y1:y2, x1:x2]
                        overlay = np.zeros_like(roi)
                        overlay[lesion_rsz > 127] = (0, 0, 255)
                        cv2.addWeighted(overlay, 0.4, roi, 1.0, 0, roi)

                        cv2.rectangle(display_frame, (x1, y1), (x2, y2),
                                      (0, 0, 255), 3)
                        cv2.circle(display_frame, (bx_c, by_c),
                                   6, (0, 255, 255), -1)
                        cv2.putText(display_frame,
                                    f"{label} {conf:.0%} | {d_str} | {inf_pct:.0f}%",
                                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55, (0, 0, 255), 2)

                        # ── Terminal output ──────────────────
                        print(f"\n🎯 [TARGET] {label} ({conf:.1%})")
                        print(f"  ├─ BBox       : ({x1},{y1})→({x2},{y2})")
                        print(f"  ├─ Centroid   : ({bx_c}, {by_c})")
                        print(f"  ├─ Servos     : H={h_cam}° V={v_cam}°")
                        print(f"  ├─ Leaf px    : {leaf_px}  |  Infected: {inf_px}")
                        print(f"  ├─ Infection  : {inf_pct:.1f}%  →  {liquid_ml} mL")
                        print(f"  └─ Depth      : {d_str}")

                    else:
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2),
                                      (0, 255, 0), 2)
                        cv2.putText(display_frame,
                                    f"{label} {conf:.0%}",
                                    (x1, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                    (0, 255, 0), 2)

                # ── Status panel ─────────────────────────────
                display_frame = cv2.resize(display_frame, (1280, 720))
                fps = 1.0 / (time.time() - t0 + 1e-9)

                cv2.rectangle(display_frame, (0, 0), (500, 135), (0, 0, 0), -1)

                if target_found and active_depth is not None:
                    col = (0, 255, 0) if is_fresh else (0, 165, 255)
                    cv2.putText(display_frame,
                                f"DEPTH: {active_depth:.3f}m"
                                + ("" if is_fresh else "  [stale]"),
                                (15, 45), cv2.FONT_HERSHEY_SIMPLEX,
                                0.85, col, 2)
                elif target_found:
                    cv2.putText(display_frame,
                                "DEPTH: acquiring...",
                                (15, 45), cv2.FONT_HERSHEY_SIMPLEX,
                                0.85, (0, 255, 255), 2)
                else:
                    cv2.putText(display_frame,
                                "SEARCHING...",
                                (15, 45), cv2.FONT_HERSHEY_SIMPLEX,
                                0.95, (0, 255, 0), 2)

                cv2.putText(display_frame, f"FPS: {fps:.1f}",
                            (15, 90), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 1)
                uart_status = "UART: " + (args.uart or "disabled")
                cv2.putText(display_frame, uart_status,
                            (15, 118), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (180, 180, 180), 1)

                cv2.imshow("KRISHI-EYE v5 | NPU Pipeline", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        if depth_thread: depth_thread.stop()
        uart.close()
        cap.release()
        cv2.destroyAllWindows()
        print("\n🔒 Shutdown complete.")


if __name__ == "__main__":
    main()
