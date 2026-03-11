#!/usr/bin/env python3
"""
KRISHI-EYE Live Pipeline: Hailo-8L + Stereo Depth + UART
Uses HailoRT GLOBAL SCHEDULER to fix "Stream Not Activated" error.

This script sends raw uint8 [0-255] images to the NPU.
Normalization is baked into the HEF models.

Usage (on RPi 5):
    source venv/bin/activate
    python legacy_hailo_live_pipeline.py
"""
import os
import cv2
import numpy as np
import time
import serial
import struct
import argparse
from pathlib import Path

from hailo_platform import (HEF, VDevice, HailoSchedulingAlgorithm, FormatType)

try:
    from stereo_calib.deploy_depth import StereoDepth
except ImportError:
    import sys
    sys.path.append(os.path.join(os.getcwd(), 'stereo_calib'))
    from deploy_depth import StereoDepth

# ============================================================
# Constants
# ============================================================
CLASSIFIER_CLASSES = ['Bacteria', 'Fungi', 'Healthy', 'Nematode',
                      'Pest', 'Phytophthora', 'Virus']

YOLO_CONF_THRESHOLD = 0.2
YOLO_IOU_THRESHOLD = 0.5

UART_PORT = "/dev/ttyS0"
BAUD_RATE = 115200

# ============================================================
# Hailo Inference Helper (Scheduler Compatible)
# ============================================================
class HailoInference:
    def __init__(self, hef_path, vdevice):
        self.hef = HEF(hef_path)
        self.infer_model = vdevice.create_infer_model(hef_path)
        self.infer_model.set_batch_size(1)
        
        # Output format FLOAT32 for easy post-processing.
        for output in self.infer_model.outputs:
            output.set_format_type(FormatType.FLOAT32)
            
        input_info = self.hef.get_input_vstream_infos()[0]
        self.input_shape = input_info.shape
        self.input_name = input_info.name
        print(f"  Loaded: {os.path.basename(hef_path)} | Input: {self.input_shape}")

    def run_scheduled(self, input_data):
        """Run synchronous inference using the Global Scheduler."""
        # input_data should be uint8 NHWC
        input_dict = {self.input_name: np.expand_dims(input_data, axis=0).astype(np.uint8)}
        return self.infer_model.run(input_dict)


# ============================================================
# Pre-Processing
# ============================================================
def preprocess_yolo(image, target_size=(320, 320)):
    h, w = image.shape[:2]
    scale = min(target_size[1] / w, target_size[0] / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (nw, nh))
    padded = np.full((target_size[0], target_size[1], 3), 114, dtype=np.uint8)
    top, left = (target_size[0]-nh)//2, (target_size[1]-nw)//2
    padded[top:top+nh, left:left+nw, :] = resized
    return padded, (scale, top, left)

def preprocess_classifier(image, target_size=(224, 224)):
    h, w = image.shape[:2]
    scale = 256.0 / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))
    cy, cx = new_h // 2, new_w // 2
    half = target_size[0] // 2
    return resized[cy-half:cy-half+target_size[0], cx-half:cx-half+target_size[1]]

def preprocess_lesion(image, target_size=(256, 256)):
    return cv2.resize(image, (target_size[1], target_size[0]))

# ============================================================
# Post-Processing
# ============================================================
def dfl_decode(box_tensor, reg_max=16):
    n = box_tensor.shape[0]
    box_tensor = box_tensor.reshape(n, 4, reg_max)
    exp = np.exp(box_tensor - np.max(box_tensor, axis=2, keepdims=True))
    softmax = exp / np.sum(exp, axis=2, keepdims=True)
    arange = np.arange(reg_max, dtype=np.float32).reshape(1, 1, reg_max)
    return np.sum(softmax * arange, axis=2)

def make_anchors(feat_sizes, strides):
    anchor_points, stride_tensor = [], []
    for (fh, fw), stride in zip(feat_sizes, strides):
        sx = np.arange(fw, dtype=np.float32) + 0.5
        sy = np.arange(fh, dtype=np.float32) + 0.5
        gy, gx = np.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(np.stack([gx.ravel(), gy.ravel()], axis=1))
        stride_tensor.append(np.full((fh * fw, 1), stride, dtype=np.float32))
    return np.concatenate(anchor_points, axis=0), np.concatenate(stride_tensor, axis=0)

def postprocess_yolo(outputs, meta, conf_threshold=YOLO_CONF_THRESHOLD, iou_threshold=YOLO_IOU_THRESHOLD):
    box_t, score_t, mask_t = [], [], []
    for name, tensor in outputs.items():
        ch = tensor.shape[-1]
        if ch == 64: box_t.append(tensor)
        elif ch == 1: score_t.append(tensor)
        elif ch == 32: mask_t.append(tensor)

    box_t.sort(key=lambda t: t.shape[1], reverse=True)
    score_t.sort(key=lambda t: t.shape[1], reverse=True)
    mask_t.sort(key=lambda t: t.shape[1], reverse=True)

    if len(box_t) < 3 or len(score_t) < 3: return []

    strides = [8, 16, 32]
    feat_sizes = [(t.shape[1], t.shape[2]) for t in box_t]

    all_boxes = np.concatenate([t.reshape(-1, 64) for t in box_t], axis=0)
    all_scores = 1 / (1 + np.exp(-np.concatenate([t.reshape(-1) for t in score_t], axis=0)))
    all_masks = np.concatenate([t.reshape(-1, 32) for t in mask_t], axis=0)

    conf_mask = all_scores > conf_threshold
    if not np.any(conf_mask): return []

    fb, fs, fm = all_boxes[conf_mask], all_scores[conf_mask], all_masks[conf_mask]
    anchors, st = make_anchors(feat_sizes, strides)
    fa, fst = anchors[conf_mask], st[conf_mask]

    ltrb = dfl_decode(fb)
    x1 = (fa[:, 0:1] - ltrb[:, 0:1]) * fst
    y1 = (fa[:, 1:2] - ltrb[:, 1:2]) * fst
    x2 = (fa[:, 0:1] + ltrb[:, 2:3]) * fst
    y2 = (fa[:, 1:2] + ltrb[:, 3:4]) * fst

    nm_boxes = [[float(x1[i]), float(y1[i]), float(x2[i]-x1[i]), float(y2[i]-y1[i])] for i in range(len(x1))]
    indices = cv2.dnn.NMSBoxes(nm_boxes, fs.tolist(), conf_threshold, iou_threshold)

    results = []
    scale, top, left = meta
    for i in indices:
        if isinstance(i, (list, np.ndarray)): i = i[0]
        bx, by, bw, bh = nm_boxes[i]
        results.append({
            'box': [int((bx-left)/scale), int((by-top)/scale), int(bw/scale), int(bh/scale)],
            'confidence': float(fs[i]), 'coeffs': fm[i]
        })
    return results

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def postprocess_classifier(outputs):
    logits = list(outputs.values())[0].ravel()
    probs = softmax(logits)
    class_id = int(np.argmax(probs))
    label = CLASSIFIER_CLASSES[class_id] if class_id < len(CLASSIFIER_CLASSES) else "Unknown"
    return label, class_id, float(probs[class_id])

def postprocess_unet(outputs, threshold=0.5):
    mask = list(outputs.values())[0].squeeze()
    mask_sigmoid = 1.0 / (1.0 + np.exp(-mask))
    return (mask_sigmoid > threshold).astype(np.uint8) * 255

def send_uart_packet(ser, x, y, z, duration, valve_id):
    header = 0xAA
    payload = struct.pack("<ffffB", x, y, z, duration, valve_id)
    checksum = 0
    for b in payload: checksum ^= b
    ser.write(struct.pack("<BB", header, 17) + payload + struct.pack("<B", checksum))


# ============================================================
# Main Pipeline
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo", default="HAILO/output/yolov8n_seg.hef")
    parser.add_argument("--cls", default="HAILO/output/potato_classifier.hef")
    parser.add_argument("--unet", default="HAILO/output/lesion_segmentation.hef")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    try:
        ser = serial.Serial(UART_PORT, BAUD_RATE, timeout=0.1)
    except Exception as e:
        print(f"UART Warning: {e}"); ser = None

    # Enable Global Scheduler
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

    with VDevice(params) as vdevice:
        print("🚀 Loading models onto Hailo-8L with Global Scheduler...")
        yolo_inf = HailoInference(args.yolo, vdevice)
        cls_inf = HailoInference(args.cls, vdevice)
        unet_inf = HailoInference(args.unet, vdevice)

        calib_dir = Path("stereo_calib")
        stereo = StereoDepth(calib_dir, image_size=(3280, 2464))

        cap = cv2.VideoCapture(args.device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 6560)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2464)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            start_time = time.time()
            left_raw = frame[:, :3280]
            right_raw = frame[:, 3280:]
            left_rect, _, _, depth_map = stereo.compute(left_raw, right_raw)

            input_y, meta_y = preprocess_yolo(left_rect)
            yolo_out = yolo_inf.run_scheduled(input_y)
            detections = postprocess_yolo(yolo_out, meta_y)

            for det in detections:
                x, y, w, h = det['box']
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(left_rect.shape[1], x+w), min(left_rect.shape[0], y+h)

                z_roi = depth_map[y1:y2, x1:x2]
                z = np.median(z_roi[z_roi > 0]) if np.any(z_roi > 0) else 0
                if z == 0 or z > 150: continue

                leaf_crop = left_rect[y1:y2, x1:x2]
                cls_out = cls_inf.run_scheduled(preprocess_classifier(leaf_crop))
                label, class_id, conf = postprocess_classifier(cls_out)

                if class_id != 2: # 2 == Healthy
                    unet_out = unet_inf.run_scheduled(preprocess_lesion(leaf_crop))
                    mask = postprocess_unet(unet_out)
                    M = cv2.moments(mask)
                    if M["m00"] > 0:
                        cx_l = int(M["m10"] / M["m00"])
                        cy_l = int(M["m01"] / M["m00"])
                        tx = x1 + int(cx_l * leaf_crop.shape[1] / 256)
                        ty = y1 + int(cy_l * leaf_crop.shape[0] / 256)
                        rx = (tx - 1640) * z / 3000
                        ry = (ty - 1232) * z / 3000
                        if ser: send_uart_packet(ser, rx, ry, z, 0.5, class_id)
                        cv2.circle(left_rect, (tx, ty), 10, (0, 0, 255), -1)

                cv2.rectangle(left_rect, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(left_rect, f"{label}", (x1, y1-10), 0, 0.6, (255, 0, 0), 2)

            fps = 1 / (time.time() - start_time)
            cv2.putText(left_rect, f"FPS: {fps:.1f}", (40, 40), 0, 1, (255, 255, 255), 2)
            cv2.imshow("KRISHI-EYE Live", cv2.resize(left_rect, (1280, 720)))
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    if ser: ser.close()
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
