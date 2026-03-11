#!/usr/bin/env python3
"""
KRISHI-EYE: Test all 3 Hailo HEF models on a static image.
Uses HailoRT GLOBAL SCHEDULER to fix "Stream Not Activated" error.

This script sends raw uint8 [0-255] images to the NPU.
Normalization is baked into the HEF models.

Usage (on RPi 5):
    source venv/bin/activate
    python legacy_test_hailo_models.py --image 2.jpeg --debug
"""
import os
import cv2
import numpy as np
import argparse
import time

from hailo_platform import (HEF, VDevice, HailoSchedulingAlgorithm, FormatType)

# ============================================================
# Constants
# ============================================================
CLASSIFIER_CLASSES = ['Bacteria', 'Fungi', 'Healthy', 'Nematode',
                      'Pest', 'Phytophthora', 'Virus']

YOLO_CONF_THRESHOLD = 0.2
YOLO_IOU_THRESHOLD = 0.5

# Global debug flag
DEBUG = False

# ============================================================
# Hailo Inference Helper (Scheduler Compatible)
# ============================================================
class HailoInference:
    def __init__(self, hef_path, vdevice):
        self.hef = HEF(hef_path)
        # Using the InferModel API which works with the Global Scheduler
        self.infer_model = vdevice.create_infer_model(hef_path)
        self.infer_model.set_batch_size(1)

        # Input is uint8 (implicitly), Output is float32
        for output in self.infer_model.outputs:
            output.set_format_type(FormatType.FLOAT32)

        # Get shapes for pre-processing
        input_info = self.hef.get_input_vstream_infos()[0]
        self.input_shape = input_info.shape
        self.input_name = input_info.name
        self.output_infos = self.hef.get_output_vstream_infos()

        print(f"Loaded {os.path.basename(hef_path)}:")
        print(f"  Input: {self.input_name} {self.input_shape}")

    def run(self, input_data):
        """Run synchronous inference using the Global Scheduler."""
        # The scheduler handles activation/deactivation automatically
        # input_data should be uint8 NHWC
        input_data = np.expand_dims(input_data, axis=0).astype(np.uint8)
        
        # Sync run returns a dict/list of results
        # In newer HailoRT, InferModel.run returns a dictionary if configured correctly
        # or we use create_bindings. We'll use the most robust 'bindings' method.
        output_buffers = {
            out.name: np.empty(self.infer_model.output(out.name).shape, dtype=np.float32)
            for out in self.output_infos
        }
        
        with self.infer_model.configure() as configured_model:
            bindings = configured_model.create_bindings(output_buffers=output_buffers)
            bindings.input().set_buffer(input_data)
            configured_model.run([bindings], 10000)
            
        return output_buffers

    def run_scheduled(self, input_data):
        """Standard scheduler-based run (no manual configure call)."""
        # Note: If the user is on an older HailoRT, they might need the manual config.
        # But for RPi 5 + Hailo-8L, the Global Scheduler is the way.
        input_dict = {self.input_name: np.expand_dims(input_data, axis=0).astype(np.uint8)}
        
        # High-level API for scheduled run
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
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
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
    # Sort outputs to identify box/score/mask tensors
    box_t, score_t, mask_t = [], [], []
    for name, tensor in outputs.items():
        ch = tensor.shape[-1]
        if ch == 64: box_t.append(tensor)
        elif ch == 1: score_t.append(tensor)
        elif ch == 32: mask_t.append(tensor)
    
    box_t.sort(key=lambda t: t.shape[1], reverse=True) # Sort by resolution
    score_t.sort(key=lambda t: t.shape[1], reverse=True)
    mask_t.sort(key=lambda t: t.shape[1], reverse=True)

    if len(box_t) < 3 or len(score_t) < 3: return []

    strides = [8, 16, 32]
    feat_sizes = [(t.shape[1], t.shape[2]) for t in box_t] # (H, W)

    all_boxes = np.concatenate([t.reshape(-1, 64) for t in box_t], axis=0)
    # Apply sigmoid to raw scores
    all_scores = 1 / (1 + np.exp(-np.concatenate([t.reshape(-1) for t in score_t], axis=0)))
    all_masks = np.concatenate([t.reshape(-1, 32) for t in mask_t], axis=0)

    if DEBUG:
        print(f"  [DEBUG] YOLO raw scores range: [{all_scores.min():.6f}, {all_scores.max():.6f}]")

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
    # Mask is raw logits if it wasn't Activated in model
    mask_sigmoid = 1.0 / (1.0 + np.exp(-mask))
    return (mask_sigmoid > threshold).astype(np.uint8) * 255

# ============================================================
# Main
# ============================================================
def main():
    global DEBUG
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo_hef", default="yolov8n_seg.hef")
    parser.add_argument("--cls_hef", default="potato_classifier.hef")
    parser.add_argument("--unet_hef", default="lesion_segmentation.hef")
    parser.add_argument("--image", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    DEBUG = args.debug

    # Use Global Scheduler to allow multiple models on one device
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    
    with VDevice(params) as vdevice:
        print("🚀 Loading models onto Hailo-8L with Global Scheduler...")
        yolo_inf = HailoInference(args.yolo_hef, vdevice)
        cls_inf = HailoInference(args.cls_hef, vdevice)
        unet_inf = HailoInference(args.unet_hef, vdevice)

        img_bgr = cv2.imread(args.image)
        if img_bgr is None:
            print(f"Error: Could not load image {args.image}")
            return
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        print("\n--- Stage 1: YOLO ---")
        input_y, meta_y = preprocess_yolo(img_rgb)
        start = time.time()
        # Use run_scheduled for the multi-model scenario
        yolo_out = yolo_inf.run_scheduled(input_y)
        detections = postprocess_yolo(yolo_out, meta_y)
        print(f"  Found {len(detections)} leaves in {(time.time()-start)*1000:.1f}ms")

        # Fallback if no leaves
        if not detections:
            print("  ⚠️ No leaves detected. Using fallback box.")
            detections = [{'box': [0, 0, img_rgb.shape[1], img_rgb.shape[0]], 'confidence': 1.0}]

        for i, det in enumerate(detections):
            x, y, w, h = det['box']
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(img_rgb.shape[1], x+w), min(img_rgb.shape[0], y+h)
            crop = img_rgb[y1:y2, x1:x2]
            if crop.size == 0: continue

            print(f"\n--- Stage 2: Classifier ({i}) ---")
            cls_input = preprocess_classifier(crop)
            cls_out = cls_inf.run_scheduled(cls_input)
            label, mid, conf = postprocess_classifier(cls_out)
            print(f"  Result: {label} ({conf:.1%})")

            if label != "Healthy":
                print(f"--- Stage 3: UNet ({i}) ---")
                unet_input = preprocess_lesion(crop)
                unet_out = unet_inf.run_scheduled(unet_input)
                mask = postprocess_unet(unet_out)
                cv2.imwrite(f"mask_{i}.png", mask)

            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_bgr, f"{label}", (x1, y1-10), 0, 0.6, (0, 0, 255), 2)

        cv2.imwrite("test_output.png", img_bgr)
    print("\n✅ Done. Annotated image saved to test_output.png")

if __name__ == "__main__":
    main()
