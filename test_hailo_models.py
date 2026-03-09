#!/usr/bin/env python3
"""
KRISHI-EYE: Test v3 — Full Pipeline on Static Images
Based on test_hailo_models_2.py (proven working) + Depth + UART packet display.

Runs: YOLO-Seg → Classifier → UNet Lesion → Depth (SCDepthV3)
Outputs: Annotated images + terminal UART packets + depth per detection.

Usage:
    python test_hailo_models_3.py --input ./test_images/
    python test_hailo_models_3.py --input ./test_images/ --output_dir ./results/
    python test_hailo_models_3.py --input single_leaf.jpg
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
YOLO_IOU_THRESHOLD  = 0.5

DEPTH_MIN_M = 0.05
DEPTH_MAX_M = 15.0

CAM_H_MIN, CAM_H_MAX = 20, 160
CAM_V_MIN, CAM_V_MAX = 10, 80
MAX_SPRAY_ML          = 800

# Logit adjustments — Fungi over-predicts
CLASS_LOGIT_ADJUSTMENTS = {
    1: -1.5,
}

DEBUG = True


# ============================================================
# Hailo Inference — EXACT same wrapper as test_hailo_models_2.py
# (configure per-run, fresh output buffers per-run)
# ============================================================
class HailoInference:
    def __init__(self, hef_path, vdevice):
        self.hef = HEF(hef_path)
        self.infer_model = vdevice.create_infer_model(hef_path)
        self.infer_model.set_batch_size(1)

        for output in self.infer_model.outputs:
            output.set_format_type(FormatType.FLOAT32)

        input_info = self.hef.get_input_vstream_infos()[0]
        self.input_shape = input_info.shape
        self.input_name = input_info.name
        self.output_infos = self.hef.get_output_vstream_infos()

        print(f"  ✓ {os.path.basename(hef_path):35s} input={self.input_shape}")

    def run(self, input_data):
        if len(input_data.shape) == 3:
            input_data = np.expand_dims(input_data, axis=0)

        input_data = np.ascontiguousarray(input_data).astype(np.uint8)

        output_buffers = {
            out.name: np.empty(self.infer_model.output(out.name).shape, dtype=np.float32)
            for out in self.output_infos
        }

        with self.infer_model.configure() as configured_model:
            bindings = configured_model.create_bindings(output_buffers=output_buffers)
            bindings.input().set_buffer(input_data)
            configured_model.run([bindings], 10000)

        return output_buffers


# ============================================================
# Pre-processing (same as v2)
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
    return cv2.resize(image, target_size)

def preprocess_lesion(image, target_size=(256, 256)):
    return cv2.resize(image, (target_size[1], target_size[0]))


# ============================================================
# Post-processing
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

def postprocess_yolo(outputs, meta, conf_threshold=YOLO_CONF_THRESHOLD,
                     iou_threshold=YOLO_IOU_THRESHOLD):
    box_t, score_t, mask_t, proto_t = [], [], [], []
    for name, tensor in outputs.items():
        shape = tensor.shape
        ch = shape[-1]
        if ch == 64: box_t.append(tensor)
        elif ch == 1: score_t.append(tensor)
        elif ch == 32:
            if 2100 in shape:
                mask_t.append(tensor)
            else:
                proto_t.append(tensor)

    if len(box_t) == 0 or len(score_t) == 0 or len(proto_t) == 0:
        if DEBUG: print(f"  [DEBUG] Missing tensors: Box={len(box_t)} Score={len(score_t)} Protos={len(proto_t)}")
        return []

    if len(box_t) == 1 and box_t[0].shape[1] == 2100:
        all_boxes = box_t[0].reshape(-1, 64)
        all_scores = score_t[0].reshape(-1)
        all_coeffs = mask_t[0].reshape(-1, 32)
        strides = [8, 16, 32]
        feat_sizes = [(40, 40), (20, 20), (10, 10)]
    else:
        return []

    if all_scores.max() > 1.0 or all_scores.min() < 0.0:
        all_scores = 1 / (1 + np.exp(-all_scores))

    conf_mask = all_scores > conf_threshold
    if not np.any(conf_mask): return []

    fb, fs, fc = all_boxes[conf_mask], all_scores[conf_mask], all_coeffs[conf_mask]
    anchors, st = make_anchors(feat_sizes, strides)
    fa, fst = anchors[conf_mask], st[conf_mask]

    ltrb = dfl_decode(fb)
    x1 = np.clip((fa[:, 0:1] - ltrb[:, 0:1]) * fst, 0, 320)
    y1 = np.clip((fa[:, 1:2] - ltrb[:, 1:2]) * fst, 0, 320)
    x2 = np.clip((fa[:, 0:1] + ltrb[:, 2:3]) * fst, 0, 320)
    y2 = np.clip((fa[:, 1:2] + ltrb[:, 3:4]) * fst, 0, 320)
    bw, bh = x2 - x1, y2 - y1

    boxes_for_nms = []
    for i in range(len(x1)):
        boxes_for_nms.append([float(x1[i]), float(y1[i]), float(bw[i]), float(bh[i])])

    indices = cv2.dnn.NMSBoxes(boxes_for_nms, fs.tolist(), conf_threshold, iou_threshold)

    results = []
    if len(indices) == 0: return []

    protos = proto_t[0].squeeze()
    scale, top, left = meta

    for i in indices:
        if isinstance(i, (list, np.ndarray, tuple)): i = i[0]
        bx, by, bw_i, bh_i = boxes_for_nms[i]

        mask = (protos @ fc[i]).reshape(80, 80)
        mask = 1 / (1 + np.exp(-mask))
        mask = cv2.resize(mask, (320, 320))
        mask_crop = mask[int(by):int(by+bh_i), int(bx):int(bx+bw_i)]
        mask_binary = (mask_crop > 0.5).astype(np.uint8)

        results.append({
            'box': [int((bx-left)/scale), int((by-top)/scale),
                    int(bw_i/scale), int(bh_i/scale)],
            'confidence': float(fs[i]),
            'mask': mask_binary
        })
    return results

def postprocess_unet(outputs, threshold=0.5):
    mask = list(outputs.values())[0].squeeze()
    mask_sigmoid = 1.0 / (1.0 + np.exp(-mask))
    return (mask_sigmoid > threshold).astype(np.uint8) * 255


# ============================================================
# Depth helpers
# ============================================================
def sample_depth_patch(depth_map, dx, dy, depth_h, depth_w, patch=7):
    """7×7 trimmed mean."""
    half = patch // 2
    y0 = max(0, dy - half);  y1 = min(depth_h, dy + half + 1)
    x0 = max(0, dx - half);  x1 = min(depth_w, dx + half + 1)
    region = depth_map[y0:y1, x0:x1].ravel()
    if region.size == 0:
        return None
    sorted_vals = np.sort(region)
    trim = max(1, len(sorted_vals) // 5)
    trimmed = sorted_vals[trim:-trim] if len(sorted_vals) > 2 * trim else sorted_vals
    return float(np.mean(trimmed))

def raw_to_metric(raw_val, scale, bias):
    if raw_val is None:
        return None
    d = scale * raw_val + bias
    return float(max(DEPTH_MIN_M, min(DEPTH_MAX_M, d)))


# ============================================================
# Coordinate / spray helpers
# ============================================================
def centroid_to_servo_angles(cx, cy, fw, fh):
    h = int(CAM_H_MIN + (cx / fw) * (CAM_H_MAX - CAM_H_MIN))
    v = int(CAM_V_MIN + (cy / fh) * (CAM_V_MAX - CAM_V_MIN))
    return max(CAM_H_MIN, min(CAM_H_MAX, h)), max(CAM_V_MIN, min(CAM_V_MAX, v))

def infection_to_spray_ml(pct):
    ml = int((pct / 100.0) * MAX_SPRAY_ML)
    return max(50, min(MAX_SPRAY_ML, ml))


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="KRISHI-EYE v3: Static Image Test + Depth + UART")
    parser.add_argument("--yolo_hef",  default="yolov8n_seg.hef")
    parser.add_argument("--cls_hef",   default="potato_classifier.hef")
    parser.add_argument("--unet_hef",  default="lesion_segmentation.hef")
    parser.add_argument("--depth_hef", default="scdepthv3.hef")
    parser.add_argument("--input",     required=True, help="Image file or directory")
    parser.add_argument("--output_dir", default="output_v3", help="Save results here")
    parser.add_argument("--bias",  type=float, default=-7.840)
    parser.add_argument("--scale", type=float, default=-1.45)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect images
    if os.path.isdir(args.input):
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        image_paths = sorted([os.path.join(args.input, f) for f in os.listdir(args.input)
                              if f.lower().endswith(valid_exts)])
    else:
        image_paths = [args.input]

    if not image_paths:
        print("No images found.")
        return

    vparams = VDevice.create_params()
    vparams.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

    with VDevice(vparams) as vdevice:
        print(f"\n🚀 Loading 4-Model Pipeline | {len(image_paths)} image(s)")
        yolo_inf  = HailoInference(args.yolo_hef, vdevice)
        cls_inf   = HailoInference(args.cls_hef, vdevice)
        unet_inf  = HailoInference(args.unet_hef, vdevice)
        depth_inf = HailoInference(args.depth_hef, vdevice)

        depth_h = depth_inf.input_shape[0]
        depth_w = depth_inf.input_shape[1]

        total_diseased = 0
        total_leaves   = 0

        for img_path in image_paths:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            print(f"\n{'─'*60}")
            print(f"📷 Processing: {img_path}")

            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"  ⚠️  Could not read: {img_path}")
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            frame_h_px, frame_w_px = img_bgr.shape[:2]
            t0 = time.time()

            # ── YOLO detection (on RGB, same as v2) ──────────
            input_y, meta_y = preprocess_yolo(img_rgb)
            yolo_out = yolo_inf.run(input_y)
            detections = postprocess_yolo(yolo_out, meta_y)

            # ── Depth (full image, once) ─────────────────────
            depth_input = cv2.resize(img_bgr, (depth_w, depth_h))
            depth_out   = depth_inf.run(depth_input)
            depth_map   = list(depth_out.values())[0].squeeze().astype(np.float32)

            print(f"  Leaves detected: {len(detections)}")
            print(f"  Depth map: shape={depth_map.shape}, range=[{depth_map.min():.3f}, {depth_map.max():.3f}]")

            for i, det in enumerate(detections):
                x, y, w, h = det['box']
                x1, y1, x2, y2 = x, y, x+w, y+h

                bb_crop = img_rgb[max(0,y1):y2, max(0,x1):x2]
                if bb_crop.size == 0: continue

                # ── Segmented crop (same as v2) ──────────────
                mask_resized = cv2.resize(det['mask'], (bb_crop.shape[1], bb_crop.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)
                segmented_crop = cv2.bitwise_and(bb_crop, bb_crop, mask=mask_resized)

                # Save segmented leaf
                leaf_path = os.path.join(args.output_dir, f"{base_name}_leaf_{i}.png")
                cv2.imwrite(leaf_path, cv2.cvtColor(segmented_crop, cv2.COLOR_RGB2BGR))

                # ── Classifier (same as v2 + logit adjustment) ─
                cls_out = cls_inf.run(preprocess_classifier(segmented_crop))
                logits = list(cls_out.values())[0].ravel().copy()     # .copy() for adjustment
                for cid_adj, adj in CLASS_LOGIT_ADJUSTMENTS.items():
                    if cid_adj < len(logits):
                        logits[cid_adj] += adj
                probs = np.exp(logits - np.max(logits))
                probs = probs / np.sum(probs)
                mid = int(np.argmax(probs))
                label = CLASSIFIER_CLASSES[mid]
                conf = float(probs[mid])

                total_leaves += 1

                # ── Centroid ─────────────────────────────────
                M = cv2.moments(mask_resized)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = (x2-x1)//2, (y2-y1)//2
                bx_c = max(0, x1) + cx
                by_c = max(0, y1) + cy

                # ── Depth at centroid ────────────────────────
                dx = int(np.clip(bx_c * depth_w / frame_w_px, 0, depth_w - 1))
                dy = int(np.clip(by_c * depth_h / frame_h_px, 0, depth_h - 1))
                raw_d = sample_depth_patch(depth_map, dx, dy, depth_h, depth_w)
                metric_d = raw_to_metric(raw_d, args.scale, args.bias)
                d_str = f"{metric_d:.3f}m" if metric_d else "N/A"

                # ── Servo / nozzle angles ────────────────────
                h_cam, v_cam = centroid_to_servo_angles(bx_c, by_c, frame_w_px, frame_h_px)
                h_nozzle = h_cam - 120
                v_nozzle = v_cam + 130
                if h_nozzle < 0:   h_nozzle += 180
                if h_nozzle > 180: h_nozzle -= 180
                if v_nozzle < 0:   v_nozzle += 180
                if v_nozzle > 180: v_nozzle -= 180

                if label != "Healthy":
                    total_diseased += 1

                    # ── UNet lesion ───────────────────────────
                    unet_input = preprocess_lesion(segmented_crop)
                    unet_out = unet_inf.run(unet_input)
                    lesion_mask = postprocess_unet(unet_out)

                    # Infection %
                    lesion_rsz = cv2.resize(lesion_mask, (bb_crop.shape[1], bb_crop.shape[0]),
                                            interpolation=cv2.INTER_NEAREST)
                    leaf_px = np.count_nonzero(mask_resized)
                    inf_px  = np.count_nonzero(lesion_rsz > 127)
                    inf_pct = (inf_px / leaf_px * 100.0) if leaf_px > 0 else 0.0
                    liquid_ml = infection_to_spray_ml(inf_pct)

                    # Save lesion mask
                    mask_path = os.path.join(args.output_dir, f"{base_name}_lesion_{i}.png")
                    cv2.imwrite(mask_path, lesion_mask)

                    # ── UART packet (simulated) ──────────────
                    uart_pkt = f"{h_cam} {v_cam} {h_nozzle} {v_nozzle} {liquid_ml} {mid}"

                    # ── Draw on annotated image ──────────────
                    roi = img_bgr[max(0,y1):y2, max(0,x1):x2]
                    overlay = np.zeros_like(roi)
                    overlay[lesion_rsz > 127] = (0, 0, 255)
                    cv2.addWeighted(overlay, 0.4, roi, 1.0, 0, roi)

                    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.circle(img_bgr, (bx_c, by_c), 6, (0, 255, 255), -1)
                    cv2.putText(img_bgr, f"{label} {conf:.0%} | {d_str} | {inf_pct:.0f}%",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    # ── Terminal output ───────────────────────
                    print(f"\n  🎯 Leaf {i}: {label} ({conf:.1%})")
                    print(f"    ├─ BBox       : ({x1},{y1})→({x2},{y2})")
                    print(f"    ├─ Centroid   : ({bx_c}, {by_c})")
                    print(f"    ├─ Depth      : {d_str}")
                    print(f"    ├─ Servos     : H_cam={h_cam}° V_cam={v_cam}°")
                    print(f"    ├─ Nozzle     : H={h_nozzle}° V={v_nozzle}°")
                    print(f"    ├─ Infection  : {inf_pct:.1f}% → {liquid_ml} mL")
                    print(f"    └─ 📡 UART    : {uart_pkt}")

                else:
                    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_bgr, f"{label} {conf:.0%} | {d_str}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    print(f"  ✅ Leaf {i}: {label} ({conf:.1%}) | Depth: {d_str}")

            elapsed = (time.time() - t0) * 1000
            output_img = os.path.join(args.output_dir, f"{base_name}_result.png")
            cv2.imwrite(output_img, img_bgr)
            print(f"\n  ⏱  {elapsed:.0f}ms | Saved → {output_img}")

        # ── Summary ──────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"📊 SUMMARY")
        print(f"  Images    : {len(image_paths)}")
        print(f"  Leaves    : {total_leaves}")
        print(f"  Diseased  : {total_diseased}")
        print(f"  Output    : {os.path.abspath(args.output_dir)}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
