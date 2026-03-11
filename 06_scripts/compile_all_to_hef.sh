#!/bin/bash
# =============================================================================
# IMPROVED Hailo ONNX → HEF Conversion Script
# Uses better calibration preprocessing (letterbox/center-crop)
# =============================================================================

set -e
export CUDA_VISIBLE_DEVICES=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"
CALIB_DIR="$PROJECT_DIR/calibration"
SCRIPTS_DIR="$SCRIPT_DIR"
OUTPUT_DIR="$PROJECT_DIR/output"

# Actual DATASET paths for calibration
YOLO_DATASET="/home/soham/Project/Leaf Segmentation/YOLO Dataset/train/images"
CLS_DATASET="/home/soham/Project/Infection Classification/Potato Leaf Disease Dataset in Uncontrolled Environment"
LESION_DATASET="/home/soham/Project/Lesion Segmentation/Lesion Segmentation Dataset 2/images/train"

mkdir -p "$OUTPUT_DIR"

# 1. Re-generate Calibration .npy files with improved scripts
echo "⚙️ Re-generating calibration data..."
python HAILO/improved_scripts/export_calibration_yolo.py --images-dir "$YOLO_DATASET" --output "$CALIB_DIR/yolo/calib_yolo_320.npy" --num-samples 1500 --size 320
python "$SCRIPTS_DIR/export_calibration_classifier.py" --images-dir "$CLS_DATASET" --output "$CALIB_DIR/classifier/calib_cls_224.npy" --num-samples 1500
python "$SCRIPTS_DIR/export_calibration_lesion.py" --images-dir "$LESION_DATASET" --output "$CALIB_DIR/lesion/calib_lesion_256.npy" --num-samples 1500

# 2. Conversion Pipeline
echo "🚀 Starting Conversion..."

# YOLOv8n-seg
echo "--- YOLOv8n-seg ---"
hailo parser onnx "$MODELS_DIR/leaf_segmentation_yolov8n.onnx" --hw-arch hailo8l --har-path yolov8n_seg.har --tensor-shapes "[1,3,320,320]" -y
hailo optimize yolov8n_seg.har --hw-arch hailo8l --calib-set-path "$CALIB_DIR/yolo/calib_yolo_320.npy" --model-script "$PROJECT_DIR/scripts/yolo_optimization.alls" --output-har-path "$OUTPUT_DIR/yolov8n_seg_quantized.har"
hailo compiler "$OUTPUT_DIR/yolov8n_seg_quantized.har" --hw-arch hailo8l --model-script "$PROJECT_DIR/scripts/yolo_compilation.alls" --output-dir "$OUTPUT_DIR"

# Potato Classifier
echo "--- Potato Classifier ---"
hailo parser onnx "$MODELS_DIR/disease_classifier_mobilenet.onnx" --hw-arch hailo8l --har-path potato_classifier.har -y
hailo optimize potato_classifier.har --hw-arch hailo8l --calib-set-path "$CALIB_DIR/classifier/calib_cls_224.npy" --model-script "$PROJECT_DIR/scripts/classifier_optimization.alls" --output-har-path "$OUTPUT_DIR/potato_classifier_quantized.har"
hailo compiler "$OUTPUT_DIR/potato_classifier_quantized.har" --hw-arch hailo8l --output-dir "$OUTPUT_DIR"

# Lesion Segmentation
echo "--- Lesion Segmentation ---"
hailo parser onnx "$MODELS_DIR/lesion_segmentation.onnx" --hw-arch hailo8l --har-path lesion_segmentation.har -y
hailo optimize lesion_segmentation.har --hw-arch hailo8l --calib-set-path "$CALIB_DIR/lesion/calib_lesion_256.npy" --model-script "$PROJECT_DIR/scripts/lesion_optimization.alls" --output-har-path "$OUTPUT_DIR/lesion_segmentation_quantized.har"
hailo compiler "$OUTPUT_DIR/lesion_segmentation_quantized.har" --hw-arch hailo8l --output-dir "$OUTPUT_DIR"

echo "✅ All models converted successfully."
ls -lh "$OUTPUT_DIR"/*.hef
