# Model Overview

The KRISHi-EYE crop inspection ecosystem handles inference logic structurally disjointed across separate specialized models running continuously to map distinct visual features inside complex real-world conditions.

The final evaluation architectures were specifically compressed and structured with hardcoded dimensions optimized directly for compilation into `Hailo Executable Formats` via the Data Flow Compiler.

## Supported Model Implementations

### YOLOv8n-Seg (Leaf Segmentation)
**Objective**: Segment entire specific leaves distinguishing target flora away from overlapping background vegetation.
- **Type**: Instance Segmentation
- **Architecture**: FP32 ONNX -> INT8 quantized HEF.
- **Input Dimensions**: 640x640x3
- **Output Classes**: `Leaf` vs `Background`
- **Dependency Pipeline**: Bounding box outputs define dynamic ROI crops sent sequentially down to the classifier and UNet segmenter architectures minimizing full-field computational loads.

### Potato Disease Classifier
**Objective**: Resolve explicit classification attributes applied exclusively towards cropped leaf abstractions.
- **Type**: Multiclass Image Classifier
- **Architecture**: MobileNet/ResNet subset FP32 ONNX -> INT8 quantized HEF.
- **Input Dimensions**: 224x224x3
- **Classification Output space**: `Bacteria`, `Fungi`, `Healthy`, `Nematode`, `Pest`, `Phytophthora`, `Virus`
- **Application Logic**: Evaluated labels determine exactly which specialized liquid fungicide/pesticide delivery valve sequence is engaged inside target UART packets. Softmax bias thresholds (`CLASS_LOGIT_ADJUSTMENTS`) actively suppress overly sensitive classes suppressing localized over-fitting triggers.

### Lesion Segmentation (UNet)
**Objective**: Dynamically trace visual damage boundaries marking total lesion spread within the localized target leaf framework. 
- **Type**: Semantic Base Segmentation
- **Architecture**: UNet FP32 ONNX -> INT8 quantized HEF.
- **Input Dimensions**: 256x256x3
- **Application Logic**: Outputs a structural mask comparing diseased pixel counts versus total leaf pixel quantities generated via YOLO bounding box volumes. Resulting percentage determines exact mL fluid actuation requirements sent back to the Teensy spray logic. 

### SCDepthV3
**Objective**: Map metric Z-Distance properties translating fixed pixel coordinates mapped by UNet centroids.
- **Type**: Monocular Depth Estimation
- **Application Logic**: Ensures subjects exist directly within actionable physical boundaries corresponding to actual boom sprayer operational ranges (Min: `0.05m`, Max: `15.0m`).

## Storage Representation

Evaluators can find the above deployments mapped across two directories maintaining distinct implementation variants: 
- `04_models/onnx` - Useful entirely for abstract structural inspection and reference mapping directly against standard execution layers (e.g. visualizing architecture through Netron).
- `04_models/hef` - The primary working binaries optimized, sliced, and fully parameterized targeting physical execution exclusively spanning across Hailo environments.
