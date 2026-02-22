# Architecture

**Analysis Date:** 2026-02-23

## Pattern Overview

**Overall:** Multi-stage edge AI pipeline optimized for NPU acceleration.

**Key Characteristics:**
- **Pipeline-based:** Decoupled stages for leaf detection, classification, and quantification.
- **Hardware-Targeted:** Specifically designed for Raspberry Pi 5 + Hailo-8L NPU.
- **Mock-Compatible:** Integrated simulation mode for development without hardware.

## Layers

**Orchestration Layer:**
- Purpose: Manages the end-to-end inference pipeline and data handoff between models.
- Location: `orchestrator.py`
- Contains: `PotatoPipeline` class, pipeline control logic, visualization.
- Depends on: `cv2`, `numpy`, `hailort` (optional).
- Used by: Entry point for live inference.

**AI Model Layer:**
- Purpose: Task-specific inference models.
- Location: `Leaf Segmentation/`, `Infection Classification/`, `Lesion Segmentation/`
- Contains: ONNX model files, training scripts, datasets.
- Depends on: `ultralytics` (for YOLO), `onnxruntime` or `hailort`.
- Used by: Orchestration layer for stage-specific analysis.

**Optimization Layer:**
- Purpose: Model calibration and conversion for Hailo NPU.
- Location: `HAILO/`
- Contains: Calibration scripts, NPU-optimized models (HEF target).
- Depends on: `hailo_sdk_client` (Dataflow Compiler).
- Used by: Production deployment on RPi5.

## Data Flow

**Inference Pipeline:**

1. **Acquisition:** Frame captured from camera via `cv2.VideoCapture` in `orchestrator.py`.
2. **Stage 1 (Leaf Detection):** Frame passed to YOLOv8n-seg model to identify leaf instances and extract ROIs.
3. **Stage 2 (Disease Analysis):** ROIs are cropped and passed in parallel/sequence to Classification and Lesion Segmentation models.
4. **Aggregation:** Results (class labels, lesion masks, depth estimates) are collected.
5. **Output:** `visualize()` overlays results on the original frame for display.

**State Management:**
- Stateless per-frame processing in the main pipeline.
- Configuration-driven via `.gsd/hailo_pipeline_config.yaml`.

## Key Abstractions

**PotatoPipeline:**
- Purpose: Encapsulates the multi-model logic and hardware abstraction.
- Examples: `orchestrator.py`
- Pattern: Strategy (Mock vs. Hardware inference).

## Entry Points

**Orchestrator CLI:**
- Location: `orchestrator.py`
- Triggers: Manual execution via python.
- Responsibilities: Initializes pipeline, starts capture loop, handles user input (e.g., 'q' to quit).

## Error Handling

**Strategy:** Graceful degradation and logging.

**Patterns:**
- **Hardware Fallback:** `ImportError` on `hailort` triggers automatic mock mode.
- **Validation:** Config file existence check during initialization.

## Cross-Cutting Concerns

**Logging:** Console print statements for initialization and warnings.
**Validation:** Image size and format validation in `HAILO/scripts`.
**Authentication:** Not applicable (Edge deployment).

---

*Architecture analysis: 2026-02-23*
