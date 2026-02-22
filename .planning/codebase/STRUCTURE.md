# Codebase Structure

**Analysis Date:** 2026-02-23

## Directory Layout

```
[project-root]/
├── HAILO/                      # NPU optimization and optimized models
│   ├── calibration/            # Calibration datasets (.npy)
│   ├── models/                 # Target ONNX models for HEF conversion
│   └── scripts/                # Calibration data preparation scripts
├── Infection Classification/   # Disease type classification task
├── Leaf Segmentation/          # YOLOv8-seg leaf detection task
├── Lesion Segmentation/        # Lesion quantification task
├── adapters/                   # AI agent instructions (Claude, Gemini, GPT)
├── docs/                       # Project documentation
├── scripts/                    # Shared utility scripts
├── orchestrator.py             # Main pipeline entry point
├── SUMMARY.md                  # Project overview and constraints
└── model_capabilities.yaml     # AI model selection guidance (internal)
```

## Directory Purposes

**HAILO/:**
- Purpose: Contains assets and scripts for Hailo-8L NPU deployment.
- Contains: Model calibration data, optimization scripts, and deployment-ready ONNX models.
- Key files: `HAILO/scripts/prepare_calib_yolo.py`, `HAILO/models/yolov8n_seg.onnx`.

**Leaf Segmentation/:**
- Purpose: Focuses on identifying potato leaves from raw images.
- Contains: YOLOv8 training scripts, dataset structure, and task goals.
- Key files: `Leaf Segmentation/train.py`, `Leaf Segmentation/GOAL.md`.

**Infection Classification/:**
- Purpose: Classifies detected leaves into disease categories (e.g., Fungi, Bacteria).
- Contains: Classification training code and image datasets.
- Key files: `Infection Classification/train.py`.

**Lesion Segmentation/:**
- Purpose: Segments individual lesions on infected leaves.
- Contains: UNet or similar segmentation model code and datasets.
- Key files: `Lesion Segmentation/train.py`.

## Key File Locations

**Entry Points:**
- `orchestrator.py`: Main execution script for the live pipeline.

**Configuration:**
- `.gsd/hailo_pipeline_config.yaml`: Pipeline parameters (FPS, model paths).

**Core Logic:**
- `orchestrator.py`: `PotatoPipeline` class implementation.

**Testing:**
- (Mock mode in `orchestrator.py` serves as a functional test harness).

## Naming Conventions

**Files:**
- Snake Case for Python scripts: `prepare_calib_yolo.py`
- Upper Case for documentation: `SUMMARY.md`, `GOAL.md`

**Directories:**
- Pascal Case or Title Case for task directories: `Leaf Segmentation`
- Upper Case for hardware/system directories: `HAILO`

## Where to Add New Code

**New Feature (e.g., New Analysis Task):**
- Create a new task directory (e.g., `Stem Analysis/`).
- Add a stage method to `PotatoPipeline` in `orchestrator.py`.
- Update configuration in `.gsd/hailo_pipeline_config.yaml`.

**New Component/Module:**
- Shared utilities should go into `scripts/`.
- Hardware-specific adapters should go into `adapters/` (if implementing code-based adapters).

**Utilities:**
- `scripts/`: General helpers.
- `HAILO/scripts/`: Calibration helpers.

## Special Directories

**venv/:**
- Purpose: Python virtual environment.
- Generated: Yes
- Committed: No (typically ignored, but present in current workspace).

**.planning/:**
- Purpose: GSD system documentation and plans.
- Generated: Yes
- Committed: Yes

---

*Structure analysis: 2026-02-23*
