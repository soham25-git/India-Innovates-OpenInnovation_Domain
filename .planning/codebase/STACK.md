# Technology Stack

**Analysis Date:** 2026-02-23

## Languages

**Primary:**
- Python 3.x - Core pipeline orchestration, model training, and validation scripts.

**Secondary:**
- Shell/PowerShell - Utility scripts for repository search and validation (`scripts/`).

## Runtime

**Environment:**
- Raspberry Pi 5 (RPi5) with Hailo-8L AI Accelerator.
- Linux (likely Debian/Raspbian based on RPi5 target).

**Package Manager:**
- `pip` - Used for Python dependency management.
- Lockfile: missing (no `requirements.txt` or `pyproject.toml` found in root, but `venv/` exists).

## Frameworks

**Core:**
- PyTorch (`torch`) - Primary deep learning framework for training and model definition.
- Ultralytics YOLOv8 (`ultralytics`) - Used for leaf segmentation tasks.
- HailoRT (`hailort`) - Native SDK for inference on Hailo AI accelerators.

**Testing:**
- Custom validation scripts - Located in `scripts/` (e.g., `validate-all.sh`, `verify_orchestration.py`).

**Build/Dev:**
- ONNX - Used for model export and intermediate representation.
- Hailo Dataflow Compiler (implied) - Used for converting ONNX to HEF (Hailo Executable Format).

## Key Dependencies

**Critical:**
- `opencv-python` (`cv2`) - Image processing and camera interface.
- `numpy` - Numerical operations and array handling.
- `timm` - PyTorch Image Models for classification backbones.
- `hailort` - Required for NPU-accelerated inference.

**Infrastructure:**
- `PyYAML` (`yaml`) - Configuration parsing.
- `albumentations` - Data augmentation for training.
- `scikit-learn` (`sklearn`) - Metrics and evaluation utilities.

## Configuration

**Environment:**
- YAML-based configuration - `orchestrator.py` expects a config file.
- `.env` files (not detected in root, but supported by standard patterns).

**Build:**
- `model_capabilities.yaml` - Registry for model selection guidance.

## Platform Requirements

**Development:**
- Python 3.10+
- CUDA (for training on Kaggle/local GPU)

**Production:**
- Raspberry Pi 5
- Hailo-8L NPU
- `hailort` driver and library installed

---

*Stack analysis: 2026-02-23*
