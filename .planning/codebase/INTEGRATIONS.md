# External Integrations

**Analysis Date:** 2026-02-23

## APIs & External Services

**AI Accelerator:**
- Hailo-8L - Hardware integration for high-performance edge inference.
  - SDK/Client: `hailort`
  - Auth: Local hardware access

**Model Repositories:**
- Ultralytics - YOLOv8 models and training integration.
- HuggingFace / Timm - Vision transformer and CNN backbones via `timm`.

**Data Sources:**
- Roboflow - Dataset source for leaf segmentation (referenced in `Leaf Segmentation/train.py`).
- Kaggle - Dataset source and training environment (referenced in `Infection Classification/train.py`).

## Data Storage

**Databases:**
- None - Project relies on flat files and local models.

**File Storage:**
- Local filesystem - Storage for `.onnx` and `.hef` models, calibration data, and dataset images.
- Structure:
  - `HAILO/models/`: Compiled and source models.
  - `*/YOLO Dataset/`: Image datasets for various modules.

**Caching:**
- None detected.

## Authentication & Identity

**Auth Provider:**
- Custom / Local - No external auth providers detected. Processing is designed for edge deployment on RPi5.

## Monitoring & Observability

**Error Tracking:**
- None - Relying on standard Python exceptions and logging.

**Logs:**
- Console logging - `orchestrator.py` and training scripts use `print` and `tqdm` for progress and status.

## CI/CD & Deployment

**Hosting:**
- Edge Deployment - Target is Raspberry Pi 5.

**CI Pipeline:**
- Custom scripts - `scripts/validate-all.sh` and related scripts for local validation before deployment.

## Environment Configuration

**Required env vars:**
- `CUDA_LAUNCH_BLOCKING` - Used in training scripts for debugging.
- `PYTHONHASHSEED` - Set for reproducibility.

**Secrets location:**
- Not applicable - No external API secrets identified in the core pipeline logic.

## Webhooks & Callbacks

**Incoming:**
- None detected.

**Outgoing:**
- None detected.

---

*Integration audit: 2026-02-23*
