# Codebase Concerns

**Analysis Date:** 2026-02-23

## Tech Debt

**Orchestration Mocking:**
- Issue: The core pipeline orchestrator relies almost entirely on mock logic and sleeps rather than actual model inference.
- Files: `orchestrator.py`
- Impact: The system cannot be deployed or tested for real-world accuracy without a complete rewrite of the inference logic.
- Fix approach: Implement `HailoRT` integration and replace mock ROIs with actual model outputs from `yolov8n_seg`, `mobilenetv2`, and `unet`.

**Hardcoded Kaggle Paths:**
- Issue: Training scripts contain hardcoded paths pointing to Kaggle input directories.
- Files: `Infection Classification/train.py`, `Leaf Segmentation/train.py`, `Lesion Segmentation/train.py`
- Impact: Scripts fail when run outside the Kaggle environment (e.g., local machine or dedicated CI).
- Fix approach: Use environment variables or a configuration file to define dataset paths.

**Missing QAT (Quantization Aware Training):**
- Issue: Comments mention full precision is critical for QAT, but QAT is not actually implemented.
- Files: `Infection Classification/train.py`, `Lesion Segmentation/train.py`
- Impact: Reduced accuracy when quantizing models for the Hailo-8L (INT8).
- Fix approach: Implement `hailo_model_optimization` or PyTorch QAT before exporting to ONNX.

## Known Bugs

**ONNX Export Compatibility:**
- Issue: Comments indicate issues in support for model conversion to HEF format.
- Files: `Infection Classification/train.py`
- Symptoms: Potential failure during the Hailo compiler phase.
- Fix approach: Use `hailo_model_zoo` compatible architectures or verify opset compatibility (currently using Opset 11).

## Security Considerations

**Unprotected Camera Access:**
- Risk: `cv2.VideoCapture(0)` is used without validation or selection logic.
- Files: `orchestrator.py`
- Current mitigation: None.
- Recommendations: Add configuration for camera index/source and implement error handling if the source is unavailable.

## Performance Bottlenecks

**Sequential ROI Processing:**
- Problem: `stage_2_analysis` processes leaf ROIs sequentially in a loop.
- Files: `orchestrator.py`
- Cause: If multiple leaves are detected, processing time scales linearly, potentially dropping FPS below the target (15-20 FPS).
- Improvement path: Implement batch inference for Stage 2 models or use Hailo's multi-context capabilities to parallelize classification and segmentation.

## Fragile Areas

**Dataset Path Parsing:**
- Files: `Lesion Segmentation/train.py`
- Why fragile: Uses `.replace("images", "masks")` which can break if the path contains "images" in parent directories.
- Safe modification: Use `pathlib` or more specific path manipulation logic.

## Scaling Limits

**Repository Size:**
- Current capacity: Large due to image datasets committed directly to git.
- Limit: Will become unmanageable as the dataset grows (currently thousands of images).
- Scaling path: Move datasets to external storage (S3, DVC, or Kaggle directly) and use `.gitignore`.

## Test Coverage Gaps

**Inference Logic:**
- What's not tested: Real NPU inference and stereo depth logic.
- Files: `orchestrator.py`
- Risk: Hardware-specific bugs won't be caught until deployment on RPi5.
- Priority: High

---

*Concerns audit: 2026-02-23*
