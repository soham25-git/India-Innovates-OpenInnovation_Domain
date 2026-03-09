# STATE.md

## Current Position
- **Phase**: 5 (ONNX to HEF Conversion)
- **Task**: Planning complete
- **Status**: Ready for execution

## Last Session Summary
Codebase mapping complete.
- 4 core components identified (Leaf Seg, Classification, Lesion Seg, Depth)
- 4 production dependencies analyzed (cv2, numpy, yaml, hailort)
- 3 technical debt items found (HailoRT placeholder, Denoising latency, Temporal smoothing)
- Phase 5 Research complete: ONNX to HEF conversion strategy defined.
All Phases (1-4) Complete:
- **Optimization:** Migrated Classification to MobileNet-V2, YOLOv8n-seg to 640x640, and UNet to 256x256.
- **Robustness:** Integrated Albumentations (Rain, Blur, Fog) and Hailo Model Zoo (Depth, Denoising).
- **Orchestration:** Built `orchestrator.py` with mock mode and ROI logic.
- **Deployment:** Created `DEPLOYMENT_GUIDE.md` for ONNX to HEF conversion.

## Next Steps (User)
1. **Download:** Pull ONNX files from Kaggle once training is finished.
2. **Convert:** Follow `DEPLOYMENT_GUIDE.md` to compile HEFs.
3. **Execute:** Run `python orchestrator.py` on the Pi 5.
