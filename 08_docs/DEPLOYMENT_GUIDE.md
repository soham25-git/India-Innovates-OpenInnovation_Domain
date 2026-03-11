# Evaluator Deployment Guide

## Context

Running the deployment logic is tied fundamentally to specific hardware ecosystems. Testing the pipeline requires execution layers found strictly on the Raspberry Pi 5 interfaced gracefully with the Hailo-8L accelerator (via the M.2 HAT interface) and functional UART/Camera hardware. 

If this hardware is unavailable, the pipeline can be statically analyzed through the `02_deployment` code traces and `04_models/onnx` schemas.

## Step 1: Physical Actuation Preparation

1. Flash `07_hardware_calibration/krishi_eye_teensy.ino` onto your target Teensy 4.1 controller.
2. Wire logic outputs referencing `CLASS_TO_VALVE` (e.g. Pin 26 corresponds physically to the Bacterial agent delivery solenoid).
3. Secure the serial port connection to the Pi 5 interface (e.g. `/dev/ttyACM0`).

## Step 2: Edge NPU Setup (Software Layer)

1. Provision the Pi 5 operating system with a standard 64-bit kernel environment (e.g. Debian Bookworm).
2. Install the `hailo-tappas-core` ecosystem managing the PCIe interface channels down to the NPU logic board.
3. Source a standard Python environment and apply constraints:

```bash
python3 -m venv krishi_env
source krishi_env/bin/activate
pip install -r 05_configs/edge_requirements.txt
```

Verify `hailortcli scan` confirms the successful mounting of the hardware accelerator.

## Step 3: Deployment Testing
Ensure the `04_models/hef/` files correctly occupy the working directory executing the logic pipeline.

```bash
cd 02_deployment/
python main_edge_inference_pipeline.py \
    --uart /dev/ttyACM0 \
    --yolo ../04_models/hef/leaf_segmentation_yolov8n.hef \
    --cls ../04_models/hef/disease_classifier_mobilenet.hef \
    --unet ../04_models/hef/lesion_segmentation_unet.hef
```

## Understanding Execution Behavior
Once deployed, the `main_edge_inference_pipeline.py` orchestrates execution asynchronously:
1. **Camera Validation**: Initially sequences via `find_working_camera()` binding to GStreamer endpoints tracking down functional hardware components (`rpicam-vid`, `v4l2src`).
2. **Buffering**: Pre-allocates NPU memory tensors exactly once per compiled graph (YOLO, Classifier, UNet). The environment cycles across frames bypassing blocking operations.
3. **Filtering**: Pre-processes shadow anomalies using CLAHE to normalize target environments directly impacting SNR outcomes.
4. **Metric Resolution**: Maps `y`, `x` pixel offsets referencing lesions over depth outputs interpolating Z-Distance metric offsets in meters.
5. **Actuation Command**: Translating infection percentile coverage dynamically mapping to liquid pump mL output, communicating across the UART chain.

## Evaluator Notes
If testing entirely simulated environments, simply supply `--no-clahe` or disable UART configurations locally to analyze output confidence and tracking cooldown parameters visually via the OpenCV bounding boxes outputting directly to virtual framebuffers.
