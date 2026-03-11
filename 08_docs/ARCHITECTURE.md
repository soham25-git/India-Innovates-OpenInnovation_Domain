# System Architecture

## Core Data Flow

At a high level, the system captures RGB frames and executes a cascaded, logic-gated computer vision pipeline. The architecture is primarily divided into edge processing on a Pi 5 mapped over a Hailo-8L core, followed by UART signal propagation to actuation circuitry (Teensy 4.1).

### The Pipeline Execution
1. **Source Acquisition**: Streams captured from the `imx219-83` (stereo camera module) using `rpicam-vid` or `v4l2src` backends.
2. **Pre-Processing Filtering**:
   - Frames mapped through Contrast Limited Adaptive Histogram Equalization (CLAHE) for illumination balancing, resolving shadow and glare failures.
3. **Stage 1 (Centroid / Subject Detection)**:
   - YOLOv8n-seg scans the full field of view to locate distinct crop leaves.
   - Identified regions of interest (ROI) are tracked by centroid. A cooldown threshold is evaluated to prevent redundant processing of the exact same leaf coordinate over consecutive frames (stabilizing logic over shaking rigs).
4. **Stage 2 (Parallel Feature Extraction)**:
   - ROI bounding boxes are scaled and pushed to **UNet Lesion Segmentation**, marking affected pixels.
   - ROI segmentations are normalized and passed into the **Disease Classifier** to output labels and confidence scores.
   - The same centroid fetches depth readings synchronously via **SCDepthV3** or stereo processing logic, mapping out physical Z-space (filtered by minimum limits `0.05m` to `15m`).
5. **Actuation Command Synthesis**:
   - Resulting coordinates are geometrically mapped into stepper servo angles (`v_cam`, `h_cam`) and corresponding nozzle offsets (`v_nozzle`, `h_nozzle`).
   - Liquid delivery volume is directly proportional to the relative percentage of lesion pixels identified on the leaf surface by UNet.
6. **UART Propagation**:
   - Disease classification defines which discrete fluid valve is triggered.
   - Outputs constructed into a unified `<pwm1> <pwm2> <sc1>...` Teensy packet mapping, passing the action onto physical motors or solenoids.

### Training & Export Lifecycle
During initial development, the system utilized full-precision model files standard to TensorFlow and PyTorch toolchains.
1. All sub-models (UNet, Classifier, YOLO) were explicitly exported strictly to **FP32 ONNX**. 
2. PyTorch QAT (Quantization Aware Training) is actively stripped, enabling pure execution control inside the deployment layer.
3. Calibration routines (`03_training/prepare_calib_*.py`) slice evaluation training images to raw unsigned `uint8` sets for DFC (Dataflow Compilation).
4. These calibrations normalize weight gradients uniformly into `hailoc` conversion graphs, finalizing natively executable binary HEF outputs (Hailo Executable Format) running specifically inside `hailort` boundaries, taking advantage of INT8 opsets across the NPU’s dedicated hardware logic structure. 
