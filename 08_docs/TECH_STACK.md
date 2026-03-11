# Technology Stack

## Hardware Ecosystem
- **Compute Layer**: Raspberry Pi 5 (Active Cooled enclosure).
- **Inference Acceleration**: Hailo-8L NPU M.2 HAT (up to 13 TOPS).
- **Vision Sensors**: IMX219-83 Stereo Camera arrays.
- **Physical Actuation**: Teensy 4.1 Development Board (handles the resulting UART packet strings translated into precision actuator steps).

## Target Runtimes & Frameworks Configurations
| Tooling | Usage & Scope |
|-----|--------|
| **Python 3.10+** | Overall architectural pipeline code (Orchestrator logic and calibration scripts). |
| **OpenCV (`cv2`)** | Rapid matrix transformations, CLAHE balancing, drawing bounding boxes, and stream ingestion tasks. |
| **HailoRT** | Low-level C++/Python runtime pushing frames to NPU silicon and recovering native INT8 streams. |
| **Hailo Dataflow Compiler (DFC)** | Command Line Interface stack scaling ONNX graphs systematically into HEF compiled binaries taking `alls` config tuning routines. |

## Model Export Formats
| Format | State | Execution Plane |
|---|---|---|
| **FP32 ONNX (v11 Opset)** | Interim Graph | Initial baseline representations verified from Kaggle/PyTorch/Darknet. |
| **Hailo HEF** | Edge Binary | Evaluation-ready native NPU structures quantized seamlessly against calibration sets. |
| **HAR** | Evaluation Sandbox | Pre-compiled model archive files tracking network metrics internally during the tuning compilation profile. |

## Communication Protocols
- **Serial (UART)**: Real-time telemetry connection executing physical spray commands spanning across standard GPIO structures to external MCU handlers directly off the Pi.
