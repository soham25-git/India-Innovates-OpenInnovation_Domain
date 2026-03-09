# Technology Stack

> Krishi-Eye Smart Sprayer Stack

## Hardware Components
| Component | Purpose |
|------------|---------|
| **Raspberry Pi 5 (8GB)** | Primary compute node for managing the camera, orchestrating the pipeline, and communicating with peripherals. |
| **Hailo-8L M.2 NPU** | Dedicated hardware neural processing unit capable of up to 13 TOPS for executing the deep learning networks. |
| **Teensy 4.1** | Real-time microcontroller responsible for interpreting UART packets and smoothly driving servos and valves. |
| **RPi Camera Module 3** | High-definition sensor for optical ingestion. |

## Neural Models (Hailo `.hef` format)
| Model | Role | Output |
|---------|---------|---------|
| **YOLOv8n-seg** | Primary detector | Leaf bounding boxes and instance segmentation masks. |
| **MobileNetV2** | Disease classifier | 7-class disease categorization (Fungi, Bacteria, Healthy, Nematode, Pest, Phytophthora, Virus). |
| **UNet** | Lesion segmenter | Pixel-wise binary mask of necrotic/infected tissue. |
| **SCDepthV3** | Monocular Depth | Estimated distance metric for precise chemical volume targeting. |

## Core Software
| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.11+ | The orchestrating programming language. |
| **OpenCV (`cv2`)** | 4.x | Image manipulation, CLAHE enhancement, and UI overlay rendering. |
| **HailoRT** | 4.17+ | The Hailo Runtime to interface with the Hailo-8L NPU chips. |
| **PySerial** | 3.5+ | Handling UART communication with the Teensy. |
| **NumPy** | 1.25+ | High-performance matrix array formatting for the NPU buffers. |
