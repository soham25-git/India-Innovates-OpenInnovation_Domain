!pip install ultralytics -q

# dataset url : https://universe.roboflow.com/potato-detection-odp7l/potato-leaf-disease-nzxek-rszlg/dataset/1

import os
import yaml
from ultralytics import YOLO
import torch
import random
import numpy as np

# Set seeds for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seeds(42)

from ultralytics import YOLO

# Path to your prepared data.yaml describing train and val splits
data_path = '/kaggle/input/sigmapotato/data.yaml'

# Use YOLOv8 nano segmentation model - most lightweight for edge deployment
model_name = 'yolov8n-seg.pt'

model = YOLO(model_name)

# Optimized training parameters for better generalization and RPI deployment
training_params = {
    'data': data_path,
    'epochs': 100,                    # Increased from 80 for better convergence
    'imgsz': 640,                     # Increased to 640 for Hailo alignment and better precision
    'batch': 32,                      # Adjusted for 640 resolution on Kaggle T4
    'patience': 15,                   # Increased early stopping patience
    'workers': 4,
    'device': [0, 1],                 # Multi-GPU if available
    'project': 'potato_yolov8_seg_optimized',
    'name': 'experiment_optimized',

    # Optimizer settings
    'optimizer': 'AdamW',             # Better than Adam for generalization
    'lr0': 0.001,                     # Initial learning rate
    'lrf': 0.01,                      # Final learning rate (1% of initial)
    'momentum': 0.937,                # For SGD if switched
    'weight_decay': 0.0005,           # L2 regularization to prevent overfitting

    # Regularization and augmentation
    'dropout': 0.1,                   # Dropout for regularization
    'augment': True,                  # Enable data augmentation
    'degrees': 10.0,                  # Rotation augmentation
    'translate': 0.1,                 # Translation augmentation
    'scale': 0.5,                     # Scale augmentation
    'shear': 2.0,                     # Shear augmentation
    'perspective': 0.0001,            # Perspective augmentation
    'flipud': 0.5,                    # Vertical flip probability
    'fliplr': 0.5,                    # Horizontal flip probability
    'mosaic': 1.0,                    # Mosaic augmentation probability
    'mixup': 0.1,                     # Mixup augmentation probability
    'copy_paste': 0.1,                # Copy-paste augmentation

    # Model configuration
    'cls': 1.0,                       # Classification loss weight
    'box': 7.5,                       # Box regression loss weight
    'dfl': 1.5,                       # DFL loss weight
    'pose': 12.0,                     # Pose loss weight (if applicable)
    'kobj': 2.0,                      # Keypoint object loss weight
    'label_smoothing': 0.1,           # Label smoothing

    # Validation and saving
    'val': True,                      # Validate during training
    'save_period': 10,                # Save checkpoint every N epochs
    'verbose': True,                  # Print training logs
    'exist_ok': True,                 # Allow overwrite existing project
    'pretrained': True,               # Use pretrained weights
    'resume': False,                  # Resume from last checkpoint

    # Advanced settings for better performance
    'amp': True,                      # Automatic Mixed Precision for faster training
    'fraction': 1.0,                  # Use full dataset
    'profile': False,                 # Don't profile for speed
    'freeze': None,                   # Don't freeze any layers initially

    # Loss function tweaks
    'hsv_h': 0.015,                   # HSV-Hue augmentation
    'hsv_s': 0.7,                     # HSV-Saturation augmentation
    'hsv_v': 0.4,                     # HSV-Value augmentation
}

# Start training with optimized parameters
print("Starting optimized YOLOv8 training for potato leaf segmentation...")
print(f"Training on: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
print(f"Available CUDA devices: {torch.cuda.device_count()}")

results = model.train(**training_params)

# Post-training optimizations
print("\nTraining completed! Applying post-training optimizations...")

# Export optimized models for Hailo-8L
best_model = YOLO(f'potato_yolov8_seg_optimized/experiment_optimized/weights/best.pt')

# Export to ONNX (Hailo optimized: Opset 11, static shape, simplified)
try:
    print(f"Exporting to ONNX (Hailo-Optimized)...")
    best_model.export(
        format='onnx', 
        imgsz=640, 
        simplify=True, 
        opset=11, 
        dynamic=False, 
        half=False
    )
    print(f"✓ ONNX export successful (FP32, Opset 11, 640x640)")
except Exception as e:
    print(f"✗ ONNX export failed: {e}")

print("\nOptimization complete! Model ready for Hailo-8L deployment on Raspberry Pi 5.")




model = YOLO('/kaggle/working/potato_yolov8_seg_optimized/experiment_optimized/weights/best.pt')

# Inference on test images
results = model.predict(source='/kaggle/input/sigmapotato/test/images', save=True, imgsz=640, conf=0.3)
import shutil

shutil.make_archive('predict', 'zip', '/kaggle/working/runs/segment/predict')
print("Predictions zipped as predictions.zip")