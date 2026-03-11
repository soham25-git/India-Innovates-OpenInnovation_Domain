from ultralytics import YOLO
import sys
import os

def verify_yolo():
    print("--- Verifying YOLOv8n-seg Export (Opset 11, 640x640, Static) ---")
    try:
        # Load a base model for export testing (will download base weights)
        model = YOLO('yolov8n-seg.pt')
        
        # Test export logic
        print("Running dummy export...")
        model.export(
            format='onnx', 
            imgsz=640, 
            simplify=True, 
            opset=11, 
            dynamic=False
        )
        print("✅ YOLO export logic verified (Opset 11, 640 static)")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

if __name__ == "__main__":
    if not verify_yolo():
        sys.exit(1)
