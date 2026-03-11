import torch
import timm
import os
import sys

def verify_export():
    print("--- Verifying ONNX Export (Opset 11, Static) ---")
    try:
        model = timm.create_model('mobilenetv2_100', pretrained=False, num_classes=7)
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        torch.onnx.export(
            model, 
            dummy_input, 
            "test.onnx", 
            opset_version=11, 
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=None
        )
        print("✅ Successfully exported to ONNX (Opset 11, static)")
        if os.path.exists("test.onnx"):
            os.remove("test.onnx")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

if __name__ == "__main__":
    if not verify_export():
        sys.exit(1)
