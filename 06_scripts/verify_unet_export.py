import torch
import segmentation_models_pytorch as smp
import sys
import os

def verify_unet():
    print("--- Verifying UNet-MobileNetV2 Export (Opset 11, 256x256, Static) ---")
    try:
        model = smp.Unet(
            encoder_name="mobilenet_v2",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            decoder_channels=(256, 128, 64, 32, 16)
        )
        model.eval()
        dummy_input = torch.randn(1, 3, 256, 256)
        
        print("Running dummy export...")
        torch.onnx.export(
            model,
            dummy_input,
            "test_unet.onnx",
            opset_version=11,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=None
        )
        print("✅ UNet export logic verified (Opset 11, 256 static)")
        if os.path.exists("test_unet.onnx"):
            os.remove("test_unet.onnx")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

if __name__ == "__main__":
    if not verify_unet():
        sys.exit(1)
