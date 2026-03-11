import timm
import torch
import sys

def verify():
    print("--- Verifying MobileNet-V2 Creation ---")
    try:
        model = timm.create_model('mobilenetv2_100', pretrained=False, num_classes=7)
        print("✅ Successfully created mobilenetv2_100 with 7 classes")
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        print(f"✅ Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

if __name__ == "__main__":
    if not verify():
        sys.exit(1)
