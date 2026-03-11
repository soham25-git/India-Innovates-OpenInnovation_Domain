!pip install segmentation-models-pytorch scikit-learn tqdm numpy pandas matplotlib tensorboard -q
# trained on kaggle
# used leaf disease segmentation dataset and plantsegv3 segmentation data
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from sklearn.metrics import precision_score, recall_score, f1_score
import gc
import warnings
warnings.filterwarnings('ignore')

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

# ==============================================================================
# CONFIGURATION - OPTIMIZED FOR HAILO 8L + RPi5 DEPLOYMENT
# ==============================================================================

class CFG:
    SEED = 42
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Dataset paths
    DATA_DIRS = [
        "/kaggle/input/plantsegv3",
        "/kaggle/input/leaf-disease-segmentation-dataset/aug_data/aug_data"
    ]
    
    # Model Architecture (Hailo 8L optimized - lightweight!)
    ARCH = "Unet"
    ENCODER = "mobilenet_v2"  # Lightweight for edge deployment
    ENCODER_WEIGHTS = "imagenet"
    DECODER_CHANNELS = (256, 128, 64, 32, 16)
    
    # Training config - OPTIMIZED FOR KAGGLE T4
    NUM_CLASSES = 1
    IMAGE_SIZE = 256  # Alignment with Hailo model zoo
    BATCH_SIZE = 48  
    NUM_WORKERS = 4
    EPOCHS = 30
    LR = 1e-3
    
    OUTPUT_MODEL_NAME = "best_binary_segmentation.pth"
    ONNX_MODEL_NAME = "lesion_segmentation.onnx"
    
    # Mixed precision
    USE_AMP = True  
    
    # Gradient clipping
    GRAD_CLIP = 1.0
    
    # Early stopping
    PATIENCE = 5
    
    # Metrics calculation frequency
    CALC_METRICS_EVERY = 10
    
    # Set seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)

# ==============================================================================
# DATASET - OPTIMIZED
# ==============================================================================

class BinaryLesionDataset(Dataset):
    def __init__(self, root_dirs, split="train", transform=None):
        self.transform = transform
        self.samples = []
        
        for root in root_dirs:
            # Find image and mask directories
            possible_img_dirs = [
                os.path.join(root, "images", split),
                os.path.join(root, split, "images"),
                os.path.join(root, "images")
            ]
            
            img_folder = None
            for d in possible_img_dirs:
                if os.path.exists(d):
                    img_folder = d
                    break
            
            if not img_folder:
                continue
            
            mask_folder = img_folder.replace("images", "masks")
            
            # Find all image-mask pairs
            files = sorted([f for f in os.listdir(img_folder)
                          if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            
            for f in files:
                self.samples.append({
                    "img_path": os.path.join(img_folder, f),
                    "mask_path": os.path.join(mask_folder, os.path.splitext(f)[0] + ".png"),
                    "mask_path_fallback": os.path.join(mask_folder, f)
                })
        
        print(f"✅ Found {len(self.samples)} images in {split} set")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load Image
        image = cv2.imread(sample["img_path"])
        if image is None:
            return torch.zeros(3, CFG.IMAGE_SIZE, CFG.IMAGE_SIZE), \
                   torch.zeros(1, CFG.IMAGE_SIZE, CFG.IMAGE_SIZE)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load Mask
        mask_path = sample["mask_path"]
        if not os.path.exists(mask_path):
            mask_path = sample["mask_path_fallback"]
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        else:
            mask = np.where(mask > 0, 1.0, 0.0).astype(np.float32)
        
        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Ensure mask is [1, H, W]
        if mask.ndim == 2:
            mask = np.expand_dims(mask, 0)
        
        return image, mask.astype(np.float32)

# ==============================================================================
# AUGMENTATIONS - BALANCED FOR ACCURACY
# ==============================================================================

def get_transforms(stage):
    if stage == "train":
        return A.Compose([
            A.Resize(CFG.IMAGE_SIZE, CFG.IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=35, p=0.3),
            # === ENVIRONMENTAL ROBUSTNESS ===
            A.RandomRain(p=0.1, slant_lower=-10, slant_upper=10, rain_type='heavy'),
            A.MotionBlur(blur_limit=7, p=0.2),
            A.OpticalFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=0.2), # "Waxy" glare
            # ================================
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(p=0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], additional_targets={'mask': 'mask'})
    else:
        return A.Compose([
            A.Resize(CFG.IMAGE_SIZE, CFG.IMAGE_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], additional_targets={'mask': 'mask'})

# ==============================================================================
# METRICS
# ==============================================================================

def calculate_metrics_fast(loader, model, threshold=0.5, max_batches=10):
    """FAST metrics on subset of data"""
    model.eval()
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            
            images = images.to(CFG.DEVICE, dtype=torch.float32)
            masks = masks.to(CFG.DEVICE, dtype=torch.float32)
            
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()
            
            all_targets.extend(masks.cpu().numpy().flatten())
            all_preds.extend(preds.cpu().numpy().flatten())
    
    all_targets = np.array(all_targets).astype(int)
    all_preds = np.array(all_preds).astype(int)
    
    intersection = np.logical_and(all_targets, all_preds).sum()
    union = np.logical_or(all_targets, all_preds).sum()
    iou = intersection / (union + 1e-6)
    
    return iou

def calculate_metrics_full(loader, model, threshold=0.5):
    """FULL metrics on entire dataset"""
    model.eval()
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Full Metrics", leave=False):
            images = images.to(CFG.DEVICE, dtype=torch.float32)
            masks = masks.to(CFG.DEVICE, dtype=torch.float32)
            
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()
            
            all_targets.extend(masks.cpu().numpy().flatten())
            all_preds.extend(preds.cpu().numpy().flatten())
    
    all_targets = np.array(all_targets).astype(int)
    all_preds = np.array(all_preds).astype(int)
    
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    intersection = np.logical_and(all_targets, all_preds).sum()
    union = np.logical_or(all_targets, all_preds).sum()
    iou = intersection / (union + 1e-6)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou
    }

# ==============================================================================
# TRAINING ENGINE
# ==============================================================================

def train_one_epoch(model, loader, criterion, optimizer, 
                   scaler=None, grad_clip=None, use_amp=True):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, masks in pbar:
        images = images.to(CFG.DEVICE, dtype=torch.float32)
        masks = masks.to(CFG.DEVICE, dtype=torch.float32)
        
        optimizer.zero_grad()
        
        # Use AMP only if not in QAT mode
        if scaler is not None and use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Full precision (critical for QAT!)
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(loader)

def validate_one_epoch(model, loader, criterion):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation", leave=False):
            images = images.to(CFG.DEVICE, dtype=torch.float32)
            masks = masks.to(CFG.DEVICE, dtype=torch.float32)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
    
    return total_loss / len(loader)

def train_normal(model, train_loader, val_loader):
    """PHASE 1: Normal training with mixed precision"""
    print("\n" + "="*70)
    print("PHASE 1: NORMAL TRAINING (30 epochs)")
    print("="*70)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=CFG.LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                     mode='min', 
                                                     factor=0.5, 
                                                     patience=3, 
                                                     verbose=True)
    
    scaler = torch.cuda.amp.GradScaler() if CFG.USE_AMP else None
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(CFG.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                    scaler=scaler, grad_clip=CFG.GRAD_CLIP,
                                    use_amp=CFG.USE_AMP)
        
        val_loss = validate_one_epoch(model, val_loader, criterion)
        
        scheduler.step(val_loss)
        
        # Calculate full metrics every N epochs
        if (epoch + 1) % CFG.CALC_METRICS_EVERY == 0:
            metrics = calculate_metrics_full(val_loader, model)
            print(f"Epoch {epoch+1}/{CFG.EPOCHS} - "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"IoU: {metrics['iou']:.4f} | F1: {metrics['f1']:.4f}")
        else:
            print(f"Epoch {epoch+1}/{CFG.EPOCHS} - "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            torch.save(model.state_dict(), CFG.OUTPUT_MODEL_NAME)
            print(f"✅ Best model saved (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= CFG.PATIENCE:
                print(f"⚠️  Early stopping at epoch {epoch+1}")
                break
        
        gc.collect()
        torch.cuda.empty_cache()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model


def export_to_onnx(model):
    """Export model to ONNX for Hailo compiler (FP32, static shapes, opset 11)"""
    print("\n" + "="*70)
    print(f"EXPORTING TO ONNX (Hailo-Optimized)")
    print("="*70)
    
    model.eval()
    
    # Use static input shape for Hailo compatibility
    dummy_input = torch.randn(1, 3, CFG.IMAGE_SIZE, CFG.IMAGE_SIZE, 
                             device=CFG.DEVICE)
    output_name = CFG.ONNX_MODEL_NAME
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_name,
            opset_version=11,  # Standard for Hailo toolchain
            input_names=['input'],
            output_names=['output'],
            verbose=False,
            dynamic_axes=None  # Strictly static for NPU efficiency
        )
        print(f"✅ ONNX model exported: {output_name}")
        print(f"📊 Model ready for Hailo compiler (FP32, Opset 11, static)!")
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print(f"🚀 Device: {CFG.DEVICE}")
    print(f"🎯 Model: {CFG.ARCH} with {CFG.ENCODER} encoder")
    print(f"📊 Image Size: {CFG.IMAGE_SIZE}x{CFG.IMAGE_SIZE}")
    print(f"🔋 Batch Size: {CFG.BATCH_SIZE} (training)")
    
    # Load datasets
    print("\n📂 Loading datasets...")
    train_dataset = BinaryLesionDataset(CFG.DATA_DIRS, split="train", 
                                       transform=get_transforms("train"))
    val_dataset = BinaryLesionDataset(CFG.DATA_DIRS, split="val", 
                                     transform=get_transforms("val"))
    
    train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE,
                             shuffle=True, num_workers=CFG.NUM_WORKERS,
                             pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CFG.BATCH_SIZE,
                           shuffle=False, num_workers=CFG.NUM_WORKERS,
                           pin_memory=True)
    
    # Create model
    print("\n🧠 Creating model...")
    model = smp.Unet(
        encoder_name=CFG.ENCODER,
        encoder_weights=CFG.ENCODER_WEIGHTS,
        in_channels=3,
        classes=CFG.NUM_CLASSES,
        decoder_channels=CFG.DECODER_CHANNELS
    )
    
    model = model.to(CFG.DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📈 Total Parameters: {total_params:,}")
    print(f"📈 Trainable Parameters: {trainable_params:,}")
    
    # PHASE 1: Normal Training
    model = train_normal(model, train_loader, val_loader)
    
    # PHASE 2: Final Evaluation
    print("\n" + "="*70)
    print("PHASE 2: FINAL EVALUATION")
    print("="*70)
    
    print("\n📊 Final validation metrics:")
    metrics = calculate_metrics_full(val_loader, model)
    print(f"IoU: {metrics['iou']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    
    # Export to ONNX
    export_to_onnx(model)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"📦 Models saved:")
    print(f"   - Normal: {CFG.OUTPUT_MODEL_NAME}")
    print(f"   - ONNX: {CFG.ONNX_MODEL_NAME}")
    print(f"\n🎯 Ready for Hailo 8L deployment on Raspberry Pi 5!")

if __name__ == "__main__":
    main()