!pip install -q timm matplotlib seaborn scikit-learn tqdm albumentations
# Trained on Kaggle
# Potato Leaf Disease Dataset in Uncontrolled Environment from kaggle itself
# Issues in support for model conversion to HEF format
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
import timm
from tqdm import tqdm
from collections import Counter
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import warnings
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
warnings.filterwarnings('ignore')

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

# ==============================================================================
# CONFIGURATION - OPTIMIZED FOR HAILO 8L + RPi5 DEPLOYMENT
# ==============================================================================

class CFG:
    SEED = 42
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Dataset path
    DATA_DIR = "/kaggle/input/potato-leaf-disease-dataset/Potato Leaf Disease Dataset in Uncontrolled Environment"
    
    # Model Architecture (Hailo 8L optimized - lightweight!)
    MODEL_NAME = "mobilenetv2_100"  # Switched from efficientnet_b0 for Hailo compatibility
    
    # Training config - OPTIMIZED FOR KAGGLE T4
    NUM_CLASSES = 7  # Bacteria, Fungi, Healthy, Phytophthora, Pests, Virus
    IMAGE_SIZE = 224
    BATCH_SIZE = 48  
    NUM_WORKERS = 4
    EPOCHS = 30
    LR = 1e-3
    
    OUTPUT_MODEL_NAME = "best_potato_classifier.pth"
    ONNX_MODEL_NAME = "disease_classifier_mobilenet.onnx"
    
    # Mixed precision
    USE_AMP = True  
    
    # Gradient clipping
    GRAD_CLIP = 1.0
    
    # Early stopping
    PATIENCE = 5
    
    # Set seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

# ==============================================================================
# CUSTOM TRANSFORM: GRID TILING SIMULATOR (for your 4x4 grid deployment)
# ==============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        # Apply only to tensor images (after ToTensor)
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class RandomGridOrCrop:
    """
    Crucial for your pipeline!
    50% grid mode: Simulates your 4x4 grid deployment
    50% standard mode: Regular random crop augmentation
    """
    def __init__(self, size=224, p_grid=0.5, grid_size=4):
        self.size = size
        self.p_grid = p_grid
        self.grid_size = grid_size
    
    def __call__(self, img):
        if random.random() < self.p_grid:
            # GRID TILE MODE (Simulates Deployment)
            w, h = img.size
            tile_w = w // self.grid_size
            tile_h = h // self.grid_size
            gx = random.randint(0, self.grid_size - 1)
            gy = random.randint(0, self.grid_size - 1)
            left = gx * tile_w
            upper = gy * tile_h
            right = left + tile_w
            lower = upper + tile_h
            img = img.crop((left, upper, right, lower))
        else:
            # STANDARD RANDOM RESIZED CROP
            img = TF.resized_crop(
                img,
                top=random.randint(0, max(0, img.height - int(0.6 * img.height))),
                left=random.randint(0, max(0, img.width - int(0.6 * img.width))),
                height=int(0.6 * img.height),
                width=int(0.6 * img.width),
                size=(self.size, self.size)
            )
        
        img = TF.resize(img, (self.size, self.size))
        return img

# ==============================================================================
# DATA LOADERS - BALANCED AUGMENTATION
# ==============================================================================

class AlbumentationsWrapper:
    """Wrapper to use Albumentations with torchvision ImageFolder"""
    def __init__(self, transform):
        self.transform = transform
        
    def __call__(self, img):
        img_np = np.array(img)
        augmented = self.transform(image=img_np)
        return augmented['image']

def get_data_loaders(data_dir):
    """Load datasets with balanced augmentation and environmental robustness"""
    
    # Standard torchvision resize/crop logic before albumentations
    # (Matches your existing pipeline logic)
    
    # Enhanced Training Transforms with Robustness Effects
    train_transform_logic = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=35, p=0.3),
        # === ENVIRONMENTAL ROBUSTNESS ===
        A.RandomRain(p=0.1, slant_lower=-10, slant_upper=10, rain_type='heavy'),
        A.MotionBlur(blur_limit=7, p=0.2),
        A.OpticalFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=0.2), # "Waxy" glare
        # ================================
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.4),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # Grid logic preserved via torchvision-style wrapper
    train_transform = transforms.Compose([
        RandomGridOrCrop(size=CFG.IMAGE_SIZE, p_grid=0.5, grid_size=4),
        AlbumentationsWrapper(train_transform_logic)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(CFG.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load Dataset
    full_dataset = datasets.ImageFolder(root=data_dir)
    class_names = full_dataset.classes
    
    print(f"✅ Classes Found: {class_names}")
    print(f"📊 Total Images: {len(full_dataset)}")
    
    # Split into train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(CFG.SEED)
    )
    
    # Apply transforms
    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform = val_transform
    
    # Weighted Sampler for Class Imbalance
    train_indices = train_ds.indices
    train_labels = [full_dataset.targets[i] for i in train_indices]
    class_counts = Counter(train_labels)
    
    print(f"\n📈 Class Distribution:")
    for cls_idx, cls_name in enumerate(class_names):
        count = class_counts.get(cls_idx, 0)
        print(f"  {cls_name}: {count} images")
    
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, sampler=sampler, 
                             num_workers=CFG.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False, 
                           num_workers=CFG.NUM_WORKERS, pin_memory=True)
    
    return train_loader, val_loader, class_names

# ==============================================================================
# TRAINING ENGINE
# ==============================================================================

def train_one_epoch(model, loader, criterion, optimizer, 
                   scaler=None, grad_clip=None, use_amp=True):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(CFG.DEVICE, dtype=torch.float32)
        labels = labels.to(CFG.DEVICE)
        
        optimizer.zero_grad()
        
        # Use AMP only if not in QAT mode
        if scaler is not None and use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Full precision (critical for QAT!)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(loader), correct / total

def validate_one_epoch(model, loader, criterion):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images = images.to(CFG.DEVICE, dtype=torch.float32)
            labels = labels.to(CFG.DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return total_loss / len(loader), correct / total, f1

def train_normal(model, train_loader, val_loader):
    """PHASE 1: Normal training with mixed precision"""
    print("\n" + "="*70)
    print("PHASE 1: NORMAL TRAINING (30 epochs)")
    print("="*70)
    
    # criterion = nn.CrossEntropyLoss()
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=CFG.LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                     mode='max', 
                                                     factor=0.5, 
                                                     patience=3, 
                                                     verbose=False)
    
    scaler = torch.cuda.amp.GradScaler() if CFG.USE_AMP else None
    
    best_val_acc = 0.0
    best_val_f1 = 0.0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(CFG.EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer,
                                               scaler=scaler, grad_clip=CFG.GRAD_CLIP,
                                               use_amp=CFG.USE_AMP)
        
        val_loss, val_acc, val_f1 = validate_one_epoch(model, val_loader, criterion)
        
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch+1}/{CFG.EPOCHS} - "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            torch.save(model.state_dict(), CFG.OUTPUT_MODEL_NAME)
            print(f"✅ Best model saved (Val Acc: {val_acc:.4f}, F1: {val_f1:.4f})")
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


def final_evaluation(model, loader, class_names):
    """Comprehensive final evaluation"""
    print("\n" + "="*70)
    print("FINAL EVALUATION METRICS")
    print("="*70)
    
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(CFG.DEVICE, dtype=torch.float32)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    print("\n" + "="*70)
    print("📊 DETAILED CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_classifier.png', dpi=100, bbox_inches='tight')
    print("✅ Confusion matrix saved to confusion_matrix_classifier.png")

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
    print(f"🎯 Model: MobileNetV2 (lightweight, edge-optimized)")
    print(f"📊 Image Size: {CFG.IMAGE_SIZE}x{CFG.IMAGE_SIZE}")
    print(f"🔋 Batch Size: {CFG.BATCH_SIZE} (training)")
    
    # Load datasets
    print("\n📂 Loading datasets...")
    train_loader, val_loader, class_names = get_data_loaders(CFG.DATA_DIR)
    
    # Create model
    print("\n🧠 Creating model...")
    model = timm.create_model(
        CFG.MODEL_NAME,
        pretrained=True,
        num_classes=CFG.NUM_CLASSES
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
    
    final_evaluation(model, val_loader, class_names)
    
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