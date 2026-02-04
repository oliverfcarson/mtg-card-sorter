"""
Train Set Symbol CNN Classifier

This script trains a ResNet18 CNN to classify set symbols.
Uses data augmentation to improve robustness to real-world capture conditions.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
import json
from datetime import datetime
from glob import glob
import random

# ============== CONFIGURATION ==============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "set_symbols_cropped")
MODEL_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "models")

# Training parameters
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
MIN_SAMPLES_PER_CLASS = 10  # Minimum images required per set

# ============== DATA TRANSFORMS ==============
# Training transforms with heavy augmentation
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(15),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
        shear=5
    ),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.2,
        hue=0.1
    ),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
])

# Validation transforms (no augmentation)
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ============== DATASET CLASS ==============
class SetSymbolDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            # Return a blank image if loading fails
            img = Image.new('RGB', (224, 224), (128, 128, 128))
            if self.transform:
                img = self.transform(img)
            return img, label


def main():
    """Main training function - must be called from if __name__ == '__main__' on Windows"""

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 50)
    print("MTG Set Symbol Classifier Training")
    print("=" * 50)
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not available. Training will be slow on CPU.")
        print("To enable GPU, install PyTorch with CUDA:")
        print("  pip uninstall torch torchvision")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

    # ============== LOAD DATASET ==============
    print("\n[1/4] Loading dataset...")

    if not os.path.exists(DATASET_DIR):
        print(f"ERROR: Dataset directory not found: {DATASET_DIR}")
        print("Run crop_set_symbols.py first!")
        return

    # Manually scan for valid images
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    class_counts = {}
    all_samples = []

    set_dirs = [d for d in glob(os.path.join(DATASET_DIR, "*")) if os.path.isdir(d)]
    print(f"       Scanning {len(set_dirs)} set directories...")

    for set_dir in set_dirs:
        set_code = os.path.basename(set_dir)
        images = []
        for ext in VALID_EXTENSIONS:
            images.extend(glob(os.path.join(set_dir, f"*{ext}")))
            images.extend(glob(os.path.join(set_dir, f"*{ext.upper()}")))

        if images:
            class_counts[set_code] = len(images)
            for img_path in images:
                all_samples.append((img_path, set_code))

    print(f"       Found {len(class_counts)} classes with images")
    print(f"       Total images found: {len(all_samples):,}")

    # Filter out classes with too few samples
    valid_classes = [c for c, count in class_counts.items() if count >= MIN_SAMPLES_PER_CLASS]
    print(f"       {len(valid_classes)} classes have >= {MIN_SAMPLES_PER_CLASS} samples")

    if len(valid_classes) == 0:
        print("ERROR: No classes have enough samples!")
        return

    # Create mapping
    class_to_idx = {c: i for i, c in enumerate(sorted(valid_classes))}
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    # Filter samples
    filtered_samples = [
        (path, class_to_idx[class_name])
        for path, class_name in all_samples
        if class_name in valid_classes
    ]

    print(f"       Total samples after filtering: {len(filtered_samples):,}")

    # Split into train/val
    num_samples = len(filtered_samples)
    num_val = int(num_samples * VALIDATION_SPLIT)
    num_train = num_samples - num_val

    random.shuffle(filtered_samples)
    train_samples = filtered_samples[:num_train]
    val_samples = filtered_samples[num_train:]

    train_dataset = SetSymbolDataset(train_samples, train_transforms)
    val_dataset = SetSymbolDataset(val_samples, val_transforms)

    print(f"       Training samples: {len(train_dataset):,}")
    print(f"       Validation samples: {len(val_dataset):,}")

    # Create data loaders (num_workers=0 for Windows compatibility)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Use 0 for Windows compatibility
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # ============== CREATE MODEL ==============
    print("\n[2/4] Creating ResNet18 model...")

    num_classes = len(valid_classes)
    print(f"       Number of classes: {num_classes}")

    # Load pretrained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    print("       Model loaded and modified for set classification")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    # ============== TRAINING ==============
    print("\n[3/4] Training...")

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % 100 == 0:
                print(f"       Epoch [{epoch+1}/{NUM_EPOCHS}] Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")

        train_acc = 100.0 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        scheduler.step(avg_val_loss)

        print(f"\n  Epoch {epoch+1}/{NUM_EPOCHS}:")
        print(f"    Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"    Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"    LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"    ** New best model! **")

    # ============== SAVE MODEL ==============
    print("\n[4/4] Saving model...")

    model.load_state_dict(best_model_state)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_OUTPUT_DIR, f"set_classifier_{timestamp}.pth")

    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class,
        'best_val_acc': best_val_acc,
    }, model_path)

    print(f"       Model saved to: {model_path}")

    latest_path = os.path.join(MODEL_OUTPUT_DIR, "set_classifier_latest.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class,
        'best_val_acc': best_val_acc,
    }, latest_path)

    print(f"       Also saved as: {latest_path}")

    mapping_path = os.path.join(MODEL_OUTPUT_DIR, "set_classes.json")
    with open(mapping_path, 'w') as f:
        json.dump({
            'class_to_idx': class_to_idx,
            'idx_to_class': {str(k): v for k, v in idx_to_class.items()},
            'num_classes': num_classes
        }, f, indent=2)

    print(f"       Class mapping saved to: {mapping_path}")

    print(f"\n{'=' * 50}")
    print(f"Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Number of set classes: {num_classes}")
    print(f"Model saved to: {model_path}")
    print(f"{'=' * 50}")


if __name__ == '__main__':
    main()
