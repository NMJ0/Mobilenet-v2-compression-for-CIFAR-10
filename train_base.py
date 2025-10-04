import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.transforms import AutoAugmentPolicy
import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import models
# Set random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomCrop(32, padding=6, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.08),
        transforms.RandomRotation(15),
        transforms.AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]),
}

image_datasets = {
    'train': datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transforms['train']),
    'val': datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transforms['val'])
}

dataloaders = {
    x: torch.utils.data.DataLoader(image_datasets[x],
                                   batch_size=128,
                                   shuffle=True if x == 'train' else False,
                                   num_workers=8,
                                   pin_memory=True)
    for x in ['train', 'val']
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.mobilenet_v2()
model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)

for idx, block in enumerate(model.features):
    for m in block.modules():
        if isinstance(m, nn.Conv2d) and m.stride == (2, 2):
            if idx in [2,]:
                m.stride = (1, 1)

model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.last_channel, 128),
    nn.BatchNorm1d(128),
    nn.SiLU(inplace=True),
    nn.Dropout(0.3),
    nn.Linear(128, 10)
)
model = model.to(device)

def make_scheduler(optimizer, num_epochs):
    return optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=6e-4,  # Higher LR to compensate for small classifier
        steps_per_epoch=len(dataloaders['train']),
        epochs=num_epochs,
        pct_start=0.1,
        div_factor=10,
        final_div_factor=100
    )

def rand_beta(alpha=1.0):
    if alpha > 0:
        return torch.distributions.Beta(alpha, alpha).sample().item()
    return 1.0

def mixup_data(x, y, alpha=0.15):  # Low mixup
    lam = rand_beta(alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.5):  # High cutmix
    lam = rand_beta(alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = math.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = torch.randint(W, (1,)).item()
    cy = torch.randint(H, (1,)).item()
    
    bbx1 = max(0, cx - cut_w // 2)
    bby1 = max(0, cy - cut_h // 2)
    bbx2 = min(W, cx + cut_w // 2)
    bby2 = min(H, cy + cut_h // 2)

    return bbx1, bby1, bbx2, bby2

def interpolation_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def plot_training_curves(train_losses, train_accuracies, val_losses, val_accuracies, save_path='training_curves_v5.png'):
    """Plot training curves during training"""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(15, 10))
    
    # Loss curves
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves - Version 5 (Minimal)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy curves
    plt.subplot(2, 2, 2)
    # Convert to percentage if needed
    train_acc_plot = [acc * 100 if acc <= 1 else acc for acc in train_accuracies]
    val_acc_plot = [acc * 100 if acc <= 1 else acc for acc in val_accuracies]
    
    plt.plot(epochs, train_acc_plot, 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, val_acc_plot, 'r-', label='Validation Accuracy', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curves - Version 5 (Minimal)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Learning rate curve
    plt.subplot(2, 2, 3)
    if len(epochs) > 1:
        # Approximate LR curve for OneCycleLR
        max_lr = 6e-4
        total_epochs = len(epochs)
        pct_start = 0.1
        
        lrs = []
        for epoch in epochs:
            if epoch <= pct_start * total_epochs:
                # Warm-up phase
                lr = max_lr * (epoch / (pct_start * total_epochs))
            else:
                # Annealing phase
                progress = (epoch - pct_start * total_epochs) / ((1 - pct_start) * total_epochs)
                lr = max_lr * (1 - progress)
                
            lrs.append(lr)
        
        plt.plot(epochs, lrs, 'g-', label='Learning Rate', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
    
    # Loss comparison (zoomed)
    plt.subplot(2, 2, 4)
    if len(epochs) > 20:
        start_idx = len(epochs) // 4  # Start from 25% of training
        plt.plot(epochs[start_idx:], train_losses[start_idx:], 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs[start_idx:], val_losses[start_idx:], 'r-', label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Loss Curves (from epoch {start_idx + 1})')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Training curves saved as '{save_path}'")

def train_model(num_epochs=300, save_path='mobilenetv2_minimal.pth'):
    best_acc = 0.0
    patience = 150
    epochs_without_improvement = 0
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.features[0][0].weight.requires_grad = True
    model.features[0][1].weight.requires_grad = True
    model.features[0][1].bias.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=6e-4,           
                            weight_decay=0.015)
    scheduler = make_scheduler(optimizer, num_epochs)
    
    # Fast unfreezing schedule (every 5 epochs)
    unfreeze_phases = [
        (5, [18, 17]),
        (10, [16, 15]),
        (15, [14, 13]),
        (20, [12, 11]),
        (25, [10, 9]),
        (30, [8, 7]),
        (35, [6, 5]),
        (40, [4, 3]),
        (45, [2, 1]),
        (50, [0]),
    ]
    unfrozen_layers = set()

    # Lists to store training curves
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    print("="*60)
    print("VERSION 5 - MINIMAL CLASSIFIER TRAINING:")
    print("="*60)
    print(f"Max LR: 6e-4")
    print(f"Weight Decay: 0.015")
    print(f"Classifier: 128 hidden units (minimal)")
    print(f"Mixup/CutMix: First 25% epochs")
    print(f"Unfreeze Interval: 5 epochs (fast)")
    print(f"Random Seed: {seed}")
    print("="*60)

    mixup_cutmix_epochs = int(num_epochs * 0.25)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        
        for phase_epoch, layer_idxs in unfreeze_phases:
            if epoch == phase_epoch:
                print(f"🔓 Unfreezing layers: {layer_idxs} at epoch {epoch}...")
                for idx in layer_idxs:
                    for param in model.features[idx].parameters():
                        param.requires_grad = True
                    unfrozen_layers.add(idx)
                optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                        lr=scheduler.get_last_lr()[0],
                                        weight_decay=0.015)

        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            if epoch < mixup_cutmix_epochs:
                if random.random() < 0.5:
                    inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=0.15)
                else:
                    inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, alpha=1.5)
                outputs = model(inputs)
                loss = interpolation_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += labels.size(0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = running_corrects.double() / total_samples if epoch >= mixup_cutmix_epochs else 0.0
        
        # Store training metrics
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc.item() if isinstance(epoch_acc, torch.Tensor) else epoch_acc)
        
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss /= dataset_sizes['val']
        val_acc = val_corrects.double() / dataset_sizes['val']
        
        # Store validation metrics
        val_losses.append(val_loss)
        val_accuracies.append(val_acc.item() if isinstance(val_acc, torch.Tensor) else val_acc)
        
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        current_lr = scheduler.get_last_lr()[0]
        print(f"Current LR: {current_lr:.8f}")

        if val_acc > best_acc:
            best_acc = val_acc
            epochs_without_improvement = 0
            # Save model with training curves
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'train_losses': train_losses,
                'train_accuracies': train_accuracies,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies,
                'seed': seed,
            }, save_path)
            print(f"✓ Best model saved with accuracy: {best_acc:.4f}")
        else:
            epochs_without_improvement += 1

        # Plot curves every 20 epochs or at the end
        if (epoch + 1) % 20 == 0 or epoch == num_epochs - 1:
            plot_training_curves(train_losses, train_accuracies, val_losses, val_accuracies)

        if epochs_without_improvement >= patience and epoch >= 50:
            print(f"\n⚠️ Early stopping triggered after {patience} epochs without improvement")
            break

    print(f"\nTraining finished. Best val accuracy: {best_acc:.4f}")

    plot_training_curves(train_losses, train_accuracies, val_losses, val_accuracies, 'final_training_curves_v5.png')

    print("\n" + "="*60)
    print("📊 VERSION 5 TRAINING SUMMARY")
    print("="*60)

    print(f"Total epochs: {len(train_losses)}")
    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    if val_accuracies:
        final_val_acc = val_accuracies[-1] * 100 if val_accuracies[-1] <= 1 else val_accuracies[-1]
        best_val_acc = max(val_accuracies) * 100 if max(val_accuracies) <= 1 else max(val_accuracies)
        best_epoch = val_accuracies.index(max(val_accuracies)) + 1
        print(f"Final validation accuracy: {final_val_acc:.2f}%")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        print(f"Best accuracy achieved at epoch: {best_epoch}")
    print(f"Seed used: {seed}")
    print(f"Model saved: {save_path}")
    print(f"Curves saved: final_training_curves_v5.png")
    print("="*60)
    

    return best_acc

if __name__ == "__main__":
 
    final_acc = train_model(num_epochs=300, save_path='base_model.pth')
    print(f"🎯 Final accuracy: {final_acc:.4f}")
