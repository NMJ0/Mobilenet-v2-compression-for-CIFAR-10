import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import copy
import models

# -----------------------------
# CONFIGURATION
# -----------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = "base_model.pth"               
target_sparsity = 0.46                           # Final sparsity target (50% weights pruned)
num_iterations = 4                              # Number of iterative pruning stages
fine_tune_epochs = 12                           # Epochs per fine-tuning stage
batch_size = 128
learning_rate = 0.00008


# Calculate per-iteration pruning fraction
per_iter_fraction = 1 - (1 - target_sparsity) ** (1 / num_iterations)
print(f"Target sparsity: {target_sparsity*100:.1f}%")
print(f"Pruning {per_iter_fraction*100:.1f}% per iteration over {num_iterations} stages")


# -----------------------------
# DATASET
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
])
train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

val_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)


# -----------------------------
# MODEL LOADING
# -----------------------------
def create_mobilenetv2_v5():
    model = models.mobilenet_v2()
    model.features[0][0] = nn.Conv2d(3,32,3,1,1,bias=False)
    for idx, block in enumerate(model.features):
        for m in block.modules():
            if isinstance(m, nn.Conv2d) and m.stride==(2,2) and idx==2:
                m.stride=(1,1)
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.last_channel,128),
        nn.BatchNorm1d(128),
        nn.SiLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128,10)
    )
    return model


model = create_mobilenetv2_v5().to(device)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded model with baseline accuracy from checkpoint")


# -----------------------------
# VALIDATION FUNCTION
# -----------------------------
def validate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return 100.0 * correct / total


# -----------------------------
# FISHER INFORMATION ESTIMATION
# -----------------------------
def estimate_fisher(model, data_loader, criterion, num_batches=None):
    model.eval()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    
    batch_count = 0
    total_samples = 0
    
    for inputs, targets in data_loader:
        model.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        for n, p in model.named_parameters():
            if p.requires_grad:
                fisher[n] += p.grad.data.pow(2) * inputs.size(0)
        
        total_samples += inputs.size(0)
        batch_count += 1
        
        if num_batches and batch_count >= num_batches:
            break
    
    for n in fisher:
        fisher[n] /= total_samples
    
    return fisher


# -----------------------------
# ITERATIVE FISHER-BASED PRUNING
# -----------------------------
def prune_by_fisher_iterative(model, fisher, prune_fraction, existing_masks=None):
    if existing_masks is None:
        scores = torch.cat([f.view(-1) for f in fisher.values()])
    else:
        scores = []
        for n, f in fisher.items():
            if n in existing_masks:
                unpruned_scores = f[existing_masks[n] > 0]
                if unpruned_scores.numel() > 0:
                    scores.append(unpruned_scores.view(-1))
        
        if scores:
            scores = torch.cat(scores)
        else:
            return existing_masks
    
    if scores.numel() == 0:
        return existing_masks if existing_masks else {}
    
    threshold = torch.quantile(scores, prune_fraction)
    
    masks = {} if existing_masks is None else copy.deepcopy(existing_masks)
    
    for n, p in model.named_parameters():
        if p.requires_grad:
            if n not in masks:
                mask = (fisher[n] >= threshold).float()
            else:
                new_prunes = (fisher[n] < threshold).float()
                mask = masks[n] * (1 - new_prunes)
            
            masks[n] = mask
            p.data.mul_(mask)
    
    return masks


# -----------------------------
# ADAPTIVE FINE-TUNING
# -----------------------------
def adaptive_fine_tune(model, train_loader, val_loader, masks, epochs, base_lr, iteration):
    model.train()
    
    lr = base_lr * (0.8 ** iteration)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    print(f"  Fine-tuning with LR: {lr:.6f}")
    
    best_val_acc = 0
    best_model_state = None
    patience = 3
    patience_counter = 0
    
    for epoch in range(epochs):
        total_loss = 0.0
        model.train()
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in masks:
                        p.data.mul_(masks[n])
            
            total_loss += loss.item() * inputs.size(0)
        
        avg_loss = total_loss / len(train_loader.dataset)
        val_acc = validate_model(model, val_loader)
        
        print(f"    Epoch {epoch+1}/{epochs} – Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc and iteration ==3:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch >= epochs // 2:
                print(f"    Early stopping at epoch {epoch+1}")
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_val_acc


# -----------------------------
# SPARSITY CALCULATION
# -----------------------------
def calculate_sparsity(model, masks):
    total_params = 0
    pruned_params = 0
    
    for n, p in model.named_parameters():
        if p.requires_grad and n in masks:
            total_params += p.numel()
            pruned_params += (masks[n] == 0).sum().item()
    
    return pruned_params / total_params if total_params > 0 else 0


# -----------------------------
# MAIN ITERATIVE PRUNING PIPELINE
# -----------------------------
criterion = nn.CrossEntropyLoss()

initial_acc = validate_model(model, val_loader)
print(f"Initial validation accuracy: {initial_acc:.2f}%")

masks = None
results = []

for iteration in range(num_iterations):
    print(f"\n{'='*50}")
    print(f"ITERATION {iteration + 1}/{num_iterations}")
    print(f"{'='*50}")
    
    print("Estimating Fisher Information...")
    fisher = estimate_fisher(model, train_loader, criterion, num_batches=50)
    
    print(f"Pruning {per_iter_fraction*100:.1f}% of remaining weights...")
    masks = prune_by_fisher_iterative(model, fisher, per_iter_fraction, masks)
    
    current_sparsity = calculate_sparsity(model, masks)
    print(f"Current sparsity: {current_sparsity*100:.1f}%")
    
    print("Fine-tuning pruned model...")
    model, best_val_acc = adaptive_fine_tune(
        model, train_loader, val_loader, masks,
        fine_tune_epochs, learning_rate, iteration
    )
    
    results.append({
        'iteration': iteration + 1,
        'sparsity': current_sparsity,
        'val_accuracy': best_val_acc
    })
    
    print(f"Iteration {iteration + 1} complete - Sparsity: {current_sparsity*100:.1f}%, Val Acc: {best_val_acc:.2f}%")

final_sparsity = calculate_sparsity(model, masks)
final_acc = validate_model(model, val_loader)

print(f"\n{'='*60}")
print("ITERATIVE FISHER PRUNING COMPLETE")
print(f"{'='*60}")
print(f"Initial accuracy: {initial_acc:.2f}%")
print(f"Final accuracy:   {final_acc:.2f}%")
print(f"Final sparsity:   {final_sparsity*100:.1f}%")
print(f"Accuracy drop:    {initial_acc - final_acc:.2f}%")

print(f"\nIteration Results:")
for r in results:
    print(f"  Iter {r['iteration']}: {r['sparsity']*100:.1f}% sparse, {r['val_accuracy']:.2f}% acc")

torch.save({
    'model_state_dict': model.state_dict(),
    'masks': masks,
    'final_sparsity': final_sparsity,
    'final_accuracy': final_acc,
    'results': results
}, "pruned_model.pth")

print(f"\nFinal model saved to: pruned_model.pth")
