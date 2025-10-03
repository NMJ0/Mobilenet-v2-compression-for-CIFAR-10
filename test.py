import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import argparse
import os
import time
import numpy as np

try:
    from quantize import (
        swap_to_quant_modules,
        freeze_all_quant,
    )
    QUANTIZE_MODULE_AVAILABLE = True
except ImportError:
    print("Warning: 'quantize.py' not found. Quantized model testing will be unavailable.")
    QUANTIZE_MODULE_AVAILABLE = False

from collections import Counter, namedtuple
import heapq

# Simple Huffman implementation
class _Node:
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol, self.freq, self.left, self.right = symbol, freq, left, right
    def __lt__(self, other):
        return self.freq < other.freq

def _build_huffman_code(freqs):
    heap = [_Node(sym, freq) for sym,freq in freqs.items()]
    heapq.heapify(heap)
    if len(heap)==1:  # edge case
        heap.append(_Node(None,0))
    while len(heap)>1:
        a, b = heapq.heappop(heap), heapq.heappop(heap)
        heapq.heappush(heap, _Node(None, a.freq+b.freq, a, b))
    root = heap[0]
    codes = {}
    def _assign(node, prefix=""):
        if node.symbol is not None:
            codes[node.symbol] = prefix or "0"
        else:
            _assign(node.left, prefix+"0")
            _assign(node.right, prefix+"1")
    _assign(root)
    return codes

def measure_activation_compression(model, test_loader, act_bits=8, device=torch.device('cpu')):
    """
    Actually quantize activations to act_bits, Huffman-compress them, and compute ratio.
    """
    activations = []
    def hook(module, inp, out):
        # Simple uniform quantization to [0,2^act_bits)
        x = out.detach().cpu().flatten()
        x_min, x_max = x.min(), x.max()
        scale = (x_max - x_min)/(2**act_bits - 1) if x_max>x_min else 1.0
        q = torch.round((x - x_min)/scale).int().tolist()
        activations.extend(q)

    # register hooks
    handles = []
    for m in model.modules():
        if isinstance(m, (nn.ReLU, nn.ReLU6, nn.SiLU, nn.Conv2d, nn.Linear)):
            handles.append(m.register_forward_hook(hook))

    # run one batch
    model.eval()
    with torch.no_grad():
        for x,_ in test_loader:
            model(x.to(device))
            break

    for h in handles: h.remove()

    # build Huffman
    freqs = Counter(activations)
    codes = _build_huffman_code(freqs)

    # compute bit lengths
    uncompressed_bits = len(activations) * act_bits
    compressed_bits = sum(len(codes[val]) for val in activations)

    ratio = uncompressed_bits / compressed_bits if compressed_bits>0 else 0
    return len(activations), ratio, uncompressed_bits, compressed_bits

def create_mobilenetv2_v5(device):
    """Creates the MobileNetV2 V5 model architecture and moves it to the specified device."""
    model = models.mobilenet_v2(weights=None)
    model.features[0][0] = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
    for idx, block in enumerate(model.features):
        for m in block.modules():
            if isinstance(m, nn.Conv2d) and m.stride == (2, 2) and idx == 2:
                m.stride = (1, 1)
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.last_channel, 128),
        nn.BatchNorm1d(128),
        nn.SiLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, 10)
    )
    return model.to(device)

def prepare_test_loader(batch_size=128):
    """Prepares the CIFAR-10 test data loader."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

def analyze_file_contents(model, ckpt, model_path, weight_bits=None):
    """Analyze file contents and parameter breakdown"""
    print(f"\n{'='*60}")
    print("FILE CONTENT ANALYSIS")
    print(f"{'='*60}")
    
    # Get state dict
    state_dict = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    
    # Categorize parameters
    weights_size = 0
    biases_size = 0
    batchnorm_size = 0
    quantization_params_size = 0
    other_size = 0
    
    weights_count = 0
    biases_count = 0
    batchnorm_count = 0
    
    print(f"{'Parameter Type':<20} {'Count':<10} {'Size (KB)':<12} {'Size (MB)':<12}")
    print("-" * 60)
    
    for name, tensor in state_dict.items():
        size_bytes = tensor.numel() * tensor.element_size()
        
        if 'weight' in name and ('conv' in name.lower() or 'linear' in name.lower() or 'classifier' in name):
            weights_size += size_bytes
            weights_count += tensor.numel()
        elif 'bias' in name:
            biases_size += size_bytes
            biases_count += tensor.numel()
        elif any(bn_key in name for bn_key in ['bn', 'norm', 'running_mean', 'running_var', 'num_batches_tracked']):
            batchnorm_size += size_bytes
            batchnorm_count += tensor.numel()
        elif any(q_key in name for q_key in ['scale', 'zp', 'min_val', 'max_val']):
            quantization_params_size += size_bytes
        else:
            other_size += size_bytes
    
    print(f"{'Conv/Linear Weights':<20} {weights_count:<10,} {weights_size/1024:<12.2f} {weights_size/(1024*1024):<12.3f}")
    print(f"{'Biases':<20} {biases_count:<10,} {biases_size/1024:<12.2f} {biases_size/(1024*1024):<12.3f}")
    print(f"{'BatchNorm Params':<20} {batchnorm_count:<10,} {batchnorm_size/1024:<12.2f} {batchnorm_size/(1024*1024):<12.3f}")
    
    if quantization_params_size > 0:
        print(f"{'Quantization Params':<20} {'-':<10} {quantization_params_size/1024:<12.2f} {quantization_params_size/(1024*1024):<12.3f}")
    
    if other_size > 0:
        print(f"{'Other Parameters':<20} {'-':<10} {other_size/1024:<12.2f} {other_size/(1024*1024):<12.3f}")
    
    total_params_size = weights_size + biases_size + batchnorm_size + quantization_params_size + other_size
    print("-" * 60)
    print(f"{'TOTAL':<20} {'-':<10} {total_params_size/1024:<12.2f} {total_params_size/(1024*1024):<12.3f}")
    
    return weights_size, biases_size, batchnorm_size, quantization_params_size, other_size

def calculate_sparsity_metrics(model):
    """Calculate different types of sparsity"""
    print(f"\n{'='*60}")
    print("SPARSITY ANALYSIS")
    print(f"{'='*60}")
    
    # Total parameter sparsity
    total_params = 0
    total_zero_params = 0
    
    # Weight-only sparsity  
    weight_params = 0
    weight_zero_params = 0
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            param_count = param.numel()
            zero_count = torch.sum(param == 0).item()
            
            total_params += param_count
            total_zero_params += zero_count
            
            # Check if it's a weight parameter (multi-dimensional and contains 'weight')
            if param.dim() > 1 and 'weight' in name:
                weight_params += param_count
                weight_zero_params += zero_count
    
    total_sparsity = (total_zero_params / total_params) * 100 if total_params > 0 else 0
    weight_sparsity = (weight_zero_params / weight_params) * 100 if weight_params > 0 else 0
    
    print(f"Total Parameter Sparsity: {total_sparsity:.2f}% ({total_zero_params:,}/{total_params:,})")
    print(f"Weight-only Sparsity:     {weight_sparsity:.2f}% ({weight_zero_params:,}/{weight_params:,})")
    
    return total_sparsity, weight_sparsity, total_params, weight_params, total_zero_params, weight_zero_params

def calculate_theoretical_sizes(model, weight_bits=None, weight_sparsity=0, total_sparsity=0):
    """Calculate theoretical model sizes considering quantization and sparsity"""
    print(f"\n{'='*60}")
    print("THEORETICAL SIZE ANALYSIS")
    print(f"{'='*60}")
    
    # Current actual size (FP32)
    actual_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # Separate weight and non-weight parameters
    weight_size_fp32 = 0
    weight_params_count = 0
    non_weight_size = 0
    
    for name, param in model.named_parameters():
        if param.dim() > 1 and 'weight' in name:  # Weight parameters
            weight_size_fp32 += param.numel() * 4  # FP32
            weight_params_count += param.numel()
        else:  # Biases, BatchNorm, etc.
            non_weight_size += param.numel() * 4  # Keep as FP32
    
    print(f"Current Model Size (FP32): {actual_size_bytes / (1024*1024):.3f} MB")
    print(f"  - Weights:     {weight_size_fp32 / (1024*1024):.3f} MB")
    print(f"  - Non-weights: {non_weight_size / (1024*1024):.3f} MB")
    
    if weight_bits:
        # Theoretical quantized size (without sparsity)
        quantized_weight_size = weight_params_count * weight_bits / 8
        quantized_total_size = quantized_weight_size + non_weight_size
        
        print(f"\nQuantized Model Size (W{weight_bits}): {quantized_total_size / (1024*1024):.3f} MB")
        print(f"  - Quantized weights: {quantized_weight_size / (1024*1024):.3f} MB")
        print(f"  - Non-weights (FP32): {non_weight_size / (1024*1024):.3f} MB")
        
        # Theoretical size with sparsity (assuming sparse storage)
        # Method: Only store non-zero weights + indices
        if weight_sparsity > 0:
            non_zero_weights = weight_params_count * (1 - weight_sparsity/100)
            
            # Storage = non_zero_weights * (weight_bits + index_bits) / 8
            # Assume 32-bit indices for general case
            index_bits = 3 # 20 bits can index up to ~1 million weights
            sparse_weight_size = non_zero_weights * (weight_bits + index_bits) / 8
            sparse_total_size = sparse_weight_size + non_weight_size
            
            print(f"\nSparse + Quantized Size (W{weight_bits}, {weight_sparsity:.1f}% sparse):")
            print(f"  Total size: {sparse_total_size / (1024*1024):.3f} MB")
            print(f"  - Sparse weights: {sparse_weight_size / (1024*1024):.3f} MB")
            print(f"  - Non-weights: {non_weight_size / (1024*1024):.3f} MB")
            print(f"  Calculation: {non_zero_weights:.0f} non-zero weights × ({weight_bits} + {index_bits} offset bits) ÷ 8")
            
            compression_ratio = actual_size_bytes / sparse_total_size
            print(f"  Overall Compression Ratio: {compression_ratio:.2f}x")
    
    else:
        # Only sparsity, no quantization
        if weight_sparsity > 0:
            non_zero_weights = weight_params_count * (1 - weight_sparsity/100)
            # FP32 weights + 32-bit indices
            sparse_weight_size = non_zero_weights * (32 + 3) / 8  # 8 bytes per non-zero weight
            sparse_total_size = sparse_weight_size + non_weight_size
            
            print(f"\nSparse Model Size (FP32, {weight_sparsity:.1f}% sparse):")
            print(f"  Total size: {sparse_total_size / (1024*1024):.3f} MB")
            print(f"  - Sparse weights: {sparse_weight_size / (1024*1024):.3f} MB")
            print(f"  - Non-weights: {non_weight_size / (1024*1024):.3f} MB")
            print(f"  Calculation: {non_zero_weights:.0f} non-zero weights × (32 + 3 offset bits) ÷ 8")
            
            compression_ratio = actual_size_bytes / sparse_total_size
            print(f"  Compression Ratio: {compression_ratio:.2f}x")
    
    return weight_size_fp32, non_weight_size

def measure_inference_time(model, test_loader, device, num_batches=50):
    """Measure model inference time"""
    print(f"\n{'='*60}")
    print("INFERENCE TIMING ANALYSIS")
    print(f"{'='*60}")
    
    model.eval()
    times = []
    
    with torch.no_grad():
        # Warmup
        for i, (inputs, _) in enumerate(test_loader):
            if i >= 5:  # 5 warmup batches
                break
            inputs = inputs.to(device)
            _ = model(inputs)
        
        # Actual timing
        for i, (inputs, _) in enumerate(test_loader):
            if i >= num_batches:
                break
            
            inputs = inputs.to(device)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            
            start_time = time.time()
            outputs = model(inputs)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            
            end_time = time.time()
            times.append(end_time - start_time)
    
    times = np.array(times)
    batch_size = test_loader.batch_size
    
    print(f"Inference Timing (over {len(times)} batches, batch_size={batch_size}):")
    print(f"  Average time per batch: {times.mean()*1000:.2f} ms")
    print(f"  Average time per image: {times.mean()*1000/batch_size:.2f} ms")
    print(f"  Throughput: {batch_size/times.mean():.1f} images/second")
    print(f"  Min batch time: {times.min()*1000:.2f} ms")
    print(f"  Max batch time: {times.max()*1000:.2f} ms")
    print(f"  Std deviation: {times.std()*1000:.2f} ms")
    
    return times.mean(), times.mean()/batch_size

def test_model_accuracy(model, test_loader, device):
    """Test model accuracy"""
    print(f"\n{'='*60}")
    print("ACCURACY EVALUATION")
    print(f"{'='*60}")
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = 100. * correct / total
    print(f"Test Accuracy: {accuracy:.2f}% ({correct}/{total})")
    return accuracy

# ================== MAIN EXECUTION BLOCK ==================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Comprehensive analysis of MobileNetV2 V5 model")
    parser.add_argument('model_path', help="Path to model checkpoint (.pth)")
    parser.add_argument("--bits", type=int, default=None, choices=[8, 6, 4], 
                        help="Weight bit-width for quantized models")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model path: {args.model_path}")

    # --- Model Loading ---
    model = create_mobilenetv2_v5(device)
    
    if args.bits is not None:
        if not QUANTIZE_MODULE_AVAILABLE:
            raise ImportError("Cannot test quantized model because 'quantize.py' could not be imported.")
        
        print(f"\n--- Loading {args.bits}-bit QUANTIZED Model ---")
        swap_to_quant_modules(model, weight_bits=args.bits, act_bits=args.bits)
        ckpt = torch.load(args.model_path, map_location=device)
        model.load_state_dict(ckpt)
        freeze_all_quant(model)
    else:
        print(f"\n--- Loading FP32 Model ---")
        ckpt = torch.load(args.model_path, map_location=device)
        state_dict = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state_dict, strict=False)
    
    # --- Comprehensive Analysis ---
    
    # 1. Accuracy
    test_loader = prepare_test_loader()
    accuracy = test_model_accuracy(model, test_loader, device)
    
    # 2. File size
    file_size_mb = os.path.getsize(args.model_path) / (1024 * 1024)
    print(f"\nFile Size: {file_size_mb:.3f} MB")
    
    # 3. File contents breakdown
    weights_size, biases_size, bn_size, quant_size, other_size = analyze_file_contents(
        model, ckpt, args.model_path, args.bits)
    
    # 4 & 5. Sparsity analysis
    total_sparsity, weight_sparsity, total_params, weight_params, zero_total, zero_weights = calculate_sparsity_metrics(model)
    
    # 6 & 7. Theoretical sizes
    theoretical_weight_size, theoretical_nonweight_size = calculate_theoretical_sizes(
        model, args.bits, weight_sparsity, total_sparsity)
    
    # 8. Inference time
    avg_batch_time, avg_image_time = measure_inference_time(model, test_loader, device)
    if args.bits is not None:
        total_acts, activation_cr, act_bits_total, act_bits_compressed = measure_activation_compression(
        model, test_loader, act_bits=8, device=device)

    print(f"\nActivation Compression Ratio: {activation_cr:.2f}×")
    print(f"  Uncompressed bits: {act_bits_total:,}")
    print(f"  Compressed bits:   {act_bits_compressed:,}")

    
    print(f"\n{'='*60}")