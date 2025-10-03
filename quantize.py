# quant_and_evaluate_and_save.py

import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import models
# ---------- core helpers (uniform quant) ----------
def qparams_from_minmax(xmin, xmax, n_bits=8, unsigned=False, eps=1e-12):
    if unsigned:
        qmin, qmax = 0, (1 << n_bits) - 1
        xmin = torch.zeros_like(xmin)
        scale = (xmax - xmin).clamp_min(eps) / float(qmax - qmin)
        zp = torch.round(-xmin / scale).clamp(qmin, qmax)
    else:
        qmax = (1 << (n_bits - 1)) - 1
        qmin = -qmax
        max_abs = torch.max(xmin.abs(), xmax.abs()).clamp_min(eps)
        scale = max_abs / float(qmax)
        zp = torch.zeros_like(scale)
    return scale, zp, int(qmin), int(qmax)

def quantize(x, scale, zp, qmin, qmax):
    return torch.clamp(torch.round(x / scale + zp), qmin, qmax)

def dequantize(q, scale, zp):
    return (q - zp) * scale

# ---------- activation fake-quant ----------
class ActFakeQuant(nn.Module):
    def __init__(self, n_bits=8, unsigned=True):
        super().__init__()
        self.n_bits = n_bits
        self.unsigned = unsigned
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zp", torch.tensor(0.0))
        self.frozen = False
        self.qmin = None
        self.qmax = None

    @torch.no_grad()
    def observe(self, x):
        self.min_val = torch.minimum(self.min_val, x.min())
        self.max_val = torch.maximum(self.max_val, x.max())

    @torch.no_grad()
    def freeze(self):
        scale, zp, qmin, qmax = qparams_from_minmax(
            self.min_val, self.max_val, n_bits=self.n_bits, unsigned=self.unsigned
        )
        self.scale.copy_(scale.to(self.scale.device))
        self.zp.copy_(zp.to(self.zp.device))
        self.qmin, self.qmax = qmin, qmax
        self.frozen = True

    def forward(self, x):
        if not self.frozen:
            self.observe(x)
            return x
        q = quantize(x, self.scale, self.zp, self.qmin, self.qmax)
        return dequantize(q, self.scale, self.zp)

# ---------- weight fake-quant wrappers ----------
class QuantConv2d(nn.Conv2d):
    def __init__(self, *args, weight_bits=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_bits = weight_bits
        self.register_buffer("w_scale", torch.tensor(1.0))
        self.register_buffer("w_zp", torch.tensor(0.0))
        self.frozen = False
        self.qmin = None
        self.qmax = None

    @torch.no_grad()
    def freeze(self):
        w = self.weight.detach().cpu()
        w_min, w_max = w.min(), w.max()
        scale, zp, qmin, qmax = qparams_from_minmax(
            w_min, w_max, n_bits=self.weight_bits, unsigned=False
        )
        self.w_scale.copy_(scale.to(self.w_scale.device))
        self.w_zp.copy_(zp.to(self.w_zp.device))
        self.qmin, self.qmax = qmin, qmax
        self.frozen = True

    def forward(self, x):
        if not self.frozen:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        q = quantize(self.weight, self.w_scale, self.w_zp, self.qmin, self.qmax)
        w_dq = dequantize(q, self.w_scale, self.w_zp)
        return F.conv2d(x, w_dq, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class QuantLinear(nn.Linear):
    def __init__(self, *args, weight_bits=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_bits = weight_bits
        self.register_buffer("w_scale", torch.tensor(1.0))
        self.register_buffer("w_zp", torch.tensor(0.0))
        self.frozen = False
        self.qmin = None
        self.qmax = None

    @torch.no_grad()
    def freeze(self):
        w = self.weight.detach().cpu()
        w_min, w_max = w.min(), w.max()
        scale, zp, qmin, qmax = qparams_from_minmax(
            w_min, w_max, n_bits=self.weight_bits, unsigned=False
        )
        self.w_scale.copy_(scale.to(self.w_scale.device))
        self.w_zp.copy_(zp.to(self.w_zp.device))
        self.qmin, self.qmax = qmin, qmax
        self.frozen = True

    def forward(self, x):
        if not self.frozen:
            return F.linear(x, self.weight, self.bias)
        q = quantize(self.weight, self.w_scale, self.w_zp, self.qmin, self.qmax)
        w_dq = dequantize(q, self.w_scale, self.w_zp)
        return F.linear(x, w_dq, self.bias)

# ---------- module swap and freeze ----------
def swap_to_quant_modules(model, weight_bits=8, act_bits=8):
    """
    (Robust Version)
    Recursively swaps modules with their quantized versions in-place.
    This version correctly handles nested modules.
    """
    for name, module in list(model.named_children()):
        # First, recurse into the child module
        swap_to_quant_modules(module, weight_bits, act_bits)

        # After recursion, check if the child module itself needs to be replaced
        if isinstance(module, nn.Conv2d):
            q = QuantConv2d(
                module.in_channels, module.out_channels, module.kernel_size,
                stride=module.stride, padding=module.padding,
                dilation=module.dilation, groups=module.groups,
                bias=(module.bias is not None), weight_bits=weight_bits
            ).to(module.weight.device)
            q.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                q.bias.data.copy_(module.bias.data)
            setattr(model, name, q)

        elif isinstance(module, nn.Linear):
            q = QuantLinear(
                module.in_features, module.out_features,
                bias=(module.bias is not None), weight_bits=weight_bits
            ).to(module.weight.device)
            q.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                q.bias.data.copy_(module.bias.data)
            setattr(model, name, q)
        
        # This now correctly handles all ReLU variants and SiLU
        elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.SiLU)):
            # Determine device from module parameters if possible, otherwise fallback
            try:
                device = next(module.parameters()).device
            except StopIteration:
                # Fallback for layers with no parameters like ReLU
                device = next(model.parameters()).device
                
            seq = nn.Sequential(OrderedDict([
                ("activation", module),
                ("aq", ActFakeQuant(n_bits=act_bits, unsigned=True).to(device))
            ]))
            setattr(model, name, seq)


def freeze_all_quant(model):
    for mod in model.modules():
        if isinstance(mod, (QuantConv2d, QuantLinear)):
            mod.freeze()
        if isinstance(mod, ActFakeQuant):
            mod.freeze()

# ---------- model utils and evaluation ----------
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

def load_model(path, device):
    model = create_mobilenetv2_v5().to(device)
    ckpt = torch.load(path, map_location=device)
    # Handle checkpoints saved with and without a wrapping dict
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def prepare_test_loader(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
    ])
    ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += (preds==y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total

def model_size_bytes_fp32(model):
    return sum(p.numel()*4 for p in model.parameters())

def model_size_bytes_quant(model, weight_bits):
    total = 0
    for name,p in model.named_parameters():
        if "weight" in name:
            # Assumes per-tensor quantization for weights
            total += p.numel() * weight_bits / 8
        elif "bias" in name:
            # Biases are typically kept in FP32
            total += p.numel() * 4
    return total

# ---------- main ----------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quantize and evaluate a pruned MobileNetV2 V5")
    parser.add_argument("model_path", help="Path to pruned checkpoint (.pth)")
    # --- ADDED ARGUMENTS ---
    parser.add_argument("--weight-bits", type=int, default=8, help="Number of bits for weight quantization (default: 8)")
    parser.add_argument("--act-bits", type=int, default=8, help="Number of bits for activation quantization (default: 8)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = prepare_test_loader()

    # Load pruned FP32 model
    base_model = load_model(args.model_path, device)
    base_acc = evaluate(base_model, test_loader, device)
    base_size = model_size_bytes_fp32(base_model) / (1024*1024)
    print(f"Base FP32 accuracy: {base_acc:.2f}%")
    print(f"Base FP32 size    : {base_size:.2f} MB")

    # --- GENERALIZED QUANTIZATION PROCESS ---
    print(f"\nStarting quantization with {args.weight_bits}-bit weights and {args.act_bits}-bit activations...")
    
    q_model = copy.deepcopy(base_model)
    swap_to_quant_modules(q_model, weight_bits=args.weight_bits, act_bits=args.act_bits)
    q_model = q_model.to(device)

    # Calibration pass
    print("Calibrating quantization ranges...")
    with torch.no_grad():
        for x, _ in test_loader:
            _ = q_model(x.to(device))
    
    # Freeze quantization parameters (scale and zero-point)
    print("Freezing quantization parameters...")
    freeze_all_quant(q_model)

    # Evaluate the quantized model
    q_acc = evaluate(q_model, test_loader, device)
    q_size = model_size_bytes_quant(q_model, weight_bits=args.weight_bits) / (1024*1024)
    
    print(f"\nQuantized (W{args.weight_bits}A{args.act_bits}) accuracy: {q_acc:.2f}%")
    print(f"Quantized (W{args.weight_bits}A{args.act_bits}) size    : {q_size:.2f} MB")

    # Save the quantized model's state_dict
    save_path = f"quantized_{args.weight_bits}a{args.act_bits}_quant.pth"
    torch.save(q_model.state_dict(), save_path)