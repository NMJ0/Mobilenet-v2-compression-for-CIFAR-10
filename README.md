


# MobileNetV2 Compression for CIFAR-10

This repository implements a comprehensive neural network compression pipeline for MobileNetV2 on CIFAR-10, including iterative Fisher-based pruning,  quantization, and evaluation tools.

## Features

- **Iterative Fisher-based Pruning**: Gradual weight pruning based on Fisher information
- **Fake Quantization**: Weight and activation quantization with configurable bit-widths
- **Comprehensive Evaluation**: Accuracy, model size, sparsity analysis, and inference timing


## Requirements

- Python 3.7+
- PyTorch 1.8+
- torchvision
- numpy
- CUDA (optional, for GPU acceleration)

## Installation

```bash
git clone https://github.com/NMJ0/Mobilenet-v2-compression-for-CIFAR-10.git
cd Mobilenet-v2-compression-for-CIFAR-10
pip install -r requirements.txt
```

## Usage

### 1. Training the baseline
```bash
python train_base.py
```
To download the CIFAR-10 dataset and train the baseline model  `base_model.pth` .


### 2. Model Pruning

To prune the model using iterative Fisher-based pruning:

**Prerequisites**: Ensure that `base_model.pth` is present after training the baseline model.

```bash
python iterative_pruning.py
```

This will:
- Load the baseline model from `base_model.pth`
- Apply iterative Fisher-based pruning
- Fine-tune after each pruning stage
- Save the pruned model

### 3. Model Quantization

To apply  quantization to a trained model:

```bash
python quantize.py model_path --weight-bits 8 --act-bits 8
```

**Parameters:**
- `model_path`: Path to the model checkpoint (.pth file)
- `--weight-bits`: Number of bits for weight quantization (default: 8)
- `--act-bits`: Number of bits for activation quantization (default: 8)

**Example:**
```bash
python quantize.py pruned_model.pth --weight-bits 4 --act-bits 8
```

### 4. Model Evaluation

#### For Non-Quantized Models

```bash
python test.py model_path
```

**Example:**
```bash
python test.py base_model.pth
python test.py pruned_model.pth
```

#### For Quantized Models

```bash
python test.py model_path --bits 8
```

**Parameters:**
- `model_path`: Path to the quantized model checkpoint
- `--bits`: Number of bits used for quantization of weights (must match the quantization settings)

**Example:**
```bash
python test.py quantized_model.pth --bits 8
python test.py quantized_model.pth --bits 4
```

## Evaluation Metrics

The evaluation script provides comprehensive analysis including:

- **Accuracy**: Test accuracy on CIFAR-10
- **Model Size**: File size and theoretical compressed size
- **Sparsity Analysis**: Overall parameter sparsity and weight-only sparsity
- **Content Breakdown**: Storage distribution (weights, biases, BatchNorm, etc.)
- **Compression Ratios**: Model, weight, and activation compression ratios
- **Inference Time**: Average inference time and throughput












