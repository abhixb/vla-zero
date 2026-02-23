# VLA-Zero

This repository contains a PyTorch implementation of a VLA-0 style model for robot action generation.

## Model

The model uses the `Qwen/Qwen2-VL-2B-Instruct` model from the Hugging Face Hub. Actions are discretized and predicted as text tokens.

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd vla-zero
    ```
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training

To train the model, run:

```bash
python train.py
```

### Inference

You can test the model's action generation capabilities by running:

```bash
python model.py
```
