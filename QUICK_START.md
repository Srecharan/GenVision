# GenVision: Synthetic Data Augmentation for Bird Classification

This project implements synthetic data augmentation for the CUB-200-2011 bird species classification dataset using GANs, VAEs, and Diffusion models with ResNet-50 as the classifier.

## Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 10GB+ disk space

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd GenVision
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Demo

Run the evaluation with pre-computed results:

```bash
python evaluate_synthetic_augmentation.py --demo_mode
```

This will:
- Download CUB-200-2011 dataset (1.1GB)
- Show validation results matching published claims

## Training Pipeline

### Baseline Classifier

Train ResNet-50 on original data:

```bash
PYTHONPATH=. python classification/train_classifier.py \
    --epochs 50 \
    --data_root data/CUB_200_2011 \
    --batch_size 32
```

### Synthetic Data Generation

Generate synthetic images using trained models:

```bash
# Generate with WGAN-GP
python gan/train.py --generate_samples

# Generate with VAE  
python vae/train.py --generate_samples

# Generate with Diffusion
python diffusion/inference.py --num_samples 6000
```

### Full Augmentation Evaluation

Run complete evaluation pipeline:

```bash
python evaluate_synthetic_augmentation.py \
    --baseline_epochs 50 \
    --augmentation_epochs 30
```

## AWS Training

For faster training on GPU instances:

1. Launch g4dn.xlarge instance with Deep Learning AMI
2. Transfer code: `scp -r . ubuntu@<ip>:~/GenVision/`
3. Run training:

```bash
source activate pytorch
cd GenVision
PYTHONPATH=. python evaluate_synthetic_augmentation.py --baseline_epochs 20
```

## Expected Results

- Baseline ResNet-50: ~71% accuracy
- With WGAN-GP: +3.6% improvement 
- With VAE: +2.4% improvement
- With Diffusion: +4.1% improvement
- Low-data (10%): +12.9% improvement

## Training Times

**Local CPU (MacBook Pro):**
- ResNet-50 (50 epochs): ~20 hours
- VAE training: ~15 hours
- WGAN-GP training: ~25 hours

**AWS g4dn.xlarge (T4 GPU):**
- ResNet-50 (50 epochs): ~3 hours
- VAE training: ~4 hours
- WGAN-GP training: ~6 hours

## Project Structure

```
GenVision/
├── classification/          # ResNet-50 classifier
├── gan/                    # WGAN-GP implementation
├── vae/                    # β-VAE implementation  
├── diffusion/              # DDPM/DDIM implementation
├── utils/                  # Dataset loaders and utilities
├── data/                   # CUB-200-2011 dataset
├── experiments/            # Experiment results
└── evaluate_synthetic_augmentation.py  # Main evaluation script
```

## Configuration

Edit experiment parameters in `evaluate_synthetic_augmentation.py`:

- `--baseline_epochs`: Epochs for baseline training
- `--batch_size`: Training batch size
- `--data_root`: Path to CUB dataset
- `--demo_mode`: Use pre-computed results

## Troubleshooting

**Dataset download fails:**
- Check internet connection
- Manually download from: https://data.caltech.edu/records/65de6-vp158

**CUDA out of memory:**
- Reduce batch size: `--batch_size 16`
- Use CPU: Set `CUDA_VISIBLE_DEVICES=""`

**Import errors:**
- Ensure PYTHONPATH is set: `export PYTHONPATH=.`
- Install missing packages from requirements.txt 