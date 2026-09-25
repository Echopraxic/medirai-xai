# Uncertainty Net - MedirAI

This repository contains code for training and testing neural networks with uncertainty quantification capabilities.

## Setup

### Environment

Create the conda environment and install dependencies:

```bash
conda env create -f environment.yml
conda activate env
```

Then install any additional dependencies from `pyproject.toml` if needed:

```bash
pip install -e .
```
## Datasets 
You can download the classification training/test data from [ISIC Archive](https://gallery.isic-archive.com/#!/topWithHeader/onlyHeaderTop/gallery?filter=%5B%22image_type%7Cclinical%3A%20close-up%22%5D) select ImageType to be "clinical: close up". Place the dataset in datasets folder within the repository.

## Running Training (train.py)

The training script trains a neural network model based on a configuration file.

### Required Arguments

- `--config`: Path to the configuration file (relative to `configs/` directory or absolute path)

### Optional Arguments

- `--search_setup`: Path to search for the config file (default: current script directory)
- `--no_date`: Do not append date/time to the output directory name
- `--continue_run`: Continue an interrupted training run
- `--seed`: Set random seed for reproducibility (integer)
- `--multi_gpu`: Use multiple GPUs if available

### Examples

**Basic training with default config:**
```bash
python train.py --config medir.json
```

**Training with a specific seed for reproducibility:**
```bash
python train.py --config medir.json --seed 42
```

**Training using multiple GPUs:**
```bash
python train.py --config medir.json --multi_gpu
```

**Continue a previous training run:**
```bash
python train.py --config medir.json --continue_run
```

**Training without appending date to output directory:**
```bash
python train.py --config medir.json --no_date
```

**Training with custom config search path:**
```bash
python train.py --config medir.json --search_setup /path/to/configs
```

### Configuration Files

Configuration files are stored in the `configs/` directory in JSON format. Key configuration sections include:

- **Network Setup**: Architecture, layer types, input/output sizes
- **Training Dataset**: Dataset selection, batch size, data augmentation
- **Training Parameters**: Learning rate, optimizer, number of epochs
- **Bayesian Parameters**: Uncertainty quantification settings
- **Paths**: Dataset and output paths

Available config files:
- `medir.json` - Standard MedirAI classification model
- `medir_resnet18.json` - ResNet18 variant
- `medir_wide.json` - Wide ResNet model
- `medir_rand.json` - Random initialization variant
- `medir_seg.json` - Segmentation model
- And more variants in the `configs/` directory

## Running Testing (test.py)

The testing script evaluates a trained model on test data.

### Required Arguments

- `--run`: The model run directory name (should exist in `output/` directory)

### Optional Arguments

- `--dataset`: Override the dataset to test on
- `--corrupt`: Test with corrupted images (format: `corruption_type-intensity`). Supports CIFAR-10-C or ImageNet-C
- `--noise`: Add noise to the dataset for robustness evaluation
- `--save_pred`: Save tensor predictions for classification or segmentation tasks

### Examples

**Basic testing on a trained model:**
```bash
python test.py --run medir/classification/run_20251210-105436
```

**Test with saved predictions:**
```bash
python test.py --run medir/classification/run_20251210-105436 --save_pred
```

**Test on a specific dataset:**
```bash
python test.py --run medir/classification/run_20251210-105436 --dataset your_dataset
```

## Output Structure

- Training outputs are saved to `output/` directory with structure: `output/{model_type}/{run_date_time}/`
  - `config.json` - Configuration used for the run
  - `log.csv` - Training metrics logged per epoch
  - `test.csv` - Test metrics (will be generated when the test.py is run after training)

- Checkpoints and model files are stored in the run directory

## Project Structure

```
├── train.py                 # Training script
├── test.py                  # Testing script
├── test_seg.py             # Segmentation testing script
├── train_seg.py            # Segmentation training script
├── configs/                # Configuration files
├── models/                 # Model architectures and utilities
├── datasets/               # Dataset handling
├── utils.py                # Utility functions
└── output/                 # Training outputs and logs
```

## Key Model Types

- **Deterministic**: Standard neural network without uncertainty
- **Bayesian**: Full Bayesian neural network with weight uncertainty
- **Layer Bayesian**: Uncertainty in specific layers

Select the model type in the configuration file under `Network.Basic Setup.network_type`.

## Other 
Check sprint slides (slide number 27 for best approach for deferral to expert) - mixture of multiple poor models would result in best performance with a hard constraint on agreement across all models.



