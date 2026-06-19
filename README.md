# Image Classification Using PyTorch

A deep learning project for image classification built with PyTorch, featuring a custom training loop with train/validation tracking and performance visualization.

---

## Overview

This project implements an image classification pipeline using a neural network trained and evaluated with PyTorch. It tracks loss and accuracy across epochs for both training and validation sets, enabling clear insight into model performance over time.

---

## Features

- Custom training loop with per-epoch logging
- Train/validation split with separate data loaders
- Loss and accuracy tracking across epochs
- Plotting of training curves for visual performance analysis
- Clean modular structure using PyTorch best practices

---

## Requirements

Install the dependencies using pip:

```bash
pip install torch torchvision matplotlib
```

**Python version:** 3.8+

---

## Project Structure

```
Image-Classification-Using-Pytorch/
│
├── model.py               # Model architecture definition
├── train.py               # Training loop and evaluation logic
├── dataset.py             # Dataset loading and preprocessing
├── utils.py               # Helper functions (plotting, metrics)
├── main.ipynb             # Jupyter Notebook (full pipeline)
└── README.md
```

---

## Usage

### 1. Clone the repository

```bash
git clone https://github.com/Muhammad-Musharraf/Image-Classification-Using-Pytorch.git
cd Image-Classification-Using-Pytorch
```

### 2. Prepare your dataset

Organize your dataset into `train/` and `val/` directories, each containing one subfolder per class:

```
data/
├── train/
│   ├── class_1/
│   └── class_2/
└── val/
    ├── class_1/
    └── class_2/
```

### 3. Run training

Open and run `main.ipynb` in Jupyter, or execute the training script directly:

```bash
python train.py
```

---

## Training Details

The training loop runs for a configurable number of epochs. At each epoch it:

- Performs a forward pass and backpropagation on the training set
- Evaluates the model on the validation set (no gradient computation)
- Logs train/validation loss and accuracy

Example output per epoch:

```
Epoch 1/20, Train Loss: 0.0342 Train Accuracy 72.5400
              Validation Loss: 0.0289 Validation Accuracy: 78.2300
=========================
```

---

## Metrics Tracked

| Metric              | Description                              |
|---------------------|------------------------------------------|
| Train Loss          | Average loss over training batches       |
| Validation Loss     | Average loss over validation batches     |
| Train Accuracy      | % of correct predictions on train set    |
| Validation Accuracy | % of correct predictions on val set      |

---

## Results

Training and validation curves are stored in lists for plotting:

```python
total_loss_train_plot
total_loss_validation_plot
total_acc_train_plot
total_acc_validation_plot
```

Use `matplotlib` to visualize:

```python
import matplotlib.pyplot as plt

plt.plot(total_acc_train_plot, label='Train Accuracy')
plt.plot(total_acc_validation_plot, label='Validation Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')
plt.show()
```

---

## License

This project is open-source and available under the [MIT License](LICENSE).

---

## Author

**Muhammad Musharraf**  
[GitHub Profile](https://github.com/Muhammad-Musharraf)
