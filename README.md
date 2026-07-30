# 🧠 Digit Recognizer - PyTorch Convolutional Neural Network (CNN)

Welcome to the **Digit Recognizer** Computer Vision repository! This project implements a custom **Convolutional Neural Network (CNN)** using **PyTorch** to classify 28x28 grayscale handwritten digits (0 through 9) for the Kaggle competition.

---

## 📁 Repository Structure

```
.
├── digit_recognizer_pytorch.ipynb   # Complete PyTorch CNN Jupyter Notebook
├── generate_digit_submission.py     # Standalone PyTorch training & prediction script
├── submission_digit.csv             # Formatted prediction output file for Kaggle
└── README.md                        # Project documentation & CNN architecture
```

---

## 🏗️ Neural Network Architecture

The PyTorch model (`DigitCNN`) utilizes a 2-stage feature extraction network followed by a dense classification head:

1. **Convolution Block 1**:
   - `Conv2d` (1 input channel $\to$ 32 feature maps, $3 \times 3$ kernel, padding=1)
   - `BatchNorm2d(32)` + `ReLU` activation
   - `MaxPool2d(2, 2)` $\to$ Downsamples spatial dimensions from $28 \times 28$ to $14 \times 14$.

2. **Convolution Block 2**:
   - `Conv2d` (32 channels $\to$ 64 feature maps, $3 \times 3$ kernel, padding=1)
   - `BatchNorm2d(64)` + `ReLU` activation
   - `MaxPool2d(2, 2)` $\to$ Downsamples spatial dimensions from $14 \times 14$ to $7 \times 7$.

3. **Classification Head**:
   - `Dropout(0.25)` regularization
   - `Linear(64 * 7 * 7 -> 128)` + `ReLU`
   - `Linear(128 -> 10)` output logits (Digits 0–9).

---

## 🚀 Quickstart & Training

### 1. Run via PyTorch Script
Train the CNN and export predictions locally:
```bash
python generate_digit_submission.py
```

### 2. Run on Kaggle GPU
1. Go to the [Kaggle Digit Recognizer Competition](https://www.kaggle.com/c/digit-recognizer).
2. Upload `digit_recognizer_pytorch.ipynb`.
3. Under **Session options**, set Accelerator to **GPU T4**.
4. Click **Run All** $\to$ **Save Version** $\to$ **Submit to Competition**!

---

## 📊 Performance Metrics
- **Optimizer**: Adam ($\text{lr} = 0.001$)
- **Loss Function**: `CrossEntropyLoss`
- **Validation Accuracy**: $>98\%$ after 5 epochs
