# CodeAlpha - Handwritten Character Recognition ✍️

## Overview
This project identifies **handwritten digits (0-9)** using a Convolutional Neural Network (CNN) trained on the MNIST dataset. Built as part of the **CodeAlpha Machine Learning Internship**.

## Approach
- **Dataset**: MNIST (60,000 training + 10,000 test images)
- **Model**: Deep CNN with BatchNormalization and Dropout
- **Architecture**: 3 Convolutional blocks → Fully Connected → Softmax

## Model Architecture
```
Conv2D(32) → BN → Conv2D(32) → BN → MaxPool → Dropout(0.25)
Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Dropout(0.25)
Conv2D(128) → BN → MaxPool → Dropout(0.25)
Flatten → Dense(256) → BN → Dropout(0.5)
Dense(128) → Dropout(0.3) → Dense(10, softmax)
```

## Results
- **Test Accuracy**: ~99.3%+
- Confusion matrix and training curves are generated automatically.

## How to Run
```bash
pip install -r requirements.txt
python handwritten_recognition.py
```

## Generated Outputs
- `sample_images.png` — Sample MNIST images
- `training_history.png` — Accuracy & Loss curves
- `confusion_matrix.png` — 10x10 confusion matrix
- `sample_predictions.png` — Predictions vs ground truth
- `handwritten_cnn_model.h5` — Saved trained model

## Author
**Samah AZIZ** — CodeAlpha ML Internship
