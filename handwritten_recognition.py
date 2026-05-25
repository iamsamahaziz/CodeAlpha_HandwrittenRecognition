"""
CodeAlpha Internship - Task 3: Handwritten Character Recognition
================================================================
Objective: Identify handwritten digits using Machine Learning.
Approach: SVM, Random Forest, KNN and MLP on MNIST dataset.
Dataset: MNIST (via sklearn - 70,000 images of handwritten digits 0-9)
Author: Samah AZIZ
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("TASK 3: HANDWRITTEN CHARACTER RECOGNITION")
print("=" * 60)

# ============================================================
# 1. LOAD MNIST DATASET
# ============================================================
print("\n" + "=" * 60)
print("1. LOADING MNIST DATASET")
print("=" * 60)

print("Downloading MNIST from OpenML (this may take a moment)...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X, y = mnist.data, mnist.target.astype(int)

print(f"[OK] Total samples: {X.shape[0]}")
print(f"[OK] Image dimensions: 28x28 = {X.shape[1]} features")
print(f"[OK] Classes: {len(np.unique(y))} (digits 0-9)")

# Use a subset for faster training (10,000 train + 2,000 test)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=2000, random_state=42, stratify=y
)
X_train, _, y_train, _ = train_test_split(
    X_train_full, y_train_full, train_size=10000, random_state=42, stratify=y_train_full
)

print(f"[OK] Training samples: {X_train.shape[0]}")
print(f"[OK] Test samples: {X_test.shape[0]}")

# ============================================================
# 2. VISUALIZE SAMPLE IMAGES
# ============================================================
print("\n" + "=" * 60)
print("2. VISUALIZING SAMPLE IMAGES")
print("=" * 60)

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    img = X_train[i].reshape(28, 28)
    ax.imshow(img, cmap='gray')
    ax.set_title(f'Label: {y_train[i]}', fontsize=12, fontweight='bold')
    ax.axis('off')

plt.suptitle('Sample MNIST Handwritten Digits', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('sample_images.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Sample images saved to 'sample_images.png'")

# Class distribution
fig, ax = plt.subplots(figsize=(10, 5))
unique, counts = np.unique(y_train, return_counts=True)
colors = plt.cm.tab10(np.linspace(0, 1, 10))
ax.bar(unique.astype(str), counts, color=colors)
ax.set_xlabel('Digit', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Class Distribution in Training Set', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Class distribution saved to 'class_distribution.png'")

# ============================================================
# 3. DATA PREPROCESSING
# ============================================================
print("\n" + "=" * 60)
print("3. DATA PREPROCESSING")
print("=" * 60)

# Normalize pixel values to [0, 1]
X_train_norm = X_train / 255.0
X_test_norm = X_test / 255.0

# Standard scaling for SVM and KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_norm)
X_test_scaled = scaler.transform(X_test_norm)

print(f"[OK] Pixel range after normalization: [{X_train_norm.min():.1f}, {X_train_norm.max():.1f}]")
print(f"[OK] Feature shape: {X_train_scaled.shape}")

# ============================================================
# 4. MODEL TRAINING & EVALUATION
# ============================================================
print("\n" + "=" * 60)
print("4. MODEL TRAINING & EVALUATION")
print("=" * 60)

models = {
    'SVM (RBF)': SVC(kernel='rbf', gamma='scale', C=10, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
    'KNN (K=3)': KNeighborsClassifier(n_neighbors=3, n_jobs=-1),
    'MLP Neural Network': MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        max_iter=50,
        batch_size=256,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15
    ),
}

results = {}

for name, model in models.items():
    print(f"\n--- {name} ---")
    print(f"  Training...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    results[name] = {
        'Accuracy': acc, 'Precision': prec, 'Recall': rec,
        'F1-Score': f1, 'y_pred': y_pred
    }

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

# ============================================================
# 5. MODEL COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("5. MODEL COMPARISON")
print("=" * 60)

import pandas as pd

comparison_df = pd.DataFrame({
    name: {k: v for k, v in vals.items() if k != 'y_pred'}
    for name, vals in results.items()
}).T.round(4)

print(comparison_df.to_string())

best_model_name = comparison_df['Accuracy'].idxmax()
print(f"\n[BEST] Best Model: {best_model_name} "
      f"(Accuracy: {comparison_df.loc[best_model_name, 'Accuracy']:.4f})")

# ============================================================
# 6. VISUALIZATIONS
# ============================================================
print("\n" + "=" * 60)
print("6. GENERATING VISUALIZATIONS")
print("=" * 60)

# --- Model comparison bar chart ---
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(metrics))
width = 0.2
colors_models = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6']

fig, ax = plt.subplots(figsize=(12, 6))
for i, (name, vals) in enumerate(results.items()):
    values = [vals[m] for m in metrics]
    ax.bar(x + i * width, values, width, label=name, color=colors_models[i], alpha=0.85)

ax.set_xlabel('Metrics', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison - Handwritten Recognition', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(metrics)
ax.legend(fontsize=10)
ax.set_ylim(0.85, 1.01)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Model comparison saved to 'model_comparison.png'")

# --- Confusion Matrix for best model ---
y_pred_best = results[best_model_name]['y_pred']
cm = confusion_matrix(y_test, y_pred_best)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=range(10), yticklabels=range(10))
ax.set_xlabel('Predicted Digit', fontsize=12)
ax.set_ylabel('Actual Digit', fontsize=12)
ax.set_title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Confusion matrix saved to 'confusion_matrix.png'")

# --- Sample Predictions ---
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
indices = np.random.RandomState(42).choice(len(X_test), 10, replace=False)

for idx, ax in zip(indices, axes.flat):
    img = X_test[idx].reshape(28, 28)
    pred = y_pred_best[idx]
    true = y_test[idx]
    color = 'green' if pred == true else 'red'

    ax.imshow(img, cmap='gray')
    ax.set_title(f'Pred: {pred} | True: {true}', fontsize=11,
                 fontweight='bold', color=color)
    ax.axis('off')

plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('sample_predictions.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Sample predictions saved to 'sample_predictions.png'")

# --- Misclassified examples ---
misclassified = np.where(y_pred_best != y_test)[0]
print(f"\n[X] Total misclassified: {len(misclassified)} / {len(y_test)} "
      f"({len(misclassified)/len(y_test)*100:.2f}%)")

if len(misclassified) > 0:
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    for idx, ax in zip(misclassified[:10], axes.flat):
        img = X_test[idx].reshape(28, 28)
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Pred: {y_pred_best[idx]} | True: {y_test[idx]}',
                     fontsize=11, fontweight='bold', color='red')
        ax.axis('off')
    plt.suptitle('Misclassified Examples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('misclassified.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Misclassified examples saved to 'misclassified.png'")

# ============================================================
# 7. CLASSIFICATION REPORT (Best Model)
# ============================================================
print("\n" + "=" * 60)
print(f"7. DETAILED CLASSIFICATION REPORT - {best_model_name}")
print("=" * 60)
print(classification_report(y_test, y_pred_best, digits=4))

print("\n" + "=" * 60)
print("[OK] TASK 3 COMPLETED SUCCESSFULLY!")
print("=" * 60)
