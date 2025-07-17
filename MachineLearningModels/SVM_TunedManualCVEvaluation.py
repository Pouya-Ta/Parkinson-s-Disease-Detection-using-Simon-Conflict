# If you are using Colab as an environment, you can uncomment the code below
"""
from google.colab import drive
drive.mount('/content/drive')
"""

from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
)
import numpy as np

# Load Data
X = np.load("lstm_selected_features.npy")  # shape: (N, 32 or 64) | Optimize this path and adjust it based on your own path for selected features after the hybrid EEGNet-LSTM models
y = np.load("lstm_labels.npy")             # shape: (N,) | Optimize this path and adjust it based on your own path for labels (as another input for the model)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []
precisions = []
recalls = []
f1s = []
all_y_true = []
all_y_pred = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
    print(f"\n--- Fold {fold} ---")

    # Split and scale
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Tuned SVM
    clf = SVC(kernel='rbf', C=100, gamma=0.01, class_weight='balanced', random_state=42)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    f1s.append(f1)
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

# Summary
print("\n=== Cross-validation Summary ===")
print(f"Mean Accuracy:     {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"Mean Precision:    {np.mean(precisions):.4f}")
print(f"Mean Recall:       {np.mean(recalls):.4f}")
print(f"Mean F1-score:     {np.mean(f1s):.4f}")

# === Sensitivity Calculation ===
# Confusion matrix: rows = true labels, columns = predicted labels
cm = confusion_matrix(all_y_true, all_y_pred)
TN, FP, FN, TP = cm.ravel()
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)

print(f"\n=== Sensitivity (TP / (TP + FN)) ===")
print(f"Sensitivity:        {sensitivity:.4f}")

print(f"\n=== Specificity (TN / (TN + FP)) ===")
print(f"SPecificity:        {specificity:.4f}")

# Classification report
print("\n=== Full Classification Report ===")
print(classification_report(all_y_true, all_y_pred))
