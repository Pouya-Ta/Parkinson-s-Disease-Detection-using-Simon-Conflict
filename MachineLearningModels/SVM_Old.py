# If you are using Colab as an environment, you can uncomment the code below
"""
from google.colab import drive
drive.mount('/content/drive')
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)

# Load Data
X = np.load("lstm_selected_features.npy")  # shape: (N, 32 or 64) | Optimize this path and adjust it based on your own path for selected features after the hybrid EEGNet-LSTM models
y = np.load("lstm_labels.npy")             # shape: (N,) | Optimize this path and adjust it based on your own path for labels (as another input for the model)
groups = np.load("lstm_groups.npy")        # shape: (N,)

# === Setup CV ===
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
accs, precs, recs, f1s = [], [], [], []
all_y_true, all_y_pred = [], []

# === SVM Loop ===
for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), 1):
    print(f"\n--- Fold {fold} ---")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # SVM (use previous best: C=10, gamma=0.1)
    clf = SVC(kernel='rbf', C=10, gamma=0.1, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Metrics
    accs.append(accuracy_score(y_test, y_pred))
    precs.append(precision_score(y_test, y_pred))
    recs.append(recall_score(y_test, y_pred))
    f1s.append(f1_score(y_test, y_pred))
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

    print(f"Accuracy:  {accs[-1]:.4f}")
    print(f"Precision: {precs[-1]:.4f}")
    print(f"Recall:    {recs[-1]:.4f}")
    print(f"F1-score:  {f1s[-1]:.4f}")

# Summary
print("\n=== Cross-validation Summary ===")
print(f"Mean Accuracy:  {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"Mean Precision: {np.mean(precs):.4f}")
print(f"Mean Recall:    {np.mean(recs):.4f}")
print(f"Mean F1-score:  {np.mean(f1s):.4f}")

# Final Report
print("\n=== Full Classification Report ===")
print(classification_report(all_y_true, all_y_pred, digits=4))
