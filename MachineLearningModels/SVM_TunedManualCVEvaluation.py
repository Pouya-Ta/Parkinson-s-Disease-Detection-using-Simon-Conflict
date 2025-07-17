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

# Define pipeline with scaling and SVM
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", class_weight="balanced"))
])

# Define hyperparameter grid
param_grid = {
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": [0.001, 0.01, 0.1, 1]
}

# Use Stratified 5-Fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid search
grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1_weighted",
    cv=cv,
    verbose=2,
    n_jobs=-1
)

grid.fit(X, y)

# Print best parameters and result
print("\n=== Best Parameters ===")
print(grid.best_params_)

print("\n=== Best Weighted F1-Score ===")
print(grid.best_score_)

# Predict on full data
y_pred = grid.predict(X)

print("\n=== Classification Report (Full Data) ===")
print(classification_report(y, y_pred))

# confusion matrix
ConfusionMatrixDisplay.from_predictions(y, y_pred, display_labels=["Healthy", "PD"])
