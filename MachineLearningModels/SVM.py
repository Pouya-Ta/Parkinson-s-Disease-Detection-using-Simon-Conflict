# If you are using Colab as an environment, you can uncomment the code below
"""
from google.colab import drive
drive.mount('/content/drive')
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Load Data
X = np.load("lstm_selected_features.npy")  # shape: (N, 32 or 64) | Optimize this path and adjust it based on your own path for selected features after the hybrid EEGNet-LSTM models
y = np.load("lstm_labels.npy")             # shape: (N,) | Optimize this path and adjust it based on your own path for labels (as another input for the model)
groups = np.load("lstm_groups.npy")        # shape: (N,)

# Define pipeline: scaling + SVM 
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", class_weight="balanced"))
])

# Define hyperparameter grid
param_grid = {
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": [0.001, 0.01, 0.1, 1]
}

# Use StratifiedGroupKFold
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

# Grid Search
grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1_weighted",
    cv=cv.split(X, y, groups),  # <<< Proper subject-level CV
    verbose=2,
    n_jobs=-1
)
