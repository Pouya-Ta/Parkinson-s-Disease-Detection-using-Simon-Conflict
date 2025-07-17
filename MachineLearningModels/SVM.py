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
y = np.load("/content/drive/MyDrive/ParkinsonClassifier/lstm_labels.npy")             # shape: (N,)
groups = np.load("/content/drive/MyDrive/ParkinsonClassifier/lstm_groups.npy")        # shape: (N,)
