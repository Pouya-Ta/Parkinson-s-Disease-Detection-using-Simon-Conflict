import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
X = np.load(r"lstm_selected_features.npy") # Optimize this path and adjust it based on your own path for selected features after the hybrid EEGNet-LSTM models
y = np.load(r"lstm_labels.npy") # Optimize this path and adjust it based on your own path for labels (as another input for the model)

# Global Feature Selection + Scaling
k_features = min(30, X.shape[1])
selector = SelectKBest(score_func=f_classif, k=k_features)
X_selected = selector.fit_transform(X, y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Define Classifiers
svm = SVC(kernel='rbf', C=10, gamma=0.1, class_weight='balanced', probability=True, random_state=42)
nb = GaussianNB(var_smoothing=1e-9)

ensemble = VotingClassifier(
    estimators=[('svm', svm), ('nb', nb)],
    voting='soft',
    weights=[2, 1]
)

# CV Setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'spec': [], 'true': [], 'pred': []}

# Cross-validation Loop
for fold, (train_idx, test_idx) in enumerate(cv.split(X_scaled, y), 1):
    print(f"\n=== Ensemble Fold {fold} ===")

    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    results['acc'].append(acc)
    results['prec'].append(prec)
    results['rec'].append(rec)
    results['f1'].append(f1)
    results['spec'].append(spec)
    results['true'].extend(y_test)
    results['pred'].extend(y_pred)

    print(f"Accuracy:   {acc:.4f}")
    print(f"Precision:  {prec:.4f}")
    print(f"Sensitivity: {rec:.4f}")
    print(f"F1-score:   {f1:.4f}")
    print(f"Specificity: {spec:.4f}")

# Final Report
print("\n=== SVM + Naive Bayes Ensemble Summary ===")
print(f"Mean Accuracy:   {np.mean(results['acc']):.4f} ± {np.std(results['acc']):.4f}")
print(f"Mean Precision:  {np.mean(results['prec']):.4f}")
print(f"Mean Sensitivity: {np.mean(results['rec']):.4f}")
print(f"Mean F1-score:   {np.mean(results['f1']):.4f}")
print(f"Mean Specificity: {np.mean(results['spec']):.4f}")

print("\n=== Full Classification Report ===")
print(classification_report(results['true'], results['pred'], zero_division=0))

# Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(results['true'], results['pred']), annot=True, fmt='d', cmap='Blues')
plt.title("SVM + NB Ensemble Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
