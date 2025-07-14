from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
)
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np

# Load data
X = np.load(r"C:\Users\Pouya\Desktop\ICBME2025\PD SC\Data\lstm_selected_features.npy")
y = np.load(r"C:\Users\Pouya\Desktop\ICBME2025\PD SC\Data\lstm_labels.npy")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []
precisions = []
recalls = []
f1s = []
specificities = []
all_y_true = []
all_y_pred = []

# Feature selection: Select top 20 features (adjust based on your feature count)
k_features = min(20, X.shape[1])  # Use 20 or fewer if feature count is lower
selector = SelectKBest(score_func=f_classif, k=k_features)

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
    print(f"\n--- Fold {fold} ---")

    # Split and scale
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Apply feature selection
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    # Define classifiers for ensemble
    knn_clf = KNeighborsClassifier(n_neighbors=7, weights='distance', metric='euclidean')
    svm_clf = SVC(kernel='rbf', C=10, gamma=0.1, class_weight='balanced', random_state=42, probability=True)
    
    # Ensemble with adjusted weights (more SVM influence)
    ensemble_clf = VotingClassifier(estimators=[
        ('knn', knn_clf), ('svm', svm_clf)], voting='soft', weights=[1, 2])
    
    ensemble_clf.fit(X_train_scaled, y_train)
    y_pred = ensemble_clf.predict(X_test_scaled)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Specificity from confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    f1s.append(f1)
    specificities.append(specificity)
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

    print(f"Accuracy:   {acc:.4f}")
    print(f"Precision:  {prec:.4f}")
    print(f"Sensitivity: {rec:.4f}")
    print(f"F1-score:   {f1:.4f}")
    print(f"Specificity: {specificity:.4f}")

# Summary
print("\n=== Cross-validation Summary ===")
print(f"Mean Accuracy:   {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"Mean Precision:  {np.mean(precisions):.4f}")
print(f"Mean Sensitivity: {np.mean(recalls):.4f}")
print(f"Mean F1-score:   {np.mean(f1s):.4f}")
print(f"Mean Specificity: {np.mean(specificities):.4f}")

print("\n=== Full Classification Report ===")
print(classification_report(all_y_true, all_y_pred, zero_division=0))
