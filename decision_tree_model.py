# =============================================================
# Part B: Decision Tree Model - Group Practical Assessment
# Course: Collaborative Software Development
# Member: Hussein Hajir (Huska22) - 670397
# Dataset: Iris Dataset
# Model: Decision Tree Classifier
# =============================================================

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    classification_report, confusion_matrix
)
import time

# ---------------------------
# 1. Load Dataset
# ---------------------------
print("=" * 55)
print("  Decision Tree Classifier - Iris Dataset")
print("=" * 55)

iris = load_iris()
X = iris.data         # Features: sepal/petal length & width
y = iris.target       # Labels: 0=setosa, 1=versicolor, 2=virginica

print(f"\n[INFO] Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
print(f"[INFO] Classes: {list(iris.target_names)}")

# ---------------------------
# 2. Train/Test Split (80/20)
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n[INFO] Training samples : {len(X_train)}")
print(f"[INFO] Testing  samples : {len(X_test)}")

# ---------------------------
# 3. Feature Scaling
# ---------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("\n[INFO] Features scaled using StandardScaler")

# ---------------------------
# 4. Build & Train Decision Tree
# ---------------------------
print("\n[INFO] Training Decision Tree Classifier...")
print("       Hyperparameters: criterion='gini', max_depth=4, random_state=42")

start_time = time.time()

model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=4,
    random_state=42
)
model.fit(X_train_scaled, y_train)

train_time = round(time.time() - start_time, 5)
print(f"[INFO] Training completed in {train_time} seconds")

# ---------------------------
# 5. Evaluate Model
# ---------------------------
y_pred = model.predict(X_test_scaled)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall    = recall_score(y_test, y_pred, average='weighted')
f1        = f1_score(y_test, y_pred, average='weighted')

print("\n" + "=" * 55)
print("  PERFORMANCE METRICS - DECISION TREE")
print("=" * 55)
print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1-Score  : {f1:.4f}")
print(f"  Train Time: {train_time} seconds")

print("\n--- Detailed Classification Report ---")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print("--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("\n[INFO] Rows = Actual class | Columns = Predicted class")
print("=" * 55)
print("  Evaluation complete.")
print("=" * 55)
