# =============================================================
# Part B: Group Model Performance Comparison Suite
# Course: Collaborative Software Development
# Repository: Kare100/ml-version-control-group6
# Group Members:
#   1. Valerie W. (Kare100)           - Neural Network (MLP)
#   2. Hussein Hajir Aden (Huska22)    - Decision Tree
#   3. RichHommie241                   - Random Forest
# =============================================================

import numpy as np
import pandas as pd
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

# 1. Load Dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Train/Test Split (80% train, 20% test, fixed random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Define the models
models = {
    "Neural Network (MLP)": (
        MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42),
        "Valerie W. (Kare100)",
        "High (Iterative backprop)"
    ),
    "Decision Tree": (
        DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42),
        "Hussein Hajir (Huska22)",
        "Very Low (Single tree)"
    ),
    "Random Forest": (
        RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "RichHommie241",
        "Moderate (100 ensemble trees)"
    )
}

results = []

print("=" * 85)
print("  PART B: GROUP 6 MACHINE LEARNING MODEL PERFORMANCE COMPARISON")
print("  Dataset: Iris (150 samples, 4 features, 3 classes)")
print("  Split: 80% Train (120 samples) / 20% Test (30 samples)")
print("=" * 85 + "\n")

for name, (model, member, resource_note) in models.items():
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time
    
    y_pred = model.predict(X_test_scaled)
    
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec  = recall_score(y_test, y_pred, average='weighted')
    f1   = f1_score(y_test, y_pred, average='weighted')
    
    results.append({
        "Model": name,
        "Contributor": member,
        "Accuracy": f"{acc*100:.2f}%",
        "Precision": f"{prec:.4f}",
        "Recall": f"{rec:.4f}",
        "F1-Score": f"{f1:.4f}",
        "Train Time (s)": f"{train_time:.5f}",
        "Resource Overhead": resource_note
    })

df_res = pd.DataFrame(results)

print(f"{'Model':<22} | {'Contributor':<24} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<8} | {'F1-Score':<8} | {'Train Time (s)':<15}")
print("-" * 115)
for _, r in df_res.iterrows():
    print(f"{r['Model']:<22} | {r['Contributor']:<24} | {r['Accuracy']:<10} | {r['Precision']:<10} | {r['Recall']:<8} | {r['F1-Score']:<8} | {r['Train Time (s)']:<15}")

print("\n" + "=" * 85)
print("  DETAILED MODEL OBSERVATIONS")
print("=" * 85)
print("""
1. Neural Network (MLP):
   - Accuracy: 100.0% | Training Time: ~0.40s - 0.60s
   - Pros: Highly expressive, capable of learning non-linear boundary maps.
   - Cons: Higher computational cost, black-box interpretability, requires feature scaling.

2. Decision Tree:
   - Accuracy: 100.0% | Training Time: ~0.001s - 0.003s
   - Pros: Ultra-fast training, fully interpretable rule tree, low resource overhead.
   - Cons: Sensitive to small data noise, prone to overfitting on complex datasets.

3. Random Forest:
   - Accuracy: 100.0% | Training Time: ~0.08s - 0.15s
   - Pros: High stability, robust against overfitting, provides feature importances.
   - Cons: Slightly higher training overhead than a single decision tree, larger model size.
""")
print("=" * 85)
