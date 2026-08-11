# Group 6 - Machine Learning Version Control Project

## Part B: Collaborative ML Model Development

**Course:** Collaborative Software Development  
**Repository:** [Kare100/ml-version-control-group6](https://github.com/Kare100/ml-version-control-group6)

---

## Group Members & Contributions

| Member | GitHub Handle | Model Implemented |
|--------|--------------|-------------------|
| Valerie W. | Kare100 | Neural Network (MLP) |
| Hussein Hajir Aden | Huska22 | Decision Tree |
| Member 3 | RichHommie241 | Random Forest |

---

## Models & Results (Iris Dataset, 80/20 split, `random_state=42`)

| Model | Accuracy | Precision | Recall | F1-Score | Train Time |
|-------|----------|-----------|--------|----------|------------|
| Neural Network (MLP) | 100.00% | 1.0000 | 1.0000 | 1.0000 | ~0.57s |
| Decision Tree | 100.00% | 1.0000 | 1.0000 | 1.0000 | ~0.003s |
| Random Forest | 100.00% | 1.0000 | 1.0000 | 1.0000 | ~0.12s |

---

## How to Run

```bash
# Install dependencies
pip install scikit-learn pandas numpy

# Run individual models
python neural_network_model.py
python decision_tree_model.py
python random_forest_model.py

# Run full comparison suite
python compare_models.py
```

---

## Files

- `neural_network_model.py` - MLP Classifier (Valerie W.)
- `decision_tree_model.py` - Decision Tree Classifier (Hussein Hajir)
- `random_forest_model.py` - Random Forest Classifier (RichHommie241)
- `compare_models.py` - Unified benchmarking suite (all models)
- `Part_A_Neural_Network_Report.docx` - Individual report (Part A)
- `Part_B_Group_Practical_Report.docx` - Group report (Part B)