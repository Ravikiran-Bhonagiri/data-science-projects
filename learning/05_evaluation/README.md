<div align="center">

# ⚖️ Module 5: Model Evaluation & Metrics

### *Measuring What Matters*

![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![Difficulty](https://img.shields.io/badge/Difficulty-Intermediate-yellow?style=flat-square)
![Topics](https://img.shields.io/badge/Topics-9-orange?style=flat-square)

**Master the art of assessing model performance beyond accuracy**

[📊 Classification](#-classification-metrics) • [📏 Regression](#-regression-metrics) • [🔄 Validation](#-cross-validation)

</div>

---

## 💡 Why Evaluation Matters

> **"Accuracy is not enough. Choose the right metric for your problem."**

**Reality check:**
- 95% accuracy on fraud detection = USELESS if you miss all frauds
- Lower RMSE doesn't mean better business value
- Cross-validation prevents overfitting disasters

---

## 📊 Classification Metrics

**For binary and multi-class problems**

<table>
<tr>
<td width="50%">

### Core Metrics

**Accuracy**
```
(TP + TN) / Total
```
⚠️ Misleading with imbalanced classes

**Precision**
```
TP / (TP + FP)
```
✅ "Of predicted positives, how many correct?"

**Recall (Sensitivity)**
```
TP / (TP + FN)
```
✅ "Of actual positives, how many caught?"

**F1-Score**  
```
2 × (Precision × Recall) / (Precision + Recall)
```
✅ Harmonic mean, balances both

</td>
<td width="50%">

### Advanced Metrics

**ROC-AUC**
- Area under ROC curve
- Threshold-independent
- Good for class imbalance

**Precision-Recall AUC**
- Better than ROC for severe imbalance
- Focuses on positive class

**Cohen's Kappa**
- Accounts for chance agreement
- For multi-class

**Matthews Correlation Coefficient**
- Balanced even with imbalance
- Range: -1 to +1

</td>
</tr>
</table>

---

## 📏 Regression Metrics

**For continuous predictions**

| Metric | Formula | When to Use |
|--------|---------|-------------|
| **MAE** | `Σ\|y - ŷ\| / n` | Same unit as target, interpretable |
| **MSE** | `Σ(y - ŷ)² / n` | Penalizes large errors more |
| **RMSE** | `√MSE` | Same units, more sensitive |
| **R²** | `1 - SS_res/SS_tot` | % variance explained (0-1) |
| **Adjusted R²** | Penalized R² | Accounts for # features |
| **MAPE** | `Σ\|y - ŷ\|/y × 100` | Percentage error |

---

## 🔄 Cross-Validation

**Ensure generalization**

<table>
<tr>
<td width="50%">

### Standard Methods

**K-Fold CV**
```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
print(f"Mean: {scores.mean():.3f} ±{scores.std():.3f}")
```

**Stratified K-Fold**
- Preserves class distribution
- Critical for imbalanced data

**Leave-One-Out (LOO)**
- K = n (each sample is test once)
- Expensive but thorough

</td>
<td width="50%">

### Specialized Methods

**Time Series CV**
- No random splits
- Respects temporal order
- Expanding/sliding window

**Group K-Fold**
- Keep groups together
- Patient data, documents

**Repeated K-Fold**
- Run K-fold multiple times
- More robust estimates

</td>
</tr>
</table>

---

## 🎯 Metric Selection Guide

**Choose based on your problem:**

### Imbalanced Classification (e.g., Fraud Detection)
```
❌ Accuracy (misleading)
✅ Precision-Recall AUC
✅ F1-Score
✅ Matthews Correlation Coefficient
```

### Medical Diagnosis (minimize false negatives)
```
✅ Recall (catch all positives)  
✅ F2-Score (weights recall 2×)
⚠️ Precision (less critical here)
```

### Spam Detection (minimize false positives)
```
✅ Precision (avoid blocking real email)
✅ F0.5-Score (weights precision 2×)
⚠️ Recall (some spam OK to miss)
```

### Regression (House Prices)
```
✅ RMSE (penalize big errors)
✅ MAPE (% errors interpretable)
❌ MSE (units squared, hard to interpret)
```

---

## 🛠️ Complete Evaluation Pipeline

```python
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, f1_score, roc_auc_score

# Define multiple metrics
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

# Cross-validate with all metrics
results = cross_validate(
    model, X, y,
    cv=5,
    scoring=scoring,
    return_train_score=True
)

# Analyze
for metric in scoring:
    train_mean = results[f'train_{metric}'].mean()
    test_mean = results[f'test_{metric}'].mean()
    gap = train_mean - test_mean
    
    print(f"{metric}: Train={train_mean:.3f}, Test={test_mean:.3f}, Gap={gap:.3f}")
    if gap > 0.1:
        print("⚠️ Overfitting detected!")
```

---

## 💡 What You'll Master

<table>
<tr>
<td width="50%">

### 📊 Classification
- ✅ Confusion matrix interpretation
- ✅ Precision vs Recall tradeoff
- ✅ ROC and PR curves
- ✅ Multi-class metrics
- ✅ Imbalanced data handling

</td>
<td width="50%">

### 📏 Regression
- ✅ MAE, MSE, RMSE differences
- ✅ R² interpretation
- ✅ Residual analysis
- ✅ Custom metrics
- ✅ Business-aligned metrics

</td>
</tr>
</table>

---

## 🚨 Common Pitfalls

**Avoid these mistakes:**

❌ **Using accuracy on imbalanced data** → 99% "accuracy" predicting all negative  
❌ **Not using cross-validation** → Overfitting goes undetected  
❌ **Data leakage in split** → Unrealistic performance estimates  
❌ **Wrong metric for problem** → Optimizing the wrong objective  
❌ **Ignoring business context** → Statistically good but business-bad model  

---

<div align="center">

**Measure Right, Build Right** ⚖️

*9 topics • Classification + Regression + Validation*

[⬅️ Unsupervised ML](../04_unsupervised_ml/) • [🏠 Home](../../README.md) • [➡️ Feature Engineering](../06_feature_engineering/)

</div>
