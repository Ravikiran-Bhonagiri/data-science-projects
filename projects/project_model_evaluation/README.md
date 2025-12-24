<div align="center">

# ✅ Model Evaluation Framework Project

### *Comprehensive Assessment Toolkit for ML Models*

![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![Type](https://img.shields.io/badge/Type-Evaluation-blue?style=flat-square)
![Notebooks](https://img.shields.io/badge/Notebooks-4-blue?style=flat-square)
![Level](https://img.shields.io/badge/Level-Intermediate-yellow?style=flat-square)

**Classification & regression metrics, cross-validation, profit curves, and calibration**

[🎯 Goal](#-project-goal) • [📚 Notebooks](#-notebooks) • [📊 Metrics](#-metrics-covered) • [🚀 Run It](#-quick-start)

</div>

---

## 🎯 Project Goal

> **"Accuracy is not enough. Choose the right metric and validate properly."**

**Build a production-ready evaluation module** with comprehensive metrics, cross-validation strategies, and business-aligned assessment for all future projects.

---

## 📚 Notebooks

**4 comprehensive notebooks covering the complete evaluation workflow**

<table>
<tr>
<td width="50%">

### 🎯 Notebook 1: Baseline & Accuracy Trap
**`01_baseline_and_accuracy_trap.ipynb`**

**The Accuracy Fallacy:**
- ✅ **Dummy Classifier:** Baseline setup
- ✅ **Imbalanced Data:** 95% accuracy trap
- ✅ **Confusion Matrix:** True understanding
- ✅ **Class Distribution:** Impact demonstration

**Key Example:**
```
Fraud Detection (99% non-fraud):
- Predict all "no fraud" → 99% accuracy!
- But catches 0% of actual fraud ❌
```

**Lesson:** Accuracy lies with imbalance

---

### 📊 Notebook 2: Advanced Metrics
**`02_advanced_metrics.ipynb`**

**Beyond Accuracy:**
- ✅ **Precision:** Of predicted positives, how many correct?
- ✅ **Recall:** Of actual positives, how many caught?
- ✅ **F1-Score:** Harmonic mean balance
- ✅ **ROC-AUC:** Threshold-independent
- ✅ **PR-AUC:** Better for imbalance
- ✅ **Matthews** Correlation:** Balanced metric

**Comparison:**
- Metric selection guide
- imbalance-class handling
- Multi-class extensions

**Output:** Metric decision flowchart

</td>
<td width="50%">

### 💰 Notebook 3: Profit Curves
**`03_profit_curves.ipynb`**

**Business-Aligned Metrics:**
- ✅ **Cost-Benefit Matrix:**
  - True Positive value
  - False Positive cost
  - False Negative cost
- ✅ **Profit Curves:** $ vs threshold
- ✅ **Optimal Threshold:** Max profit point
- ✅ **ROI Calculation:** Expected value

**Real Example:**
```
Marketing Campaign:
- True Positive: +$50 (conversion)
- False Positive: -$5 (wasted ad)
- Optimal threshold: 0.35 (not 0.5!)
- Profit increase: +$45K
```

**Output:** Business-optimized thresholds

---

### 📐 Notebook 4: Calibration
**`04_calibration.ipynb`**

**Probability Reliability:**
- ✅ **Calibration Plots:** Predicted vs actual
- ✅ **Brier Score:** Probabilistic accuracy
- ✅ **Calibration Methods:**
  - Platt scaling (logistic)
  - Isotonic regression
- ✅ **Model Comparison:** Pre/post calibration

**Why It Matters:**
- "80% probability" should mean 80%!
- Critical for decision-making
- Affects cost-sensitive predictions

**Output:** Calibrated probability models

</td>
</tr>
</table>

---

## 📊 Metrics Covered

<details>
<summary><strong>📊 Classification Metrics (8 Metrics)</strong></summary>

| Metric | Formula | When to Use | Range |
|--------|---------|-------------|-------|
| **Accuracy** | (TP+TN)/Total | Balanced classes only | [0,1] |
| **Precision** | TP/(TP+FP) | Minimize false alarms | [0,1] |
| **Recall** | TP/(TP+FN) | Catch all positives | [0,1] |
| **F1-Score** | 2×P×R/(P+R) | Balance P and R | [0,1] |
| **ROC-AUC** | Area under curve | Threshold-free | [0,1] |
| **PR-AUC** | Precision-Recall area | Imbalanced data | [0,1] |
| **MCC** | Correlation coefficient | Balanced, any class ratio | [-1,1] |
| **Cohen's Kappa** | Agreement vs chance | Multi-class | [-1,1] |

</details>

<details>
<summary><strong>📏 Regression Metrics (6 Metrics)</strong></summary>

| Metric | Formula | Interpretation | Best Value |
|--------|---------|----------------|------------|
| **MAE** | Σ\|y-ŷ\|/n | Average error (same units) | 0 |
| **MSE** | Σ(y-ŷ)²/n | Penalizes large errors | 0 |
| **RMSE** | √MSE | Error in original units | 0 |
| **R²** | 1-SS_res/SS_tot | % variance explained | 1 |
| **Adj R²** | Penalized R² | Accounts for features | 1 |
| **MAPE** | Σ\|y-ŷ\|/y | Percentage error | 0 |

</details>

<details>
<summary><strong>🔄 Cross-Validation Methods</strong></summary>

**Standard:**
- K-Fold (k=5 or 10)
- Stratified K-Fold (preserves class distribution)
- Leave-One-Out (expensive but thorough)

**Specialized:**
- Time Series CV (respects temporal order)
- Group K-Fold (keeps groups together)
- Repeated K-Fold (multiple runs)

**Code Example:**
```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

# For imbalanced classification
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='f1')

print(f"F1: {scores.mean():.3f} ± {scores.std():.3f}")
```

</details>

---

## 💡 Key Learnings

<details>
<summary><strong>The Accuracy Trap (Real Example)</strong></summary>

**Credit Card Fraud Detection:**

```
Dataset: 10,000 transactions, 100 fraudulent (1%)

Model 1: Predicts all "legitimate"
- Accuracy: 99% ✅
- Recall: 0% ❌ (catches NO fraud!)

Model 2: Tuned for recall
- Accuracy: 94% 
- Recall: 85% ✅ (catches 85 of 100 frauds)

Winner: Model 2 (despite lower accuracy!)
```

**Lesson:** Accuracy is misleading with imbalance

</details>

<details>
<summary><strong>Optimal Threshold ≠ 0.5</strong></summary>

**Marketing Campaign Example:**

```
Cost-Benefit:
- True Positive: +$100 (customer acquired)
- False Positive: -$10 (wasted marketing)

Threshold Analysis:
- 0.5 (default): $50K profit
- 0.3 (optimized): $78K profit (+56%!)
- 0.7 (too high): $22K profit

Optimal: 0.32 (maximizes expected profit)
```

**Lesson:** Business metrics drive threshold selection

</details>

<details>
<summary><strong>Calibration Matters</strong></summary>

**Medical Diagnosis Example:**

**Uncalibrated Model:**
```
Says "70% probability of disease"
Actual rate: 40% (overconfident!)
```

**After Platt Scaling:**
```
Says "70% probability"
Actual rate: 68% ✅ (well-calibrated)
```

**Impact:** Doctors can trust probabilities for treatment decisions

</details>

---

## 🚀 Quick Start

### Installation

```bash
# Navigate to project
cd projects/project_model_evaluation

# Install dependencies
pip install -r requirements.txt
```

### Run Notebooks

```bash
# Launch Jupyter
jupyter notebook notebooks/

# Execute in order:
# 1. 01_baseline_and_accuracy_trap.ipynb
# 2. 02_advanced_metrics.ipynb
# 3. 03_profit_curves.ipynb
# 4. 04_calibration.ipynb
```

---

## 🎯 Metric Selection Guide

**Quick Decision Tree:**

```
Your Problem:
│
├─ Imbalanced Classes?
│  ├─ Yes → Use F1, PR-AUC, MCC (NOT accuracy)
│  └─ No → Accuracy OK
│
├─ Cost-Sensitive?
│  ├─ Yes → Use Profit Curves (custom threshold)
│  └─ No → Use standard metrics
│
├─ Need Probabilities?
│  ├─ Yes → Calibrate model (Platt/Isotonic)
│  └─ No → Hard predictions OK
│
└─ Multi-Class?
   ├─ Yes → Macro/Micro/Weighted averaging
   └─ No → Binary metrics
```

---

## 💼 Business Value

**Production-Ready Evaluation:**

| Scenario | Metric | Why |
|----------|--------|-----|
| **Fraud Detection** | Recall, F2 | Minimize false negatives|
| **Email Spam** | Precision, F0.5 | Minimize false positives |
| **Credit Approval** | Profit Curve | Maximize revenue |
| **Medical Diagnosis** | Calibrated probabilities | Trust thresholds |
| **Recommendation** | PR-AUC | Imbalanced (few clicks) |

---

## 🎓 What You'll Master

<table>
<tr>
<td width="50%">

### 📊 Metrics Mastery
- ✅ 8 classification metrics
- ✅ 6 regression metrics
- ✅ When to use each
- ✅ Interpretation pitfalls
- ✅ Multi-class extensions

</td>
<td width="50%">

### 🔬 Advanced Techniques
- ✅ Cross-validation strategies
- ✅ Profit curve optimization
- ✅ Model calibration
- ✅ Threshold tuning
- ✅ Business-aligned evaluation

</td>
</tr>
</table>

---

## 📁 Project Structure

```
project_model_evaluation/
├── 📊 notebooks/          # 4 comprehensive guides
│   ├── 01_baseline_and_accuracy_trap.ipynb
│   ├── 02_advanced_metrics.ipynb
│   ├── 03_profit_curves.ipynb
│   └── 04_calibration.ipynb
│
├── 🔧 src/                # Reusable evaluation functions
│   ├── metrics.py
│   ├── calibration.py
│   ├── profit_curves.py
│   └── visualization.py
│
└── 📄 requirements.txt    # Dependencies
```

---

## 🏆 Key Takeaways

> **"The right metric can change everything. A 0.32 threshold (not 0.5) increased profit by $28K in the marketing example."**

**For Data Scientists:**
- ✅ Accuracy fails with imbalance
- ✅ F1, PR-AUC better for real problems
- ✅ Business metrics beat statistical metrics
- ✅ Calibration enables trust in probabilities

---

## 🔗 Related Resources

**Continue Learning:**
- 📚 [Evaluation Module](../../learning/05_evaluation/) - Theory & concepts
- 📞 [Telco Churn](../project_telco_churn/) - Metrics in action ($3.9M)
- ⚙️ [Feature Engineering](../project_feature_engineering/) - Pipeline integration

---

<div align="center">

**Measure Right, Build Right** ✅

*4 notebooks • 14+ metrics • Business-aligned evaluation*

[⬅️ Feature Engineering](../project_feature_engineering/) • [🏠 Home](../../README.md) • [➡️ Text EDA](../project_text_eda/)

</div>
