<div align="center">

# 🤖 Module 3: Supervised Machine Learning

### *Teaching Machines to Predict*

![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![Difficulty](https://img.shields.io/badge/Difficulty-Intermediate-yellow?style=flat-square)
![Guides](https://img.shields.io/badge/Guides-10-orange?style=flat-square)

**Master the algorithms powering 90% of industry ML jobs**

[🎯 Quick Guide](#-which-model) • [📚 Learning Path](#-learning-path) • [🚀 Get Started](#-typical-workflow)

</div>

---

## 💡 The Reality

> **"Master XGBoost and Random Forests → Qualified for 90% of industry ML jobs"**

Supervised Learning = Teaching machines to map **Inputs (X)** → **Outputs (y)**

---

## 📚 Learning Path

<table>
<tr>
<td width="33%">

### 🎯 Phase 1: Foundations
**Interview Essentials**

**[01. Linear Regression](./01_linear_regression.md)**
- Predict numbers
- Understand coefficients
- Baseline for everything

**[02. Logistic Regression](./02_logistic_regression.md)**
- Binary classification
- Probability outputs
- Interpretable results

</td>
<td width="33%">

### 🏆 Phase 2: Production
**Kaggle Winners**

**[03. Decision Trees](./03_decision_trees.md)**
- Interpretable logic
- "20 Questions" approach

**[04. Random Forest](./04_ensemble_bagging.md)**
- Wisdom of crowds
- Robust, minimal tuning

**[05. XGBoost/LightGBM](./05_ensemble_boosting.md)**
- ⭐ **State-of-the-art**
- Highest accuracy
- Industry standard

</td>
<td width="33%">

### ⚡ Phase 3: Specialized
**Domain-Specific**

**[06. SVM](./06_support_vector_machines.md)**
- High-dimensional data
- Small datasets

**[07. KNN](./07_knn_algorithms.md)**
- Recommendations
- Anomaly detection

**[08. Naive Bayes](./08_naive_bayes.md)**
- Text classification
- NLP baseline

</td>
</tr>
</table>

### 🎓 Phase 4: Expert Techniques

**[09. Model Calibration](./09_model_calibration.md)** - Make probabilities trustworthy  
**[10. Hyperparameter Tuning](./10_hyperparameter_tuning.md)** - Automate optimization with Optuna

---

## 🎯 Which Model?

**Quick decision guide:**

| Your Situation | Best Model | Why? |
|----------------|------------|------|
| 📊 **Tabular data** (Excel-like) | **XGBoost / LightGBM** | State-of-the-art accuracy |
| ⏱️ **Need it working in 5 mins** | **Random Forest** | No tuning, handles messy data |
| 📝 **Text data** (NLP) | **Naive Bayes / SVM** | Handles sparse high-dim data |
| 📖 **Explanation critical** | **Linear / Logistic / Tree** | Show logic to stakeholders |
| ⚡ **Ultra-low latency** (<1ms) | **Logistic Regression** | Just a dot product |

---

## 🛠️ Typical Workflow

```python
# 1. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Baseline (Always start simple!)
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)  # e.g., 70% accuracy

# 3. Real Model
model = XGBClassifier()
model.fit(X_train, y_train)  # e.g., 85% accuracy

# 4. Tune Hyperparameters
study = optuna.create_study()
study.optimize(objective_function)  # e.g., 88% accuracy

# 5. Evaluate
print(classification_report(y_test, predictions))
```

---

## 🎯 What You'll Master

<table>
<tr>
<td width="50%">

### 📊 Core Algorithms
- ✅ Linear & Logistic Regression
- ✅ Decision Trees
- ✅ Random Forests
- ✅ XGBoost & LightGBM
- ✅ SVM, KNN, Naive Bayes

</td>
<td width="50%">

### 🚀 Advanced Skills
- ✅ Train/test splitting
- ✅ Cross-validation
- ✅ Hyperparameter tuning
- ✅ Model calibration
- ✅ Ensemble methods
- ✅ Production deployment

</td>
</tr>
</table>

---

## 📚 Complete Guide List

**All guides in this module:**

1. **[Linear Regression](./01_linear_regression.md)** - Predict continuous values and understand coefficient interpretation
2. **[Logistic Regression](./02_logistic_regression.md)** - Binary classification with probability outputs
3. **[Decision Trees](./03_decision_trees.md)** - Interpretable "20 Questions" logic for predictions
4. **[Ensemble Bagging / Random Forest](./04_ensemble_bagging.md)** - Wisdom of crowds for robust predictions
5. **[Ensemble Boosting / XGBoost/ LightGBM](./05_ensemble_boosting.md)** - State-of-the-art gradient boosting for highest accuracy
6. **[Support Vector Machines](./06_support_vector_machines.md)** - Handle high-dimensional, small datasets effectively  
7. **[K-Nearest Neighbors (KNN)](./07_knn_algorithms.md)** - Instance-based learning for recommendations and anomaly detection
8. **[Naive Bayes](./08_naive_bayes.md)** - Fast baseline for text classification and NLP
9. **[Model Calibration](./09_model_calibration.md)** - Make probability predictions trustworthy
10. **[Hyperparameter Tuning](./10_hyperparameter_tuning.md)** - Automate optimization with Optuna and grid search

---

<div align="center">

**Master Prediction, Master Data Science** 🤖

*10 comprehensive guides • From Linear Regression to XGBoost*

[⬅️ Statistics](../02_statistics/) • [🏠 Home](../../README.md) • [➡️ Unsupervised ML](../04_unsupervised_ml/)

</div>
