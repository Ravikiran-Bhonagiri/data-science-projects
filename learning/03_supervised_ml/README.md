# Module 3: Supervised Machine Learning

Welcome to the **most commonly used** domain of Data Science.
Supervised Learning is about teaching a machine to map **Inputs (X)** to **Outputs (y)**.

If you master the algorithms in this folder—specifically **XGBoost** and **Random Forests**—you are qualified for 90% of industry ML jobs.

---

## 📚 The Grandmaster Series

We have broken this module down into **10 specialized guides**.
Do not read them all at once. Follow this path:

### Phase 1: The Foundations (Interview Essentials)
*Start here to understand the core math and "easy" models.*
1.  **[Linear Regression](./01_linear_regression.md):** The "Crystal Ball" for predicting numbers.
2.  **[Logistic Regression](./02_logistic_regression.md):** The "Yes/No" decision maker.

### Phase 2: The Modern Toolkit (Kaggle Winners)
*This is what you will actually use in production. Master these.*
3.  **[Decision Trees](./03_decision_trees.md):** The "20 Questions" logic.
4.  **[Random Forests](./04_ensemble_bagging.md):** The "Wisdom of Crowds." (Robust & Easy)
5.  **[Boosting (XGBoost/LGBM)](./05_ensemble_boosting.md):** The "Speed Racer." (Highest Accuracy)

### Phase 3: The Specialized Tools
*Use these for specific data types (Text, Small Data, etc).*
6.  **[Support Vector Machines](./06_support_vector_machines.md):** For high-dimensional, small datasets.
7.  **[KNN (K-Nearest Neighbors)](./07_knn_algorithms.md):** For recommendation systems & anomalies.
8.  **[Naive Bayes](./08_naive_bayes.md):** The baseline for NLP/Text classification.

### Phase 4: Expert Techniques
*How to turn a "good" model into a "perfect" one.*
9.  **[Model Calibration](./09_model_calibration.md):** Why "90% confidence" is often a lie.
10. **[Hyperparameter Tuning](./10_hyperparameter_tuning.md):** Automating optimization with **Optuna**.

---

## 🧠 Quick Reference: Which Model?

| Scenario | Recommendation | Why? |
|----------|----------------|------|
| **Tabular Data (Excel-like)** | **XGBoost / LightGBM** | State-of-the-art accuracy. |
| **"I need it working in 5 mins"** | **Random Forest** | No tuning required, handles messy data. |
| **Text Data (NLP)** | **Naive Bayes / SVM** | Handles sparse, high-dim data well. |
| **Explanation is #1 Priority** | **Linear / Logistic / Tree** | You can show the coefficient/logic to a boss. |
| **Ultra-Low Latency (<1ms)** | **Logistic Regression** | It's just a dot product. Instant. |

---

## 🛠️ Typical Workflow

```python
# 1. Split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# 2. Preprocess (Scale/Encode)
# Trees don't need scaling, but SVM/KNN/Linear DO independent scaling!

# 3. Baseline (Always start simple!)
dumb_model = DummyClassifier(strategy="most_frequent")
dumb_model.fit(X_train, y_train) # e.g. 70% accuracy

# 4. The Real Model (e.g. XGBoost)
model = XGBClassifier()
model.fit(X_train, y_train) # e.g. 85% accuracy

# 5. Tune
study = optuna.create_study()
study.optimize(... ) # e.g. 88% accuracy

# 6. Evaluate
print(classification_report(y_test, preds))
```
