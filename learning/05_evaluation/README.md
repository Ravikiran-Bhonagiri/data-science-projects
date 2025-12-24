# Module 5: Model Evaluation & Storytelling

**Building a model is only 20% of the work.**
The remaining 80% is evaluating if it works, understanding where it fails, and explaining it to someone who signs the checks.

Most Data Scientists stop at "Accuracy = 90%".
**Grandmasters** ask: "Is 90% good? What is the cost of the 10% error? And does this actually make money?"

---

## 📚 The Grandmaster Series

### Phase 1: The Trade-offs (Theory)
*Understanding the balance of the universe.*
1.  **[Bias vs Variance](./01_bias_variance_tradeoff.md):** The "Goldilocks Problem". Overfitting vs Underfitting.
2.  **[Precision vs Recall](./02_precision_recall_tradeoff.md):** The "Spam Filter Dilemma". Tuning the decision threshold.

### Phase 2: The Scorecards (Metrics)
*How to keep score properly.*
3.  **[Classification Metrics](./03_classification_metrics.md):** Beyond Accuracy. F1, Kappa, Log-Loss.
4.  **[Regression Metrics](./04_regression_metrics.md):** The Price is Right. MAE vs RMSE vs RMSLE.

### Phase 3: The Reality Check (Technique)
*Testing yourself rigorously.*
5.  **[Cross-Validation Strategies](./05_cross_validation_strategies.md):** The "Exam". Stratified, GroupKFold, and TimeSeriesSplit.

### Phase 4: The Impact (Business)
*The most important file in this entire portfolio.*
6.  **[Business Value Analysis](./06_business_value_analysis.md):** "The CEO's Language". Calculating ROI, Profit Curves, and Lift.

### Phase 5: Advanced Topics
7.  **[Handling Class Imbalance](./07_class_imbalance.md):** SMOTE, Class Weights, and Resampling for fraud/disease detection.
8.  **[Model Calibration](./08_model_calibration.md):** When your 90% prediction actually means 50%. Platt scaling and temperature scaling.

---

## 🧠 Quick Reference: Which Metric?

| Problem Type | Scenario | Metric | Why? |
|---|---|---|---|
| **Classification** | Balanced Data (50/50) | Accuracy / AUC | Simple proxy for quality. |
| **Classification** | Imbalanced (Fraud / Cancer) | **PR-AUC / F1** | Accuracy is a lie. |
| **Classification** | Probability Matters (Betting) | **Log-Loss** | Punishes false confidence. |
| **Regression** | Cost of Error is Linear | **MAE** | Robust to outliers. |
| **Regression** | Large Errors are Fatal | **RMSE** | Squares errors to punish misses. |
| **Business** | Optimization | **Total Profit** | Evaluate using the `Profit Cost Matrix`. |
