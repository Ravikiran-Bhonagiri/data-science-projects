# Feature Selection: The Curse of Dimensionality

**The Cancer Genomics Problem:**
- Dataset: 500 patients
- Features: 20,000 gene expression levels
- Target: Cancer vs Healthy

**What happens without selection:**
```python
model = LogisticRegression()
model.fit(X_train, y_train)  
# Training accuracy: 99.8%
# Test accuracy: 52.1%
```

**Why?** With 20,000 features and only 500 samples, the model finds spurious correlations.
Gene #14,582 correlates with cancer in training purely by random chance.

**The Math:** With 20k random features and α=0.05, you expect ~1,000 "significant" features by luck alone.

---

## Real Production Scenario: NLP Spam Filter

- Vocabulary: 50,000 words
- Actual signal: ~200 words (pharmacy terms, urgency words)
- **Before selection:** Model size 2.3GB, latency 400ms
- **After Boruta:** 187 features, model size 12MB, latency 8ms
- **Business impact:** Can run on mobile devices instead of cloud ($140k/year cost saving)

---

## The Methods

---

## 1. The Basics (Filter Methods)

Fast, but dumb. They look at Y, but ignore model dynamics.

*   **Variance Threshold:** Drop columns where 99.9% of values are the same. (Zero variance = Zero info).
*   **Correlation:** Drop features that correlate with *each other* (Multicollinearity).
    *   If `Feature A` and `Feature B` have 0.99 correlation, keep one. Drop the other.

---

## 2. Wrapper Methods (RFE)

**Recursive Feature Elimination.** The "Hunger Games" of features.
1.  Train model on ALL features.
2.  Find the weakest feature (lowest coefficient/importance).
3.  **Kill it.**
4.  Repeat until you have 10 features left.

*   **Pro:** Finds the *best* subset for that specific model.
*   **Con:** Very slow (trains the model 50 times).

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
rfe = RFE(model, n_features_to_select=5)
fit = rfe.fit(X, y)

print(f"Chosen Features: {fit.support_}")
```

---

## 3. Embedded Methods (Lasso & Trees)

Some models select features *while* they learn.

*   **Lasso (L1 Regularization):**
    *   It forces coefficients to become **Exactly Zero**.
    *   It naturally deletes useless features.
*   **Random Forest Importance:**
    *   "How much did impurity decrease when we split on this feature?"
    *   **Warning:** It is biased towards high-cardinality features (Numerical IDs).

---

## 4. Advanced: Boruta (The Shadow Realm)

Random Forest Importance can be random. How do you know if a score of "0.05" is good?
**Boruta** answers this.

1.  Make a copy of every feature and **Shuffle it** (destroying relationship with Y). These are "Shadow Features".
2.  Train a Random Forest on Real + Shadow features.
3.  **The Test:** Is Real Feature X more important than the **Best Shadow Feature**?
    *   Yes? It's better than random noise. **Keep it.**
    *   No? It's indistinguishable from noise. **Kill it.**

This is statistically robust.

---

## 5. Advanced: SHAP (Game Theory)

**SHAP (SHapley Additive exPlanations)** is the Gold Standard for explaining "Why".
It can also select features.
It calculates the marginal contribution of a feature across all possible combinations.

```python
import shap
import xgboost as xgb

model = xgb.XGBClassifier().fit(X, y)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Summary Plot
shap.summary_plot(shap_values, X, plot_type="bar")
```
User SHAP to find features that push the model output (high magnitude shap values).
