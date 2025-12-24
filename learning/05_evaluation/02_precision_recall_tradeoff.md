# Precision-Recall Tradeoff: The Cancer Screening Dilemma

**The FDA Problem:** You've built an AI to detect breast cancer from mammograms.

**Scenario A: Maximize Recall (Catch Every Cancer)**
- Set threshold = 0.15 (flag anything with >15% cancer probability)
- **Result:** You catch 98% of cancers (Recall = 0.98)
- **Cost:** 60% false positive rate
- **Real impact:** 60 out of 100 healthy women get called back for invasive biopsies
- **Outcome:** Class-action lawsuit for unnecessary medical procedures. FDA approval denied.

**Scenario B: Maximize Precision (Only Flag When Sure)**
- Set threshold = 0.90 (flag only when >90% sure)
- **Result:** 95% of your flags are actual cancers (Precision = 0.95)
- **Cost:** You only catch 45% of cancers (Recall = 0.45)
- **Real impact:** 55 women out of 100 with cancer get "all clear" results
- **Outcome:** Deaths. Malpractice lawsuits. Criminal investigation.

**The Truth:** There is no perfect threshold. Every choice trades one error for another.

---

## The Business Impact: Credit Card Fraud

**Your bank processes 10M transactions/day. 0.1% are fraudulent (10,000 fraud cases).**

**Model 1: High Precision (threshold=0.95)**
- Precision: 90%
- Recall: 30%
- **What happens:** You catch 3,000 fraud cases, miss 7,000
- **Customer experience:** Legitimate transactions rarely blocked (low false alarms)
- **Financial loss:** $7M/day in undetected fraud ($2.55B/year)

**Model 2: High Recall (threshold=0.20)**
- Precision: 15%  
- Recall: 85%
- **What happens:** You catch 8,500 fraud cases
- **Customer experience:** 100,000 legitimate transactions blocked daily
- **Outcome:** Customers leave bank due to constant declined cards

**The Real Solution:**
```python
# Different thresholds for different transaction amounts
if amount < 50:
    threshold = 0.80  # High precision (don't annoy customers)
elif amount < 500:
    threshold = 0.50  # Balanced
else:  # amount >= 500
    threshold = 0.15  # High recall (catch expensive fraud)
```

---

## The Metrics

## 2. Visualizing the Curve

We don't pick a threshold blindly. We plot them.

```python
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Get probabilities (This is crucial! No .predict())
y_scores = model.predict_proba(X_test)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)

plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
plt.xlabel("Threshold")
plt.legend()
plt.title("The Tradeoff: Choosing your destiny")
plt.show()
```

### How to pick the Point?
Look at the graph.
*   Do you need Precision > 90%? Find where the Blue line hits 0.9.
*   Check the corresponding Green line (Recall). Is it acceptable (e.g., 40%)?
*   If yes, **pick that threshold.**

---

---

## 3. The ROC Curve (Receiver Operating Characteristic)

The PR Curve focuses on *Precision* (Minimizing False Alarms).
The **ROC Curve** focuses on **Separation**.

It plots **Recall (TPR)** vs **False Positive Rate (FPR)**.
*   **TPR (Y-axis):** "How many of the bad guys did we catch?" (Higher is better).
*   **FPR (X-axis):** "How many innocent people did we arrest?" (Lower is better).

### The AUC (Area Under Curve)
The **AUC** is a single number (0.0 to 1.0) that summarizes the curve.
*   **AUC = 0.5**: Random Guessing (A diagonal line).
*   **AUC = 1.0**: Perfect Separation (God Mode).
*   **AUC = 0.0**: Perfectly Wrong (It predicts "Health" when it's "Cancer").

**Intuitive Meaning of AUC:**
If you pick a random **Positive** point and a random **Negative** point, what is the probability that your model ranks the Positive higher?
*   AUC 0.82 means: "82% of the time, our model gives a higher score to the fraud than the non-fraud."

```python
from sklearn.metrics import roc_curve, roc_auc_score

# 1. Calculate FPR, TPR
fpr, tpr, thresholds = roc_curve(y_test, y_scores)

# 2. Calculate AUC
auc_score = roc_auc_score(y_test, y_scores)

# 3. Plot
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], 'k--') # Random line
plt.xlabel("False Positive Rate (False Alarms)")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.legend()
plt.show()
```

---

## 4. ROC vs PR Curves (The War of the Graphs)

| Curve | Name | Axes | Use When... |
|-------|------|------|-------------|
| **ROC** | Receiver Operating Characteristic | TPR vs FPR | Data is **Balanced** (50/50). You care about overall ranking. |
| **PR** | Precision-Recall | Precision vs Recall | Data is **Imbalanced** (Fraud = 1%). You care about the minority class. |

**Why the difference?**

---

## 5. Implementation: How to actually change it?

Scikit-Learn models don't have a `threshold` parameter. `model.predict()` is hardcoded to 0.5.
**You must do it manually.**

```python
# 1. Get the probabilities (The raw scores)
y_prob = model.predict_proba(X_test)[:, 1]

# 2. Define your business logic
# "We need 99% Precision. The graph says that happens at threshold 0.85."
custom_threshold = 0.85

# 3. Apply the threshold
y_pred_custom = (y_prob >= custom_threshold).astype(int)

# 4. Evaluate
print(f"Precision with 0.85 threshold: {precision_score(y_test, y_pred_custom)}")
```

### Finding the "Optimal" Threshold (F-Score or Youden's J)
If you don't have a specific business constraint, you can maximize the F1-Score or Youden's Index (`Sensitivity + Specificity - 1`).

```python
import numpy as np
from sklearn.metrics import f1_score

thresholds = np.arange(0, 1, 0.01)
scores = [f1_score(y_test, (y_prob >= t).astype(int)) for t in thresholds]

# Find the peak
ix = np.argmax(scores)
print(f"Best Threshold={thresholds[ix]:.2f}, F-Score={scores[ix]:.3f}")
```
