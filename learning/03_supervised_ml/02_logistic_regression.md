# Logistic Regression: The Spam Filter That Actually Works

**The Gmail Spam Revolution (2004):**

Before machine learning, spam filters used keyword blacklists.

**The failure:**
- Block "viagra" → Spammers write "v1agra"
- Block "million dollars" → Spammers use images
- Legitimate emails mentioning drugs (pharmacies) get blocked
- **Accuracy: 60%**, constant cat-and-mouse

**Gmail's solution: Logistic Regression**
```python
# Features:
- word_counts (TF-IDF of 10,000 words)
- sender_reputation
- html_ratio
- link_count

model.predict_proba() → 0.97 spam probability
```

**Result:**
- 99.9% spam detection
- <0.05% false positives (legitimate emails blocked)
- **Key:** Probabilities allow tunable thresholds

---

## Production Use Cases

### 1. **Credit Card Transactions**
**Real-time fraud detection** (must respond in <100ms)
- Feature: amount, merchant_category, time_since_last, location
- Logistic: Fast inference, probability-based risk scoring

### 2. **Medical Diagnosis (FDA Approved)**
**Why hospitals use logistic over deep learning:**
- Interpret coefficients: "Each +10 points blood pressure → 1.5× stroke risk"
- Required for clinical trials and regulatory approval
- Neural net is a black box

---

## 1. The Sigmoid Function
Converts linear equation ($z$) into a probability ($0 < p < 1$).

$$P(y=1) = \frac{1}{1 + e^{-z}}$$

- **Log-Odds:** The raw output of the linear part is the "log-odds" of the positive class.
- **Interpreting Coefficients:**
  - $\beta = 0.69$: Increasing $x$ by 1 increases log-odds by 0.69.
  - $e^{0.69} \approx 2.0$: Increasing $x$ by 1 **doubles** the odds of the event happening.

---

## 2. Decision Boundary & Thresholds
Default threshold is 0.5.
- $p \ge 0.5 \rightarrow 1$
- $p < 0.5 \rightarrow 0$

**Adjusting Thresholds:**
- **Medical diagnosis:** Lower threshold (e.g., 0.1) to catch ALL sick people (High Recall).
- **Spam filter:** Higher threshold (e.g., 0.9) to never block real email (High Precision).

---

## 3. Evaluation Metrics (Beyond Accuracy)

**Accuracy is misleading** when classes are imbalanced (e.g., 99% healthy, 1% sick).

| Metric | Formula | Focus |
|--------|---------|-------|
| **Precision** | $TP / (TP + FP)$ | "Of all predicted Positives, how many are real?" (Avoid False Alarms) |
| **Recall** | $TP / (TP + FN)$ | "Of all real Positives, how many did we find?" (Avoid Missed Cases) |
| **F1-Score** | $2 \times \frac{P \times R}{P + R}$ | Harmonic mean. Balance between P and R. |

### Confusion Matrix
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
```

---

## 4. ROC Curve & AUC

**ROC (Receiver Operating Characteristic):** Plots *True Positive Rate (Recall)* vs *False Positive Rate* at ALL thresholds.

**AUC (Area Under Curve):**
- **0.5:** Random guessing
- **0.7-0.8:** Good
- **0.9+:** Excellent

```python
from sklearn.metrics import roc_curve, roc_auc_score

# Get Probabilities (NOT class labels)
y_probs = model.predict_proba(X_test)[:, 1]

# Calculate AUC
auc = roc_auc_score(y_test, y_probs)
print(f"AUC Score: {auc:.4f}")

# Plot ROC
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--') # Random guess line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.show()
```

---

## 5. Multiclass Classification
What if $y$ has 3+ classes?

### A. One-vs-Rest (OvR)
- Trains N models (Red vs Not-Red, Blue vs Not-Blue).
- Implementation: `LogisticRegression(multi_class='ovr')`

### B. Softmax (Multinomial)
- Generalizes sigmoid to output a probability vector.
- Implementation: `LogisticRegression(multi_class='multinomial')`

```python
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X_train, y_train)
```
