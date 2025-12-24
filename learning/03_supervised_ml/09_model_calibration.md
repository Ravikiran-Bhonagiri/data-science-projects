# Model Calibration: When "90% Confident" is a Lie

You trained a model. It predicts a patient has a 90% chance of cancer.
You operate. It turns out they were healthy.
The model was "confident," but it was **wrong**.

**Calibration** is the difference between a model's *predicted probability* and the *actual frequency* of the event.
-   **Perfect Calibration:** If a model predicts 70% rain for 10 days, it should rain **exactly 7 times**.
-   **Uncalibrated:** The model predicts 99% simply because the decision boundary is far away (margin), not because it's actually certain.

### 🧠 Why does this matter? (The "Cost" of Confidence)
1.  **Risk-Sensitive Domains:** In Medicine or Finance, you don't act on the *prediction* (Buy/Sell), you act on the *Expected Value* ($Prob \times Value$). If Prob is wrong, your money is gone.
2.  **Human-AI Trust:** If a doctor sees a "99% Confident" AI fail twice, they will never trust it again.

---

## 1. The Mathematics of Miscalibration

Why do models lie? It typically comes down to their loss function.

### A. The Liars (Uncalibrated)
1.  **Naive Bayes:** Pushes probabilities to 0 and 1 because of the feature independence assumption. It is **Overconfident**.
2.  **Random Forest:** Averaging trees pulls predictions away from 0 and 1. It rarely predicts >0.9 or <0.1. It is **Underconfident** at the extremes.
3.  **SVM:** Maximizes margin, not probability. Raw values are distances, not probabilities.

### B. The Honest Models (Calibrated)
1.  **Logistic Regression:** It minimizes **Log-Loss**, which is a strictly proper scoring rule.
    $$LogLoss = - \frac{1}{N} \sum (y_i \log(p_i) + (1-y_i) \log(1-p_i))$$
    Minimizing this *forces* $p_i$ to match the true distribution.

---

## 2. Measuring Calibration: ECE (The Gold Standard)

Reliability diagrams are visual. For a single number, we use **Expected Calibration Error (ECE)**.

We divide predictions into $M$ bins (e.g., 10 bins of 10% width).
$$ECE = \sum_{m=1}^{M} \frac{|B_m|}{N} | Accuracy(B_m) - Confidence(B_m) |$$

-   $|B_m|$: Number of samples in bin $m$.
-   **Accuracy($B_m$):** Actual fraction of positives in that bin.
-   **Confidence($B_m$):** Average predicted probability in that bin.

```python
import numpy as np
from sklearn.calibration import calibration_curve

def expected_calibration_error(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    # We need the weights (number of items) per bin to implement the formula
    # Sklearn's calibration_curve doesn't return weights, so we calculate them:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_prob, bin_edges) - 1
    
    ece = 0.0
    for i in range(n_bins):
        mask = binids == i
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            weight = np.sum(mask) / len(y_true)
            ece += weight * np.abs(bin_acc - bin_conf)
            
    return ece
```

---

## 3. Fixing Miscalibration (Post-Processing)

We can wrap our broken model $f(x)$ in a second model $g(p)$ that maps "wrong probability" to "right probability".
$$P_{calibrated} = g(f(x))$$

### Method A: Platt Scaling (Parametric)
Assumes the distortion is S-shaped (Logistic). Originally designed for SVMs.
$$P(y=1|f) = \frac{1}{1 + e^{-(Af + B)}}$$
-   **Pros:** Needs little data. Works well if error is symmetric.
-   **Cons:** Cannot correct non-sigmoid distortions (e.g., Random Forest's "S-curve").

### Method B: Isotonic Regression (Non-Parametric)
Finds a non-decreasing step function that minimizes error (using the **Pool Adjacent Violators Algorithm** - PAVA).
-   **Pros:** Can fix ANY monotonic distortion. **Best for Random Forests.**
-   **Cons:** Prone to overfitting on small datasets (<1000 samples).

### Method C: Temperature Scaling (For Neural Nets)
A single parameter $T$ scales the logits before the Softmax.
$$Softmax(z/T)$$
-   $T > 1$: Softens the distribution (reduces overconfidence).
-   $T < 1$: Sharpens it.
-   **Standard for Deep Learning.**

---

## 4. Advanced Implementation

```python
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss

# 1. Train Base Model (The Liar)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# 2. Diagnose
probs_uncal = rf.predict_proba(X_val)[:, 1]
print(f"Brier (Uncalibrated): {brier_score_loss(y_val, probs_uncal):.4f}")

fig, ax = plt.subplots(figsize=(10, 8))
CalibrationDisplay.from_predictions(y_val, probs_uncal, n_bins=10, ax=ax, name="Random Forest")

# 3. Apply Isotonic Regression (The Fixer)
# Note: 'cv=prefit' means "don't retrain the RF, just learn the calibration on X_val"
cal_rf = CalibratedClassifierCV(rf, cv="prefit", method='isotonic')
cal_rf.fit(X_val, y_val)

probs_cal = cal_rf.predict_proba(X_val)[:, 1]
print(f"Brier (Calibrated): {brier_score_loss(y_val, probs_cal):.4f}")

# 4. Compare
CalibrationDisplay.from_predictions(y_val, probs_cal, n_bins=10, ax=ax, name="Calibrated RF")
plt.show()
```

---

## 5. When NOT to Calibrate

Use this checklist. Do NOT calibrate if:
1.  **You only care about Ranking (AUC):** Calibration does NOT change the order of predictions. It only changes the values. Your AUC will stay exactly the same.
2.  **Dataset is Tiny:** Isotonic regression increases model complexity. On <1000 rows, you will overfit the calibration set.
3.  **Class Imbalance is Extreme:** If you have 99.9% negatives, calibration plots become noisy and arguably useless without specialized binning schemes.
