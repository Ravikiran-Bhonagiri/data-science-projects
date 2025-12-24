# Model Calibration: When Probabilities Lie

**The Insurance Pricing Catastrophe:**

Insurance company deploys ML model to price policies.

**Model confidence:**
- "This customer has 80% chance of filing a claim"
- Price premium accordingly: $800/year

**Reality after 1 year:**
- Of all customers where model said "80% risk"
- Actual claim rate: **35%**

**Financial impact:**
- Overcharged customers → They leave
- $120M in lost premiums
- Class-action lawsuit for discriminatory pricing

**The problem:** Model's **predicted probabilities were not calibrated**.

---

## What is Calibration?

**A calibrated model:**
- Says "70% probability" → Actually happens 70% of the time
- If you bucket all "90% predictions" → ~90% are correct

**An uncalibrated model:**
- Says "90% probability" → Actually happens 50% of the time
- The probability number is **meaningless**

**Why models are uncalibrated:**
- Random Forest: Probabilities stuck near 0.1 or 0.9 (rarely in between)
- XGBoost: Overconfident (says 95% when it should say 70%)
- Neural Networks: Can go either way
- SVM: Not even designed to output probabilities

---

## Measuring Calibration: Reliability Diagrams

```python
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Get predicted probabilities
y_proba = model.predict_proba(X_test)[:, 1]

# Calculate calibration curve
fraction_of_positives, mean_predicted_value = calibration_curve(
    y_test, y_proba, n_bins=10
)

# Plot
plt.plot(mean_predicted_value, fraction_of_positives, 's-')
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve')
plt.show()
```

**Reading the plot:**
- Points on diagonal = well calibrated
- Points above diagonal = underconfident (says 40%, actually 60%)
- Points below diagonal = overconfident (says 80%, actually 50%)

---

## Solution 1: Platt Scaling (Logistic Calibration)

**The idea:** Fit a logistic regression to map raw scores to calibrated probabilities.

```python
from sklearn.calibration import CalibratedClassifierCV

# Train base model
base_model = RandomForestClassifier()
base_model.fit(X_train, y_train)

# Calibrate using validation set
calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv='prefit')
calibrated_model.fit(X_val, y_val)  # Use held-out validation set

# Now probabilities are calibrated
y_proba_calibrated = calibrated_model.predict_proba(X_test)[:, 1]
```

**When to use:**
- Model is systematically too confident or not confident enough
- Sigmoid shape to the miscalibration
- Works well for SVMs, boosted models

---

## Solution 2: Isotonic Regression (Non-Parametric)

**More flexible than Platt scaling:**

```python
calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv='prefit')
calibrated_model.fit(X_val, y_val)
```

**When to use:**
- Calibration error is non-monotonic (wiggly)
- You have lots of data (needs more samples than Platt)
- Works well for Random Forests, AdaBoost

**Warning:** Can overfit if validation set is too small (<10k samples).

---

## Solution 3: Temperature Scaling (Neural Networks)

**For deep learning models:**

```python
import torch

class TemperatureScaling(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature

# Train temperature on validation set
temp_model = TemperatureScaling(base_model)
# ... optimize temperature to minimize NLL on validation
```

**What it does:**
- Divides logits by temperature T
- T > 1: Makes model less confident (softens probabilities)
- T < 1: Makes model more confident (sharpens probabilities)

**Production use:** Before serving a neural network, always apply temperature scaling.

---

## Business Impact: When Calibration Matters

### 1. **Insurance/Credit Scoring**
**Requirement:** Probabilities must equal actual risk
**Why:** Regulators audit pricing fairness
**Consequence of failure:** Lawsuits, license revocation

### 2. **Medical Decision Support**
**Requirement:** Doctor needs to trust the "37% cancer" number
**Why:** Treatment decisions depend on it
**Consequence of failure:** Wrong treatment, malpractice

### 3. **Betting/Trading**
**Requirement:** Use probability to calculate expected value
**Why:** $EV = P(win) × payoff - P(lose) × stake$
**Consequence of failure:** Lose money

### 4. **Ad Bidding (NOT Required)**
**Requirement:** Just need relative ranking
**Why:** Doesn't matter if probability is 0.3 or 0.5, just that A > B
**Consequence:** None, can skip calibration

---

## Expected Calibration Error (ECE)

**Quantitative metric:**

```python
def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    
    ece = 0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_accuracy = y_true[mask].mean()
            bin_confidence = y_prob[mask].mean()
            bin_weight = mask.sum() / len(y_true)
            ece += bin_weight * abs(bin_accuracy - bin_confidence)
    
    return ece

# Before calibration: ECE = 0.23
# After calibration: ECE = 0.04 (much better)
```

**Interpretation:**
- ECE = 0: Perfect calibration
- ECE < 0.05: Production-ready
- ECE > 0.15: Do not deploy for probability-sensitive applications

---

## Production Checklist

Before deploying a model where probabilities matter:

1. ✓ Plot calibration curve on held-out test set
2. ✓ Calculate ECE
3. ✓ If ECE > 0.05, apply Platt or Isotonic calibration
4. ✓ Re-validate ECE on final test set
5. ✓ Document calibration method in model card
