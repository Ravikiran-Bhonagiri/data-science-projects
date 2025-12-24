# Handling Class Imbalance: The 1% Problem

**The Medical Diagnostic Failure:**

Hospital deploys AI to detect sepsis (life-threatening infection).

**The data:**
- 100,000 ICU patient records
- 800 sepsis cases (0.8%)
- 99,200 healthy patients

**Naive model:**
```python
model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy: 99.2%
# Sepsis detected: 0 cases
```

**What happened:** Model learned to predict "Not Sepsis" for everyone. Gets 99.2% accuracy doing nothing.

**Real consequence:** 800 patients at risk of death, model catches zero.

---

## Why Standard Models Fail

**The optimization problem:**

Models minimize overall error. With 99.2% healthy patients:
- Predicting "healthy" for everyone = 99.2% accuracy
- Actually detecting sepsis = More errors (false positives drag down accuracy)

**The model's logic:** "Why bother learning the minority class?"

---

## Solution 1: Class Weights (Penalize Errors Differently)

**Make minority class errors more costly:**

```python
from sklearn.linear_model import LogisticRegression

# Calculate weight ratio
n_healthy = 99200
n_sepsis = 800
weight_ratio = n_healthy / n_sepsis  # 124

# Apply class weights
model = LogisticRegression(class_weight={0: 1, 1: 124})
model.fit(X_train, y_rain)
```

**What this does:**
- Missing a sepsis case costs 124× more than a false alarm
- Forces model to pay attention to minority class

**Result:**
- Sepsis detection: 720/800 (90% recall)
- False alarms: 4,000 (acceptable in medical context)

**Production tip:** Use `class_weight='balanced'` for automatic calculation.

---

## Solution 2: SMOTE (Synthetic Minority Oversampling)

**The problem with simple oversampling:**
- Copying minority samples 100× → Model memorizes them
- Overfitting on the few samples you have

**SMOTE creates synthetic samples:**

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy=0.1)  # Bring minority to 10% of majority
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

model.fit(X_resampled, y_resampled)
```

**How it works:**
1. Find k-nearest neighbors of each minority sample
2. Draw a line to a random neighbor
3. Create new sample somewhere on that line

**Why it works:**
- New samples are **similar** but not identical
- Fills the decision boundary space
- Prevents memorization

**When to use:**
- Tabular data with many features
- Fraud detection (0.1% fraud rate)
- Rare disease detection

**When NOT to use:**
- Time series (creates impossible past data)
- Images (synthetic images often unrealistic)

---

## Solution 3: Undersampling (Reduce Majority)

**The idea:** Instead of creating more minority, delete majority.

```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(sampling_strategy=0.5)  # 1:2 minority:majority ratio
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
```

**Pros:**
- Fast training (less data)
- No synthetic data (keeps it real)

**Cons:**
- Throws away potentially useful data
- Only viable when you have LOTS of majority samples

**Production use case:**
- Click prediction (98% don't click)
- You have 10M samples, can afford to keep only 1M

---

## Solution 4: Threshold Tuning (Post-Processing)

**Don't change data. Change the decision boundary.**

```python
# Train normally
model.fit(X_train, y_train)

# Get probabilities
y_proba = model.predict_proba(X_test)[:, 1]

# Lower threshold to catch more minority cases
threshold = 0.15  # Instead of default 0.5
y_pred = (y_proba >= threshold).astype(int)
```

**When to use:**
- Can't retrain model (it's in production)
- Need different thresholds for different segments
- Business cost analysis determines optimal threshold

**Example:** Credit card fraud
- Transactions < $50: threshold = 0.7 (high precision, don't annoy customers)
- Transactions > $1000: threshold = 0.1 (high recall, catch expensive fraud)

---

## Solution 5: Ensemble with Resampling (Advanced)

**Combine multiple models trained on different balanced samples:**

```python
from imblearn.ensemble import BalancedRandomForestClassifier

model = BalancedRandomForestClassifier(
    n_estimators=100,
    sampling_strategy='not majority'  # Undersample majority for each tree
)
model.fit(X_train, y_train)
```

**How it works:**
- Each tree sees a different balanced subset
- Aggregate predictions across all trees
- Gets diversity without throwing away data permanently

**Production result:** Often best performance on heavily imbalanced data (e.g., 0.1% fraud).

---

## Decision Framework

| Imbalance Ratio | Data Size | Recommendation |
|----------------|-----------|----------------|
| **1:10** (10% minority) | Any | Class weights |
| **1:100** (1% minority) | Small (<100k) | SMOTE + Class weights |
| **1:100** (1% minority) | Large (>1M) | Undersampling + Ensemble |
| **1:1000** (0.1% minority) | Large | BalancedRandomForest |
| **Any** | Can't retrain | Threshold tuning |
