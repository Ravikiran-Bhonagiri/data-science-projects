# 7. Handling Class Imbalance

## The Problem
Class Imbalance occurs when one class dominates the other.
*   **Examples:** Fraud Detection (99.9% Legit, 0.1% Fraud), Disease Diagnosis, Spam Filtering.
*   **The Trap:** A model predicting "Majority Class" for everyone achieves 99.9% accuracy but fails completely at its actual job (finding the minority case).

---

## 1. The Metrics Trap (Don't use Accuracy!)
When classes are imbalanced, **Accuracy is useless**.

### Use These Instead:
*   **Precision:** Of all predicted positives, how many were real? (Avoid False Alarms)
*   **Recall (Sensitivity):** Of all real positives, how many did we find? (Don't miss Fraud)
*   **F1-Score:** Harmonic mean of Precision and Recall.
*   **ROC-AUC:** Measures how well the model separates classes, regardless of the threshold.

---

## 2. Resampling Techniques (Data Level)

We can artificially balance the dataset before training.

### A. Undersampling (Reduce Majority)
Randomly remove samples from the majority class.
*   **Pros:** Fast, reduces data size.
*   **Cons:** Throwing away potentially useful data.
*   **When to use:** You have a HUGE dataset (millions of rows).

```python
from sklearn.utils import resample

# Separate majority and minority classes
df_majority = df[df.balance == 0]
df_minority = df[df.balance == 1]

# Downsample majority class
df_majority_downsampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=len(df_minority), # match minority n
                                 random_state=42) 
```

### B. Oversampling (Duplicate Minority)
Randomly duplicate samples from the minority class.
*   **Pros:** No information loss.
*   **Cons:** Causes overfitting (model memorizes specific minority examples).

### C. SMOTE (Synthetic Minority Over-sampling Technique) - **Recommended**
Instead of blindly duplicating, SMOTE creates **new, synthetic** examples.
1.  It picks a minority point.
2.  It finds its "K-nearest neighbors" (similar minority points).
3.  It creates a new point on the line between them.

*   **Result:** The minority class "region" is filled out, helping the model generalize better.

```python
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Apply SMOTE ONLY to Training data (Never touch Test data!)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Original shape: {y_train.value_counts()}")
print(f"Resampled shape: {y_train_resampled.value_counts()}")
```

---

## 3. Algorithmic Techniques (Model Level)

Instead of changing the data, tell the model to pay more attention to the minority.

### Class Weights
Most Scikit-Learn models have a `class_weight='balanced'` parameter. This penalizes the model heavily for making mistakes on the minority class.

```python
from sklearn.ensemble import RandomForestClassifier

# The model will automatically weight the minority class higher
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)
```
*   **Pros:** No extra computation or data manipulation needed.
*   **Cons:** Sometimes less effective than SMOTE for extremely rare classes.

---

## 4. Best Practice Strategy

1.  **Always Split First:** Do Train/Test split **before** any resampling.
2.  **Try Class Weights First:** It's the simplest "free" fix.
3.  **Try SMOTE:** If class weights aren't enough, use SMOTE on the training set.
4.  **Evaluate Correctly:** Look at the Confusion Matrix and AUC, not Accuracy.

### Checklist
- [ ] Did I split Train/Test before resampling? (CRITICAL)
- [ ] Did I use Precision/Recall/AUC?
- [ ] Is my baseline model better than "Dummbly predicting majority"?
