# Cross-Validation: The Data Leakage Disaster

**The Medical AI Scandal:**

A prestigious research team publishes: "Our AI detects COVID-19 from chest X-rays with 99.7% accuracy!"

**Their validation:**
```python
# They did train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)  # 99.7%!
```

**The problem they missed:** The same patient appeared in both train AND test sets (multiple X-rays per patient on different days).

**What the model actually learned:**
- "Patient #4,582 has a cracked rib in the upper-left. If I see that crack, it's COVID."
- It memorized individual patients, not COVID patterns.

**Real-world deployment:** 67% accuracy. $40M lawsuit. Retracted paper.

---

## The Right Way: Prevent Patient Leakage

```python
from sklearn.model_selection import GroupKFold

# Ensure all X-rays from same patient stay together
gkf = GroupKFold(n_splits=5)
for train_idx, test_idx in gkf.split(X, y, groups=patient_ids):
    # Now patient #4,582's X-rays are ALL in train or ALL in test
    # Never split across folds
```

---

## Real Production Scenarios

### 1. E-Commerce Recommendations: User Leakage
**Setup:** Predict product purchases. You have 5 purchases per user.

**Wrong way (Standard K-Fold):**
- User A's purchase history: 5 transactions
- 4 go to train, 1 goes to test
- Model learns: "User A likes electronics" from the 4 transactions
- Predicts the 5th easily ✓
- **Test accuracy: 94%**

**Deployment:** New users (no history). **Accuracy: 51%** (random guessing)

**Right way:** `GroupKFold` by `user_id`

### 2. Time Series: Future Peeking
**Setup:** Stock price prediction

**Wrong way (Shuffle=True):**
```python
KFold(n_splits=5, shuffle=True)  # Disaster!
```
- You train on "March 15" and test on "March 1"
- **You're using the future to predict the past**
- Test score: 92% (You're cheating)
- Live trading: -$250k in 2 weeks

**Right way:**
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
# Fold 1: Train [Jan], Test [Feb]
# Fold 2: Train [Jan-Feb], Test [Mar]
# Never test on past data
```

---

## The Decision Tree

### C. Group K-Fold (The Leakage Preventer) 🚨
*   **Scenario:** You have 5 X-Rays per patient.
*   **The Trap:** If Patient A's X-Ray 1 is in Train, and X-Ray 2 is in Test, the model memorizes Patient A's bones.
*   **The Fix:** **GroupKFold**. It ensures all X-Rays from Patient A are in the SAME fold.
*   **Use When:** Multiple rows belong to the same entity (User, Session, Device).

### D. Time Series Split (The Time Traveler) ⏳
*   **Scenario:** Predicting stock prices.
*   **The Trap:** Regular CV might train on "2022" and test on "2020". You are predicting the past using the future.
*   **The Fix:** **TimeSeriesSplit**.
    *   Fold 1: Train [Jan], Test [Feb]
    *   Fold 2: Train [Jan, Feb], Test [Mar]
    *   Fold 3: Train [Jan, Feb, Mar], Test [Apr]

---

## 2. Implementation

```python
from sklearn.model_selection import StratifiedKFold, GroupKFold, TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# 1. Stratified (Standard Classification)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
print(f"Mean AUC: {scores.mean():.3f}")

# 2. Group K-Fold (Medical/User data)
gkf = GroupKFold(n_splits=5)
# Note: You must pass 'groups' (e.g., patient_ids)
scores = cross_val_score(model, X, y, cv=gkf, groups=patient_ids, scoring='accuracy')

# 3. Time Series
tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    # Train / Validate manually...
```

---

## 3. Nested CV (The Expert Move)

If you use CV to tune hyperparameters (GridSearch), and THEN report that score... you are cheating again. You optimized for that specific CV split.

**Nested CV** is Inception.
*   **Outer Loop (5 folds):** Estimates Model Error.
*   **Inner Loop (3 folds):** Tunes Hyperparameters.

It is computationally expensive ($5 \times 3 = 15$ runs), but it is the **unbiased truth** of how good your AutoML pipeline is.
