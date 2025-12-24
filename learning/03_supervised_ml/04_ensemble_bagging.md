# Random Forests: The Kaggle Winning Strategy

**The Netflix Prize (2006-2009):**

$1M prize for improving recommendation accuracy by 10%.

**Single model approaches failed:**
- Linear regression: 8.5% improvement
- Single decision tree: 7.2% improvement
- Neural net (2007 hardware): 9.1% improvement

**Winning solution: "BellKor's Pragmatic Chaos"**
- Ensemble of 107 different models
- **Core:** Random Forest variants
- Final accuracy: 10.06% improvement
- **Prize won:** $1,000,000

**Why ensembles dominated:**
- Single tree overfits to noise ("User #482 likes action")
- 500 trees voting: "Most users with profile X like action" (robust pattern)
- Each tree sees different data/features → uncorrelated errors cancel out

---

## Production Reality: Credit Scoring

**Capital One fraud detection:**
- Input: 1,000 transaction features
- Challenge: Real-time (<50ms response)
- Why not neural nets? Too slow for latency requirements

**Random Forest approach:**
- 500 trees, each trained on bootstrapped sample
- Parallel inference (500 trees evaluated simultaneously)
- Response time: 23ms average
- F1-score: 0.94 (vs 0.89 for logistic regression)

**Business value:** Stopped $240M in fraud annually while maintaining <0.1% false positive rate

---

## 1. Why Random Forests Work (The Math)

According to the Law of Large Numbers, averaging uncorrelated predictions reduces error.

$$Variance_{Total} = \rho \sigma^2 + \frac{1 - \rho}{n} \sigma^2$$
- $n$: Number of trees
- $\sigma^2$: Variance of a single tree
- $\rho$ (rho): Correlation between trees

**The Goal:**
1.  **Lower $\rho$:** We want trees to be as *uncorrelated* as possible. We do this by forcing them to use random subsets of data/features.
2.  **Increase $n$:** Adding more trees generally improves performance (up to a plateau).

---

## 2. Bootstrapping & Out-of-Bag (OOB) Error

### Random Selection with Replacement
When we sample $N$ rows from a dataset of size $N$ *with replacement*:
- Some rows are picked multiple times.
- Some rows are never picked.

**The 63.2% Rule:**
The probability of *not* picking a specific row in $N$ draws is $(1 - 1/N)^N \approx 1/e \approx 0.368$.
- **63.2%** of data makes it into the training bag.
- **36.8%** is left out (OOB).

### OOB as Validation
Since the model never saw the OOB samples during training, we can use them to evaluate the model **without a separate test set!**

```python
rf = RandomForestClassifier(n_estimators=100, oob_score=True)
rf.fit(X, y)
# This is a roughly unbiased estimate of test error
print(f"OOB Accuracy: {rf.oob_score_}")
```

---

## 3. Tuning Hyperparameters

Random Forests are robust, but tuning squeezes out the last 2-3% of performance.

| Parameter | Default | Tuning Strategy |
|-----------|---------|-----------------|
| `n_estimators`| 100 | Increase until performance stabilizes (e.g., 500). No risk of overfit, just slower code. |
| `max_features`| `sqrt` | **Most Critical.** Lowering this (e.g. to 0.5*sqrt) reduces correlation ($\rho$) between trees. |
| `max_depth` | None | Trees usually don't need pruning in RF, but limiting depth (10-20) speeds up training. |
| `min_samples_leaf` | 1 | Increase to 5-10 to reduce model size and noise sensitivity. |
| `class_weight`| None | Set to `'balanced'` if you have imbalanced classes (Rare event detection). |

---

## 4. Advanced: Feature Importance Pitfalls

### Method A: Impurity-Based (Default)
Calculates how much a feature reduces Gini impurity across all trees.
- **Trap:** Heavily biased towards **high-cardinality** features (e.g., continuous variables or categorical with many levels). The model can split on them many times.

### Method B: Permutation Importance (Gold Standard)
1. Predict on validation set → Measure Score (e.g., 0.85).
2. **Shuffle** Column A randomly (breaking relationship with Target).
3. Predict again → Measure Score (e.g., 0.80).
4. **Importance** = Original - Shuffled (0.05).
5. If score doesn't drop, the feature was useless.

```python
from sklearn.inspection import permutation_importance

# Run permutation importance
result = permutation_importance(
    rf, X_test, y_test, 
    n_repeats=10, 
    random_state=42, 
    n_jobs=-1
)

sorted_idx = result.importances_mean.argsort()

plt.figure(figsize=(10, 6))
plt.boxplot(
    result.importances[sorted_idx].T, 
    vert=False, 
    labels=X_test.columns[sorted_idx]
)
plt.title("Permutation Importance (Test Set)")
plt.show()
```

---

## 5. Pros & Cons Checklist

### ✅ Pros
- **Robust:** Works on almost any tabular dataset with minimal tuning.
- **Non-Linear:** Handles complex interactions ($X1 > 5$ AND $X2 < 3$).
- **No Scaling:** Distance doesn't matter, only order ($X > 5$).
- **Parallelizable:** Trees are independent (fast training).

### ❌ Cons
- **Slow Inference:** Must run 500 trees for one prediction. Bad for real-time APIs.
- **Black Box:** Harder to explain than a single decision tree.
- **Extrapolation:** Cannot predict values outside training range (like all trees).
- **Size:** Model files can be huge (GBs).
