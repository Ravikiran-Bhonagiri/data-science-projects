# Encoding Strategies: Beyond One-Hot

**The Problem:** You're building a recommendation system for Netflix. You have a `Genre` column with values like "Sci-Fi Thriller", "Korean Drama", "Stand-Up Comedy". 273 unique genres.

**Attempt 1: One-Hot Encoding**
- Result: 273 sparse binary columns
- Model size: 15GB → 180GB
- Training time: 4 hours → 38 hours
- **Production impact:** Infrastructure costs $12k/month → $95k/month

**Attempt 2: Label Encoding**
- `Sci-Fi Thriller = 1`, `Korean Drama = 2`, `Stand-Up Comedy = 3`
- **Fatal flaw:** Model learns "Comedy = 3× more than Sci-Fi". Mathematically nonsense.
- Deployment ROC-AUC: 0.52 (basically random)

**The Real Solution: Target Encoding**
- Replace each genre with **average watch time** for that genre
- `Sci-Fi Thriller → 47 minutes`, `Korean Drama → 89 minutes`, `Stand-Up → 22 minutes`
- **Result:** 1 column. Model learns actual user behavior. AUC improves to 0.78.

---

## The Production Playbook

### When One-Hot Works
**Use case:** Payment method (Credit/Debit/PayPal) - 3-5 categories max
```python
pd.get_dummies(df['payment_method'], prefix='pay')
```

### When Label Encoding Works
**Use case:** T-shirt sizes (XS < S < M < L < XL) - ordinal relationship is real
```python
size_map = {'XS': 1, 'S': 2, 'M': 3, 'L': 4, 'XL': 5}
df['size_encoded'] = df['size'].map(size_map)
```

### When Target Encoding Wins
**Use case:** High cardinality + tree models (City, Product SKU, User Agent)

---

## 2. Advanced: Target Encoding (The Grandmaster Move)

**Idea:** Instead of replacing "Paris" with a random number, replace it with **how likely people in Paris are to Churn.**

$$Value(Paris) = \text{Mean of Target for Paris}$$

*   If 20% of Parisians churn, `Paris` becomes `0.20`.
*   If 5% of Londoners churn, `London` becomes `0.05`.

**Result:** You collapsed 10,000 cities into **1 extremely predictive number**.
Tree models LOVE this.

### ⚠️ The Danger: Data Leakage
If you calculate the mean using the *entire* dataset, you are using the Target (Answer) to generate the Feature (Question).
**You are cheating.** The model will overfit massively.

**The Fix:**
1.  **Leave-One-Out:** Calculate mean of everyone *else*.
2.  **Smoothing:** Weigh the category mean against the global mean (for rare cities).
3.  **Noise:** Add random noise to prevents memorization.

```python
# The Safe Way: Category Encoders library
# pip install category_encoders

import category_encoders as ce

# Target Encoding with Smoothing
encoder = ce.TargetEncoder(cols=['City'], smoothing=10)
df['City_Encoded'] = encoder.fit_transform(df['City'], df['Churn'])
```

---

## 3. CatBoost Encoding (Ordered Target)

CatBoost (the algorithm) invented a genius way to fix Target Leakage.
It simulates "Time".
*   For Row 5, it calculates the mean of rows 1-4.
*   For Row 6, it calculates the mean of rows 1-5.
It never peeks into the future.

```python
encoder = ce.CatBoostEncoder(cols=['City'])
df['City_Encoded'] = encoder.fit_transform(df['City'], df['Churn'])
```

---

## 4. Weight of Evidence (WoE)

Used heavily in Credit Risk / Banking.
It measures specific "Lift" of a category.

$$WoE = \ln \left( \frac{\% \text{ of Non-Events}}{\% \text{ of Events}} \right)$$

*   **WoE > 0:** This category is "Good" (Safe).
*   **WoE < 0:** This category is "Bad" (Risky).
*   **WoE = 0:** No signal.

**Superpower:** It handles outliers and missing values gracefully, and scales standard logistic regression perfectly.
