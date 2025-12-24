# High Cardinality: The Million-Category Problem

**E-Commerce Product Recommendations:**
- Products: 2.4 million SKUs
- Task: Predict purchase probability

**The Failure Modes:**

**Attempt 1: One-Hot Encoding**
- Creates 2.4M columns
- Memory requirement: 180GB for training set alone
- Result: **OOM error**, can't even load data

**Attempt 2: Label Encoding**
```python
df['product_id'] = df['product_id'].astype('category').cat.codes
```
- Model learns: Product #1,582,093 > Product #12
- Result: Random predictions, no learning

**Attempt 3: Drop the feature**
- Lose all product-specific information
- Can only use price/category (coarse)
- Result: Mediocre recommendations, revenue impact -$2.1M/quarter

---

## The Solutions That Actually Work

### 1. Target Encoding (The Quick Win)
Replace each SKU with its **historical conversion rate**

```python
# Encode with smoothing to handle rare products
from category_encoders import TargetEncoder
encoder = TargetEncoder(cols=['product_id'], smoothing=50)
df['product_score'] = encoder.fit_transform(df['product_id'], df['purchased'])
```

**Result:** 1 column, captures product quality, works immediately

---

## 1. Frequency Encoding (The Simple Fix)

Replace the category with **"How common it is"**.
*   Paris (10,000 rows): `0.25` (25% of data).
*   TinyTown (5 rows): `0.0001`.

**Logic:** If being from a big city matters (e.g., fast internet), this captures it.
**Flaw:** Paris and London might both be 0.25. They look identical to the model (Collision).

---

## 2. The Hashing Trick (Fixed Width)

Instead of a dictionary, use a **Hash Function**.
Map the 1 million IDs into a fixed bucket size (e.g., 1000 columns).
`Hash("New York") % 1000 -> Column 42`.
`Hash("Tokyo") % 1000 -> Column 860`.

*   **Pro:** Fixed memory usage. Very fast.
*   **Con:** **collisions**. "New York" and "MyPony" might both land in Column 42. The model gets confused. (Usually acceptable if buckets are large enough).

```python
from sklearn.feature_extraction import FeatureHasher

# 10 input columns only, no matter how many cities exist
h = FeatureHasher(n_features=10, input_type='string')
f = h.transform(df['City'])
```

---

## 3. Entity Embeddings (The Neural Way)

This is how ChatGPT handles words (Vocabulary = 50,000).
It maps every word to a **Vector of size 300** (e.g., `[0.1, -0.5, 0.8, ...]`).

You can do this for Zip Codes or Store IDs.
1.  Input Layer: One-Hot (Virtual).
2.  Embedding Layer: Compresses into Dense Vector (e.g., size 10).
3.  Output: Prediction.

**Result:** The Neural Net *learns* Similarities.
*   Zip Code 90210 and 10001 land close together in vector space (Rich areas).
*   It discovers geography/demographics automatically.

**Use Keras/PyTorch for this.** It is the strongest method for High Cardinality.
