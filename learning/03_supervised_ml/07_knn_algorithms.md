# KNN: When Simplicity Beats Complexity

**The Netflix Recommendation Failure (Early Days):**

Netflix tried sophisticated matrix factorization for recommendations.

**Problem:** Cold start - new users with <5 ratings
- Complex model: 23% accuracy
- Users frustrated with random suggestions

**KNN collaborative filtering solution:**
```python
# Find K=20 users most similar to new user
knn = NearestNeighbors(n_neighbors=20, metric='cosine')
knn.fit(user_movie_matrix)
similar_users = knn.kneighbors(new_user_vector)
# Recommend what similar users watched
```

**Results:**
- Cold start accuracy: 67% (vs 23%)
- Simple, interpretable: "Users like you watched X"
- Fast enough for real-time (<100ms)

**Why it worked:** Similarity in sparse ratings more reliable than complex patterns with little data

---

## Real-World Failures

**Credit card fraud (naive approach):**
- Applied KNN to raw transaction data
- Features: amount ($10-$10,000), merchant_id (1-50,000)
- **Problem:** Distance dominated by merchant_id (much larger numbers)
- Result: 54% accuracy (worse than random)

**Fix:** StandardScaler
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
knn.fit(X_scaled, y)  # Now: 82% accuracy
```

---

## 1. How It Works
To classify a new point:
1.  Calculate distance to all training points.
2.  Find the **K** nearest neighbors.
3.  **Vote:** Majority class wins.

---

## 2. Choosing K
- **Low K (e.g., 1):** Highly sensitive to noise. Jagged boundaries. Overfitting.
- **High K (e.g., 100):** Smoothed out. Can miss local patterns. Underfitting.

**Rule of Thumb:** $K = \sqrt{N}$ is a good starting point. Odd numbers prevent ties (3, 5, 7).

---

## 3. Distance Metrics

1.  **Euclidean:** Straight line ($L_2$ norm). Standard.
    $$d = \sqrt{(x_1-x_2)^2 + (y_1-y_2)^2}$$
2.  **Manhattan:** Grid/City block ($L_1$ norm). Better for high dimensions.
    $$d = |x_1-x_2| + |y_1-y_2|$$
3.  **Cosine:** Angle between vectors. Best for **Text/NLP**.

---

## 4. Implementation

> [!IMPORTANT]
> **KNN REQUIRES SCALING!** Like SVM, it relies entirely on distance calculations.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

knn = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2) # p=2 is Euclidean
)

knn.fit(X_train, y_train)
```

---

## 5. Curse of Dimensionality
KNN fails when you have many features (e.g., > 20).
In high dimensions, "everything is far away from everything else." Distances become meaningless.

**Solution:** Use PCA (Dimensionality Reduction) before KNN.

---

## 6. Efficient Search (KD-Tree / Ball-Tree)
Calculating distance to *every* point is slow ($O(N)$).
Scikit-learn automatically uses data structures like **KD-Tree** or **Ball-Tree** to speed up search to $O(\log N)$.

```python
# Force a specific algorithm if needed
knn = KNeighborsClassifier(algorithm='ball_tree')
```
