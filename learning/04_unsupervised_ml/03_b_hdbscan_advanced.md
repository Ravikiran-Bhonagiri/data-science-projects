# HDBSCAN: Solving the "Variable Density" Problem

**The Geospatial Mapping Failure:**

You are mapping Uber pickups across a whole state.

**DBSCAN's fatal flaw:**
- It uses a **single** density threshold (`epsilon`).
- **Downtown (High Density):** Needs `eps=0.01` (tight clusters).
- **Rural Suburbs (Low Density):** Needs `eps=0.5` (loose clusters).
- **Result:** You can't pick one number. If you pick `0.01`, suburbs become noise. If you pick `0.5`, downtown becomes one giant useless blob.

**HDBSCAN Solution:**
It builds a hierarchy of clusters and automatically selects the most stable ones at *varying densities*.
- Finds tight clusters in cities.
- Finds loose clusters in suburbs.
- **Simultaneously.**

**Why it's the new standard:** it removes the most difficult parameter (`epsilon`) entirely.

---

## 1. Implementation (Use the `hdbscan` or `sklearn` library)
*Note: Added to Sklearn in v1.3+*

```python
from sklearn.cluster import HDBSCAN
from sklearn.datasets import make_moons

X, _ = make_moons(n_samples=500, noise=0.1)

# Only one real parameter: "Use at least this many points to form a cluster"
hdb = HDBSCAN(min_cluster_size=15)
labels = hdb.fit_predict(X)

# Plot
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
plt.title(f"HDBSCAN found {len(set(labels)) - 1} clusters")
plt.show()

# -1 Labels are Noise
print(f"Noise points detected: {sum(labels == -1)}")
```

## 2. Soft Clustering with HDBSCAN
HDBSCAN can also provide probabilities (like GMM)!

```python
# How strong is the membership? (0 = weak/noise, 1 = strong/core)
prob_scores = hdb.probabilities_
```
