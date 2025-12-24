# Hierarchical Clustering: Tracking COVID Variants

**The WHO Variant Classification Problem (2020-2021):**

Scientists discovered new COVID mutations daily. How to organize them?

**K-Means approach failed:**
- "Pick K=5 variants" → But how many variants exist? Unknown.
- Forced arbitrary splits between closely related strains
- Couldn't show evolutionary relationships

**Hierarchical clustering solution:**
```python
# Genetic sequences of 10,000 COVID samples
dendrogram = hierarchy.linkage(genetic_distance_matrix, method='ward')
```

**The result: A phylogenetic tree**
```
Root
├── Original Wuhan strain
├── Alpha variant (UK)
│   └── Alpha.1 sublineage
├── Delta
│   ├── Delta.1
│   └── Delta.2 (more contagious)
└── Omicron
    ├── BA.1
    ├── BA.2
    └── BA.5 (immune escape)
```

**Why hierarchical:**
- Shows WHO descended from WHAT
- Can "cut" tree at any level (major variants vs sub-lineages)
- Informs vaccine development (which variants to target)

---

## Production Use Cases

### 1. **Customer Segmentation (E-Commerce)**
**Problem:** Hard to pick "K=5 customer types"

**Hierarchical approach:**
- Level 1: [Active, Inactive]
- Level 2: [High-value Active, Low-value Active, Recently Churned, Long-gone]
- Level 3: Further splits based on behavior

**Business value:** Marketing targets different levels based on campaign budget

### 2. **Document Clustering (News)**
Tree structure for topic hierarchy:
- Politics → US → Elections → Swing States
- Sports → NFL → Week 12 → Thursday Night Game

---

## 1. Agglomerative vs Divisive

1.  **Agglomerative (Bottom-Up):**
    *   Start: Everyone is their own cluster ($N$ clusters).
    *   Step: Find the 2 closest people and marry them. Now $N-1$ clusters.
    *   Repeat until everyone is in 1 giant cluster.
    *   **This is the standard approach.**

2.  **Divisive (Top-Down):**
    *   Start: Everyone is in 1 cluster.
    *   Step: Split the group in the most logical place.
    *   Computationally heavy. Rarely used.

---

## 2. Linkage: How do we measure "Closeness"?
When merging Cluster A and Cluster B, how do we measure distance?

| Method | Definition | Effect |
|--------|------------|--------|
| **Single** | Min distance between closest points | Creates long, stringy chains. Handles "banana" shapes. |
| **Complete** | Max distance between farthest points | Creates tight, spherical clumps. |
| **Average** | Average of all pairs | Balanced compromise. |
| **Ward** | Minimize Variance increase | **Default.** Most similar to K-Means logic. |

---

## 3. Implementation: The Dendrogram

```python
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=200, centers=3, random_state=42)

# 1. The Dendrogram (Visualization)
plt.figure(figsize=(12, 6))
plt.title('The Dendrogram')
# 'ward' minimizes variance within clusters
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.axhline(y=10, color='r', linestyle='--', label='Cut Threshold')
plt.show()
```

-   The **Y-axis** is distance. The taller the vertical line, the more different the two merged groups were.
-   We usually "cut" the tree (draw a horizontal line) where the vertical lines are longest (biggest jump in difference).

### The Sklearn Model
Once you pick your $K$ from the Dendrogram, run the actual model.

```python
# 2. The Model (Note: No 'predict' method, only 'fit_predict')
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
y_hc = hc.fit_predict(X)

plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=50, c='red')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=50, c='blue')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=50, c='green')
plt.title('Hierarchical Clusters')
plt.show()
```

---

## 4. Pros & Cons

### ✅ Pros
-   **No need to pick K:** The specific $K$ is a choice you make *after* seeing the tree.
-   **Interpretability:** The tree structure is extremely informative for taxonomy.

### ❌ Cons
-   **Performance:** $O(N^2)$ or $O(N^3)$.
    -   N=1,000: Fast.
    -   N=10,000: Slow.
    -   N=100,000: **Impossible.** Your RAM will crash.
    -   *Use K-Means for big data.*
