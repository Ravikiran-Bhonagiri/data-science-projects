# t-SNE & UMAP: The "Map Makers"

PCA is linear. It assumes data lies on flat sheets.
But data is often a crumpled ball of paper, or a Swiss Roll.
If you squash a Swiss Roll flat (PCA), you crush the layers together.
**Manifold Learning** (t-SNE, UMAP) attempts to "unroll" the paper carefully to preserve the local structure.

### 🧠 When should you actually use this?

**1. The Genomics "Atlas":**
You have 50,000 cells. You want to see "clusters" of cell types on a 2D poster.
*   **The Tool:** UMAP.
*   **Result:** Beautiful islands of cells. "These are T-Cells, these are B-Cells."

**2. The Word Embedding Visualizer:**
You trained a Word2Vec model. "King", "Queen", "Man", "Woman" are vectors in 300 dimensions.
*   **The Tool:** t-SNE.
*   **Result:** You see "King" and "Queen" floating right next to each other.

---

## 1. t-SNE (t-Distributed Stochastic Neighbor Embedding)

**The Vibe:** "Preserve Local Neighbors at all costs."
It looks at a point and says: "Who are your closest friends? Okay, I will make sure they stay close to you in 2D. I don't care about the Global Structure (far away points)."

**Major Gotchas:**
1.  **Stochastic:** It changes every time you run it. (Set `random_state`).
2.  **Perplexity:** The knob you turn. Usually 5-50. It effectively guesses "number of neighbors."
3.  **Global Structure:** Distances between far-away clusters *mean nothing*. You cannot say "Cluster A is close to Cluster B" reliably.

```python
from sklearn.manifold import TSNE
import seaborn as sns

tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=y)
plt.title("t-SNE Projection")
plt.show()
```

---

## 2. UMAP (Uniform Manifold Approximation and Projection)

**The Modern King.**
t-SNE is slow (bad for large data) and ignores global structure.
UMAP is **Fast**, scales to millions of points, and (often) preserves global structure better.

> [!NOTE]
> UMAP is a separate library: `pip install umap-learn`

```python
import umap
# pip install umap-learn

reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

sns.scatterplot(x=X_umap[:,0], y=X_umap[:,1], hue=y)
plt.title("UMAP Projection")
plt.show()
```

**Key Parameters:**
-   `n_neighbors`: Size of local neighborhood. Large (e.g., 50) = Big Picture. Small (e.g., 5) = Local details.
-   `min_dist`: How tightly to pack points together. Low (0.1) = Clumpy. High (0.99) = Spread out.

---

## 3. Comparison: PCA vs t-SNE vs UMAP

| Method | Type | Interpretation | Scalability | Use Case |
|--------|------|----------------|-------------|----------|
| **PCA** | Linear | Excellent (Loadings) | Fast | Feature reduction for models. |
| **t-SNE** | Non-Linear (Probabilistic) | **None** (Only clusters visual) | Slow ($O(N^2)$) | Visualization of small/med data. |
| **UMAP** | Non-Linear (Topology) | Good (Better than t-SNE) | **Very Fast** | Visualization of Big Data. |

> [!WARNING]
> **NEVER use t-SNE/UMAP coordinates as features for a classifier.**
> They distort density and distance too much. Use them for *Exploration* and *Plotting*. Use PCA for feature engineering.
