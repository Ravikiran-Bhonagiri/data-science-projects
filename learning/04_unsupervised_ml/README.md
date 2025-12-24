<div align="center">

# 🔮 Module 4: Unsupervised Machine Learning

### *Discovering Hidden Patterns*

![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![Difficulty](https://img.shields.io/badge/Difficulty-Intermediate-yellow?style=flat-square)
![Guides](https://img.shields.io/badge/Guides-8-orange?style=flat-square)

**Find structure in unlabeled data through clustering and dimensionality reduction**

[📊 Clustering](#-clustering-algorithms) • [🗜️ Reduction](#-dimensionality-reduction) • [✅ Validation](#-validation)

</div>

---

## 💡 What is Unsupervised Learning?

> **"No labels, no problem. Let the data reveal its own structure."**

Unlike supervised learning, we don't have target labels (y). Instead, we discover:
- 🎯 **Clusters:** Natural groupings in data
- 🗜️ **Patterns:** Hidden structure
- 📉 **Compressed representations:** Reduce dimensions

---

## 📊 Clustering Algorithms

**Find natural groups in your data**

<table>
<tr>
<td width="50%">

### Centroid-Based

**[01. K-Means](./01_kmeans_clustering.md)**
- Simple, fast, scalable
- Requires K (number of clusters)
- Spherical clusters

**Best for:** Customer segmentation, image compression

</td>
<td width="50%">

### Density-Based

**[02. DBSCAN](./02_dbscan_density_clustering.md)**
- Arbitrary shapes
- Auto-detects cluster count
- Handles noise/outliers

**[03. HDBSCAN](./03_b_hdbscan_advanced.md)** ⭐
- Hierarchical DBSCAN
- Variable density
- More robust

**Best for:** Geospatial, anomaly detection

</td>
</tr>
<tr>
<td width="50%">

### Hierarchical

**[04. Hierarchical Clustering](./04_hierarchical_clustering.md)**
- Dendrogram visualization
- Agglomerative/divisive
- No K required upfront

**Best for:** Taxonomy, biology

</td>
<td width="50%">

### Validation

**[08. Cluster Validation](./08_cluster_validation_metrics.md)**
- Silhouette score
- Davies-Bouldin index
- Calinski-Harabasz

**Best for:** Choosing K, comparing methods

</td>
</tr>
</table>

---

## 🗜️ Dimensionality Reduction

**Compress high-dimensional data for visualization and analysis**

<table>
<tr>
<td width="33%">

### Linear Methods

**[05. PCA](./05_dimensionality_reduction.md)**
- Principal Component Analysis
- Preserves variance
- Fast, interpretable
- **Use:** Feature reduction

</td>
<td width="33%">

### Non-Linear Methods

**[06. t-SNE](./06_tsne_visualization.md)**
- Visualization specialist
- Preserves local structure
- Slow but beautiful
- **Use:** 2D/3D viz

</td>
<td width="33%">

### Modern Approaches

**[07. UMAP](./07_umap_advanced.md)**
- Faster than t-SNE
- Preserves global + local
- Production-ready
- **Use:** Best of both worlds

</td>
</tr>
</table>

---

## 🎯 Quick Decision Guide

| Your Goal | Best Algorithm | Why? |
|-----------|----------------|------|
| **Customer segmentation** | K-Means | Fast, interpretable clusters |
| **Anomaly detection** | DBSCAN / HDBSCAN | Identifies outliers naturally |
| **Visualize high-dim data** | t-SNE / UMAP | 2D projection preserves patterns |
| **Feature reduction** | PCA | Linear, fast, preserves variance |
| **Arbitrary cluster shapes** | DBSCAN | Not limited to spheres |
| **Don't know K** | DBSCAN / Hierarchical | Auto-determines clusters |

---

## 🛠️ Typical Workflow

```python
# 1. Standardize (Critical for K-Means, DBSCAN)
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)

# 2. Determine K (for K-Means)
from sklearn.metrics import silhouette_score
for k in range(2, 11):
    clusters = KMeans(n_clusters=k).fit_predict(X_scaled)
    score = silhouette_score(X_scaled, clusters)
    print(f"K={k}, Silhouette={score:.3f}")

# 3. Fit Best Model
model = KMeans(n_clusters=4)  # or HDBSCAN(), etc.
labels = model.fit_predict(X_scaled)

# 4. Visualize with Dimensionality Reduction
from sklearn.manifold import TSNE
X_2d = TSNE(n_components=2).fit_transform(X_scaled)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis')
```

---

## 💡 What You'll Master

<table>
<tr>
<td width="50%">

### 🎯 Clustering Skills
- ✅ K-Means algorithm
- ✅ DBSCAN for irregular shapes
- ✅ HDBSCAN for variable density
- ✅ Hierarchical clustering
- ✅ Cluster validation metrics

</td>
<td width="50%">

### 🗜️ Reduction Skills
- ✅ PCA fundamentals
- ✅ t-SNE visualization
- ✅ UMAP manifold learning
- ✅ Isomap geodesic distances
- ✅ Choosing right method

</td>
</tr>
</table>

---

## 📊 Real-World Applications

| Industry | Use Case | Technique |
|----------|----------|-----------|
| 🛒 **Retail** | Customer segmentation | K-Means |
| 🏥 **Healthcare** | Patient grouping | Hierarchical |
| 🌍 **Geospatial** | Location clustering | DBSCAN |
| 🔬 **Biology** | Gene expression | PCA + t-SNE |
| 🎯 **Marketing** | Market segments | K-Means + validation |

---

## 📚 Complete Guide List

**All guides in this module:**

1. **[K-Means Clustering](./01_kmeans_clustering.md)** - Master centroid-based clustering for customer segmentation and image compression
2. **[Hierarchical Clustering](./02_hierarchical_clustering.md)** - Build dendrograms and understand agglomerative clustering for taxonomy
3. **[DBSCAN & Isolation Forest](./03_dbscan_isolation_forest.md)** - Discover arbitrary-shaped clusters and detect outliers with density-based methods
4. **[HDBSCAN Advanced](./03_b_hdbscan_advanced.md)** - Handle variable-density clusters with hierarchical DBSCAN
5. **[PCA Dimensionality Reduction](./04_pca_dimensionality_reduction.md)** - Reduce dimensions while preserving variance using Principal Component Analysis
6. **[t-SNE & UMAP](./05_t_sne_umap.md)** - Create beautiful 2D visualizations of high-dimensional data
7. **[Gaussian Mixture Models](./06_gaussian_mixture_models.md)** - Probabilistic clustering with soft assignments and density estimation
8. **[Association Rules](./07_association_rules.md)** - Discover frequent itemsets and rules for market basket analysis
9. **[Cluster Validation Metrics](./08_cluster_validation_metrics.md)** - Evaluate clustering quality with silhouette scores and other metrics
10. **[Topic Modeling (LDA/NMF)](./09_topic_modeling_lda_nmf.md)** - Extract hidden topics from text using Latent Dirichlet Allocation and NMF
11. **[Neural Unsupervised (Autoencoders)](./10_neural_unsupervised_autoencoders.md)** - Learn compressed representations with neural network autoencoders

---

<div align="center">

**Master Unsupervised Learning** 🔮

*11 comprehensive guides • Clustering + Dimensionality Reduction*

[⬅️ Supervised ML](../03_supervised_ml/) • [🏠 Home](../../README.md) • [➡️ Model Evaluation](../05_evaluation/)

</div>
