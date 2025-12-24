# Module 4: Unsupervised Machine Learning

Welcome to the **"Wild West"** of Data Science.
In Supervised Learning, you have a teacher (labels). You know the answer.
In **Unsupervised Learning**, you have no teacher. You are an explorer dropped into a jungle of data, trying to find maps, tribes, and structures on your own.

---

## 📚 The Grandmaster Series

### Phase 1: Clustering (The Organizers)
*How to group similar things together.*
1.  **[K-Means Clustering](./01_kmeans_clustering.md):** The "Divider". Fast, simple, forces everyone into a circle.
2.  **[Hierarchical Clustering](./02_hierarchical_clustering.md):** The "Taxonomist". Builds family trees.
3.  **[DBSCAN & Isolation Forest](./03_dbscan_isolation_forest.md):** The "Anomaly Hunters".
4.  **[HDBSCAN (Advanced)](./03_b_hdbscan_advanced.md):** The "Modern Standard". Best features of Hierarchical + DBSCAN.
5.  **[Gaussian Mixtures (GMM)](./06_gaussian_mixture_models.md):** The "Soft" Organizer. Probabilistic clustering.

### Phase 2: Dimensionality Reduction (The Compressors)
*How to simplify 100 features down to 3.*
6.  **[PCA (Principal Component Analysis)](./04_pca_dimensionality_reduction.md):** The "Zipper". Compresses data linearly.
7.  **[t-SNE & UMAP](./05_t_sne_umap.md):** The "Map Makers". Unrolls 3D shapes into 2D maps.
8.  **[Autoencoders (Neural)](./10_neural_unsupervised_autoencoders.md):** The "Deep Compressor". Non-linear compression & anomaly detection.

### Phase 3: Patterns & Validation (Expert Level)
9.  **[Association Rules (Market Basket)](./07_association_rules.md):** The "Recommender". Finding "Beer & Diapers" rules.
10. **[Topic Modeling (LDA/NMF)](./09_topic_modeling_lda_nmf.md):** The "Librarian". Finding hidden topics in text.
11. **[Cluster Validation](./08_cluster_validation_metrics.md):** The "Truth Serum". Silhouette, Davies-Bouldin, and Stability.

---

## 🧠 Decision Framework: Which Algorithm?

| Goal | Data Shape | Size | Recommendation |
|------|------------|------|----------------|
| **Customer Segmentation** | Globular / Unknown | Small (<10k) | **Hierarchical** (to see the tree) |
| **Customer Segmentation** | Globular | Large (>10k) | **K-Means** (Fast) |
| **Fraud / Outliers** | Messy | Any | **Isolation Forest** |
| **Spatial Clustering** | Roads / Rivers | Any | **DBSCAN** |
| **Reduce Features for ML** | Linear | Any | **PCA** (Preserves Variance) |
| **Make a Plot for Boss** | Complex | Any | **UMAP** (Preserves Clusters) |

---

## 🛠️ Typical Exploratory Workflow

```python
# 1. Scale Data (CRITICAL for Unsupervised!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Visualize with UMAP (Get a feel for the shape)
reducer = umap.UMAP()
embedding = reducer.fit_transform(X_scaled)
plt.scatter(embedding[:,0], embedding[:,1])

# 3. Cluster with K-Means (Business Logic)
kmeans = KMeans(n_clusters=4)
labels = kmeans.fit_predict(X_scaled)

# 4. Describe Clusters
df['cluster'] = labels
print(df.groupby('cluster').mean())
```
