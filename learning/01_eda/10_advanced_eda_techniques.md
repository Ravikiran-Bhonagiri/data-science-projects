# 10. High-Dimensional EDA (PCA & t-SNE)

When you have 50, 100, or 1000 columns (e.g., Genes, Pixel data, Text vectors), standard scatter plots are impossible. You can't plot 100 dimensions.

**Solution:** Dimensionality Reduction. Crush the 100 dimensions down to 2 or 3 so we can see them.

---

## 1. PCA (Principal Component Analysis)
Linear technique. Finds the "directions" (Components) that maximize variance.

**Goal:** See global structure. Are there distinct clusters?
**Pros:** Fast, deterministic.
**Cons:** Can't capture complex non-linear manifolds (spirals).

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# 1. Scale Data (CRITICAL for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numerical)

# 2. Reduce to 2 Components
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)

# 3. Create Plotting DataFrame
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Target'] = df['Target']

# 4. Visualize
sns.scatterplot(x='PC1', y='PC2', hue='Target', data=pca_df, alpha=0.7)
plt.title('PCA: Data Projected to 2D')
```

---

## 2. t-SNE (t-Distributed Stochastic Neighbor Embedding)
Non-linear technique. Focuses on keeping similar neighbors close together.

**Goal:** Finding local clusters. Fantastic for image/text data.
**Pros:** Reveals clusters that PCA misses.
**Cons:** Slow, stochastic (different every time), axes have no meaning (distance doesn't mean much globally).

```python
from sklearn.manifold import TSNE

# Reduce to 2D
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_results = tsne.fit_transform(X_scaled)

tsne_df = pd.DataFrame(data=tsne_results, columns=['Dim1', 'Dim2'])
tsne_df['Target'] = df['Target']

sns.scatterplot(x='Dim1', y='Dim2', hue='Target', data=tsne_df)
plt.title('t-SNE Visualization')
```

---

## 3. Correlation Filtering
With 1000 columns, your correlation heatmap looks like static noise.

**Action:** Filter to only show high correlations.

```python
# Calculate correlation
corr = df.corr()

# Setup Mask to ignore diagonal/low correlations
threshold = 0.7
high_corr_pairs = corr[((corr > threshold) | (corr < -threshold)) & (corr != 1.0)]

# Drop all-NaN rows/cols to confirm we only have high-correlations
high_corr_pairs = high_corr_pairs.dropna(how='all', axis=0).dropna(how='all', axis=1)

# Now plot ONLY the highly correlated subset
sns.heatmap(high_corr_pairs, annot=True, center=0)
```

---

## Summary
*   **PCA:** Your first check. Fast look at variance.
*   **t-SNE/UMAP:** Deep dive into clustering structure.
*   **Filter Heatmaps:** Don't plot everything; plot what matters.
