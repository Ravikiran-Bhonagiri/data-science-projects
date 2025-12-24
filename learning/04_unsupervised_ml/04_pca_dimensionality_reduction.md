# PCA: When 20,000 Features Become 50

**The Cancer Genomics Problem:**

You're analyzing tumor samples to predict treatment response.

**The data:**
- 500 patients
- 20,000 gene expression measurements per patient
- Target: Will chemotherapy work?

**Naive approach: Use all 20,000 genes**
```python
model = LogisticRegression()
model.fit(X_train, y_train)  # X has 20,000 columns
# Result: 99% training accuracy, 52% test accuracy
```

**What happened:** Massive overfitting. The model learned "Gene #14,582 predicts response" but it's random noise.

**PCA solution:**
```python
pca = PCA(n_components=50)  # Compress 20,000 → 50
X_pca = pca.fit_transform(X_scaled)
model.fit(X_pca_train, y_train)
# Result: 76% training, 74% test (generalizes!)
```

**Why it works:**
- PCA finds the 50 "meta-genes" (combinations of real genes) with most variation
- Discards 19,950 dimensions of noise
- **Explained variance:** 50 components capture 89% of information

**Clinical impact:** Model now deployed in 47 hospitals for treatment planning.

---

## Production Scenarios

### 1. **Recommendation Systems (Spotify)**
- Raw: 50M songs × features
- PCA: 300 "taste profiles"
- Speed: 12ms → 0.8ms per recommendation

### 2. **Facial Recognition**
- Raw: 1024×1024 pixels = 1M features
- PCA: 150 "eigenfaces"
- Accuracy maintained, inference 100× faster

---

## 1. How It Works (The Geometric Rotation)

PCA doesn't just delete columns. It combines them.
It rotates the graph axes to find the "Widest Scatter."

1.  **Center the data** (Mean = 0).
2.  **Find PC1:** The direction of *maximum variance* (the longest spread).
3.  **Find PC2:** The direction of max variance perpendicular (orthogonal) to PC1.
4.  **Repeat.**

> [!CRITICAL]
> **PCA is scale-sensitive.** If you don't use `StandardScaler` first, the variable with the biggest numbers (e.g., Salary vs Age) will dominate PC1 completely.

---

## 2. Implementation & The "Scree Plot"

How many components do we keep? We check the **Cumulative Variance Explained**.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import numpy as np

data = load_breast_cancer()
X = data.data

# 1. Scale (Mandatory)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. PCA
pca = PCA().fit(X_scaled) # No limit yet, we just want to see

# 3. Scree Plot
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance Explained')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Threshold')
plt.title('How many components do we need?')
plt.legend()
plt.grid()
plt.show()

# Insight: We might see that 10 components explain 98% of the data. 
# We can drop the other 20 columns!
```

---

## 3. Interpreting the "Black Box" (Loadings)

A common complaint: "PCA ruins interpretability. What does 'Component 1' even mean?"
You can check the **Loadings** (which original feature contributed to the component).

```python
import pandas as pd

pc1_loadings = pca.components_[0]
feature_names = data.feature_names

# Sort by absolute contribution
df = pd.DataFrame({'Feature': feature_names, 'Weight': pc1_loadings})
df['Abs_Weight'] = df['Weight'].abs()
print(df.sort_values('Abs_Weight', ascending=False).head(5))

# Output: "Component 1 is heavily powered by 'Mean Radius', 'Perimeter', 'Area'."
# So PC1 = "Tumor Size".
```

---

## 4. PCA for Visualization (2D Projection)

We can't see 30 dimensions. But we can see 2.

```python
# Reduce to 2 dims
pca_2d = PCA(n_components=2)
X_pca = pca_2d.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data.target, cmap='viridis', alpha=0.5)
plt.xlabel('First Principal Component (Size-ish)')
plt.ylabel('Second Principal Component (Texture-ish)')
plt.title('Cancer Data projected to 2D')
plt.show()
```

If the colors separate well in 2D, a linear classifier will probably work well on the full dataset!
