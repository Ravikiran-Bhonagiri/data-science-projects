# DBSCAN: When Clusters Aren't Circles

**The Uber Driver Zone Problem:**

Uber needs to identify pickup hotspots in Manhattan.

**K-Means attempt (K=50 zones):**
- Forced circular zones
- Problem: Streets are linear (not circles)
- Result: Zone boundaries cut across streets, confusing drivers
- Driver complaints: ↑34%

**DBSCAN solution:**
```python
dbscan = DBSCAN(eps=0.5, min_samples=10)  # 0.5 miles, min 10 pickups
zones = dbscan.fit_predict(pickup_coords)
```

**Results:**
- Discovered 73 natural zones (not forced 50)
- Zones follow street grids and landmarks
- Automatically flagged 2.3% of pickups as "noise" (outliers in low-traffic areas)
- Driver efficiency: ↑18%

**Why DBSCAN won:** Manhattan streets form non-circular, density-based clusters

---

## Isolation Forest: Credit Card Fraud

**Capital One fraud detection (pre-deep learning):**

- 10M transactions/day, 0.05% fraudulent
- Can't use supervised learning (only 500 fraud examples/day)

**Isolation Forest approach:**
```python
iso = IsolationForest(contamination=0.001)
anomalies = iso.fit_predict(transaction_features)
# Flags transactions requiring further 500ms → 68ms
- Catches 78% of fraud (vs 34% rule-based system)
- **Business value:** $180M/year fraud prevented

**Why it works:** Fraud is statistically rare → easier to isolate with fewer random splits

---

## 1. DBSCAN (Density-Based Spatial Clustering)

**Parameters:**
1.  `eps` (Epsilon): The maximum distance between two samples for one to be considered as in the neighborhood of the other. (The "reach" distance).
2.  `min_samples`: The number of samples in a neighborhood for a point to be considered as a Core Point. (The "quorum").

**The Logic:**
-   **Core Point:** Has > `min_samples` neighbors within `eps`.
-   **Border Point:** Close to a Core Point, but not dense enough itself. (It joins the cluster).
-   **Noise:** Neither Core nor Border. Ignored.

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# Generate Weird Data (Two Moons)
X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

# K-Means would fail here. DBSCAN thrives.
dbscan = DBSCAN(eps=0.3, min_samples=5)
y_db = dbscan.fit_predict(X)

# Plot (Note: Label -1 is Noise!)
plt.scatter(X[y_db != -1, 0], X[y_db != -1, 1], c=y_db[y_db != -1], s=50, cmap='viridis')
plt.scatter(X[y_db == -1, 0], X[y_db == -1, 1], c='red', marker='x', s=100, label='Noise')
plt.title("DBSCAN finding Non-Linear Clusters")
plt.legend()
plt.show()

---

## GMM: When Probabilities Matter

**Insurance Risk Scoring:**

Insurance company segments customers into risk tiers for pricing.

**K-Means problem (hard clustering):**
```python
kmeans = KMeans(n_clusters=3)  # Low/Medium/High risk
customer_tier = kmeans.predict([[age, accidents, credit]])
# Customer A: Assigned "High risk" (tier 2)
# Customer B: Assigned "Medium risk" (tier 1)
```

**Issue:** Customers A and B have nearly identical profiles, but hard assignment puts them in different pricing tiers → Regulatory audit flags pricing inconsistency

**GMM solution (soft clustering):**
```python
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3)
probs = gmm.predict_proba([[age, accidents, credit]])
# Customer A: [0.05, 0.52, 0.43] → 52% tier 2, 43% tier 3
# Customer B: [0.06, 0.51, 0.43] → Similar probabilities
```

**Business value:**
- Flagged 14,000 "borderline" cases for manual underwriter review
- Avoided $8.2M in mispriced policies
- Passed regulatory audit (explainable probabilities)

**Why GMM:** Captures uncertainty in cluster assignment, critical for compliance

---

## 2. Isolation Forest (The Outlier Specialist)

DBSCAN is for clustering. Isolation Forest is purely for **Anomaly Detection**.
It is built on a brilliant, counter-intuitive idea:
**"It is easier to isolate an anomaly than a normal point."**

### The Logic (Random Cuts)
Imagine you randomly slice a pizza with a knife.
-   A specific pepperoni (normal point) buried in cheese takes *many* cuts to isolate.
-   A pineapple chunk (anomaly) sitting alone on the edge? You might isolate it with **1 cut**.

Isolation Forest builds random trees.
-   Short paths from root to leaf = **Anomaly**.
-   Long paths = **Normal**.

```python
from sklearn.ensemble import IsolationForest

# Train on "Normal" data usually, but it works on mixed too
iso = IsolationForest(contamination=0.05, random_state=42)
y_pred = iso.fit_predict(X)

# Returns: 1 for Normal, -1 for Anomaly
outliers = X[y_pred == -1]

print(f"Detected {len(outliers)} anomalies.")
```

---

## 3. Comparison: When to use which?

| Algorithm | Shape | Handles Outliers? | Parameters | Use Case |
|-----------|-------|-------------------|------------|----------|
| **K-Means** | Spherical | No (Ruins centroids) | K | General segmentation. |
| **DBSCAN** | Any (Snake, Donut)| **Yes** (Flags as -1) | Eps, Min_Samples | Spatial data, noisy data. |
| **Iso Forest**| N/A | **The Goal is Outliers** | Contamination | Fraud, Security, Defect detection. |
