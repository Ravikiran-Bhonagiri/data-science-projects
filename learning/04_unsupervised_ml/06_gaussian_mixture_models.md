# Gaussian Mixture Models (GMM): The "Soft" Clustering

K-Means is a bit of a dictator. It says: "You belong to Cluster A. Period."
But what if a data point is right on the border?
What if it's 51% Cluster A and 49% Cluster B?

**Gaussian Mixture Models (GMM)** are the "probabilistic" cousin of K-Means.
It doesn't give you a hard label. It gives you a probability distribution.
*   "You are 80% Cluster A, 19% Cluster B, and 1% Cluster C."

### 🧠 When should you actually use this?

**1. The Voice Assistant:**
You record someone speaking. Is that sound an "Ah" or an "Eh"?
*   **The Reality:** Speech is continuous. It causes ambiguity.
*   **Why GMM?** It models the underlying *probability* of the phoneme, allowing the system to keep multiple options open until it hears the rest of the sentence.

**2. The Risk Manager:**
You are classifying risky customers.
*   **The Problem:** K-Means forces a "Safe/Risky" label. But a customer on the border is the most interesting one!
*   **Why GMM?** It identifies the "uncertain" middle ground, which you can flag for manual review.

---

## 1. The Math: "Mixtures of Gaussians"

K-Means assumes clusters are **Spherical** (circles).
GMM assumes clusters are **Gaussian** (bell curves).
Crucially, a Gaussian can be stretched! It can be an oval (elliptical).

**Components:**
1.  **$\mu$ (Mean):** The center of the cluster.
2.  **$\Sigma$ (Covariance):** The shape/width of the cluster (circle, oval, tilted oval).
3.  **$\pi$ (Mixing Coeff):** How big/dense the cluster is.

The Algorithm uses **Expectation-Maximization (EM):**
1.  **E-Step:** Estimate the probability that each point belongs to each cluster.
2.  **M-Step:** Update the parameters ($\mu, \Sigma, \pi$) to maximize the likelihood of the data given those probabilities.
3.  **Repeat.**

---

## 2. Implementation: GMM in Scikit-Learn

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# Generate data with stretched shapes (Anisotropic)
X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.60, random_state=0)
rng = np.random.RandomState(13)
X_stretched = np.dot(X, rng.randn(2, 2)) # Stretch it!

# 1. Fit GMM
gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
gmm.fit(X_stretched)

# 2. Soft Clustering (Probabilities)
probs = gmm.predict_proba(X_stretched)
print("Probability of Point 0 belonging to each cluster:")
print(f"Cluster 0: {probs[0][0]:.3f}, Cluster 1: {probs[0][1]:.3f} ...")

# 3. Hard Clustering (for plotting)
labels = gmm.predict(X_stretched)

# Plot
plt.figure(figsize=(8,6))
plt.scatter(X_stretched[:, 0], X_stretched[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
plt.title("GMM handles Stretched Clusters (Ovals) perfectly!")
plt.show()
```

---

## 3. Generative AI: GMM can "Dream"

Because GMM learns the *probability distribution* of the data, it can simply *sample* from that distribution to create **new data** that looks like the original.
(K-Means cannot do this).

```python
# Generate 10 new samples that "look like" the training data
X_new, y_new = gmm.sample(10)
print(X_new)
```

**Use Case:** Synthetic Data Generation for preserving privacy. Train GMM on real sensitive data, sample synthetic data, share the synthetic data.

---

## 4. Model Selection: BIC and AIC

How many clusters ($N$) do we choose?
In K-Means, we used Inertia (Elbow).
In GMM, we use **BIC (Bayesian Information Criterion)**.
$$BIC = -2 \ln(\hat{L}) + k \ln(n)$$
*   It rewards model fit (Likelihood $L$).
*   It penalizes complexity ($k$ parameters).
*   **Lower is Better.**

```python
n_components = np.arange(1, 15)
models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(X_stretched)
          for n in n_components]

plt.plot(n_components, [m.bic(X_stretched) for m in models], label='BIC')
plt.xlabel('n_components')
plt.ylabel('BIC Score (Lower is Better)')
plt.legend()
plt.show()
```
