# Decision Trees: The "20 Questions" Game

Imagine playing "20 Questions."
"Is it an animal?" **Yes.**
"Does it bark?" **No.**
"Does it meow?" **Yes.**
**It's a cat!**

That is exactly what a Decision Tree does. It slices your data into smaller and smaller boxes until it finds the answer. It's the only machine learning model that thinks exactly like a human.

### 🧠 When should you actually use this?

**1. The Emergency Room Doctor:**
A patient enters with chest pain. You need a protocol that any nurse can follow *instantly*.
*   **The Protocol:** "Is Age > 50?" -> (Yes) -> "Is Blood Pressure > 140?" -> (Yes) -> **Code Red.**
*   **Why Trees?** You can print this out and tape it to the wall. It requires no computer to interpret. It's fully transparent.

**2. The Customer Churn Analyst:**
Your boss asks: "Why are people leaving our Netflix-competitor service?"
*   **The Model:** Finds that `Usage < 5 hours` AND `Contract = Month-to-Month` = **Churn**.
*   **Why Trees?** It gives you a specific segment to target. "Let's offer a discount to people who watch less than 5 hours!"

---

## 1. Mathematical Foundation

A decision tree splits the feature space into rectangles. The goal is to create "pure" leaf nodes.

### Classification Criteria

#### A. Gini Impurity (Default)
Measures the probability that a randomly selected element would be incorrectly classified.
$$Gini(p) = 1 - \sum_{i=1}^{C} p_i^2$$
- $p_i$ is probability of class $i$ in the node.
- **Range:** [0, 0.5] (for binary class).
- **Behavior:** Computationally faster (no logs). Favors larger partitions.

#### B. Entropy (Information Gain)
Measures the level of "surprise" or disorder in the node.
$$Entropy(H) = - \sum_{i=1}^{C} p_i \log_2(p_i)$$
- **Information Gain:** $Gain = H(Parent) - \sum \frac{N_{child}}{N_{parent}} H(Child)$
- **Range:** [0, 1].
- **Behavior:** Sensitive to class imbalance. Tends to produce slightly more balanced trees.

### Regression Criteria
#### C. Mean Squared Error (MSE)
Minimizes the L2 loss within each leaf. The prediction is the **mean** of samples.
$$MSE = \frac{1}{N} \sum (y_i - \bar{y})^2$$

#### D. Mean Absolute Error (MAE)
Minimizes the L1 loss. The prediction is the **median**. Robust to outliers.

---

## 2. The Bias-Variance Tradeoff in Trees

| Scenario | Depth | Bias | Variance | Behavior |
|----------|-------|------|----------|----------|
| **Shallow Tree** | Low (e.g., 2) | High | Low | **Underfitting.** Misses patterns. |
| **Deep Tree** | High (e.g., 20) | Low | High | **Overfitting.** Memorizes noise. |

> [!CRITICAL]
> **Trees assume nothing.** Unlike Linear Regression (which assumes a straight line), trees can fit ANY shape. This flexibility is why they overfit so easily.

---

## 3. Regularization: Pruning Strategies

Pruning is essential to creating generalizable trees.

### A. Pre-Pruning (Hyperparameters)
Stop the tree *before* it becomes too complex.
- `max_depth`: Hard limit. Good first step.
- `min_samples_split`: Increasing this (e.g., to 20) prevents isolating noise.
- `min_samples_leaf`: Forces leaves to have a "quorum" (e.g., 10). **Most effective** smoothing parameter.
- `max_features`: Randomly selecting features at each split (key for Random Forests).

### B. Post-Pruning (Cost Complexity Pruning)
Grow a massive tree, then cut back branches that don't justify their complexity.
Formula: $R_\alpha(T) = R(T) + \alpha|T|$
- $R(T)$: Misclassification rate
- $|T|$: Number of leaves
- $\alpha$: Complexity penalty (Tune this!)

---

## 4. Comprehensive Implementation

### Classification Example
We will visualize the decision boundary to see how the tree "chops" the space.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier, plot_tree
from mlxtend.plotting import plot_decision_regions

# 1. Generate Non-Linear Data
X, y = make_moons(n_samples=200, noise=0.25, random_state=42)

# 2. Train Models (Constrained vs Unconstrained)
tree_deep = DecisionTreeClassifier(max_depth=None, random_state=42)
tree_pruned = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42)

tree_deep.fit(X, y)
tree_pruned.fit(X, y)

# 3. Visualization
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Deep Tree (Overfit)
plot_decision_regions(X, y, clf=tree_deep, ax=ax[0])
ax[0].set_title(f"Unconstrained Tree (Train Acc: {tree_deep.score(X, y):.2f})")

# Pruned Tree (Good Fit)
plot_decision_regions(X, y, clf=tree_pruned, ax=ax[1])
ax[1].set_title(f"Pruned Tree (Train Acc: {tree_pruned.score(X, y):.2f})")

plt.show()
```

### Regression Tree Logic
When predicting a continuous value, the tree doesn't output a probability; it outputs a discrete value (the average of the inputs in that leaf).

```python
from sklearn.tree import DecisionTreeRegressor

# Generate Sine Wave with Noise
X_reg = np.sort(5 * np.random.rand(80, 1), axis=0)
y_reg = np.sin(X_reg).ravel() + 0.1 * np.random.randn(80)

# Train
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X_reg, y_reg)
regr_2.fit(X_reg, y_reg)

# Visualizing the "Staircase" Effect
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
plt.figure(figsize=(10, 6))
plt.scatter(X_reg, y_reg, s=20, edgecolor="black", c="darkorange", label="Data")
plt.plot(X_test, regr_1.predict(X_test), color="cornflowerblue", label="max_depth=2", linewidth=2)
plt.plot(X_test, regr_2.predict(X_test), color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()
```

---

## 5. Feature Importance Implementation

How does Sklearn calculate importance?
It calculates the **weighted decrease in impurity** summed over all nodes where that feature is used.

**Interpretation:**
- "Feature A reduced the Gini impurity by 0.3 total across the tree."
- **Flaw:** It biases high cardinality features (e.g., "Transaction ID").

```python
import pandas as pd

importances = pd.DataFrame({
    'Feature': ['X1', 'X2'],
    'Importance': tree_pruned.feature_importances_
}).sort_values('Importance', ascending=False)
print(importances)
```

---

## 6. Corner Cases & Gotchas

1.  **Extrapolation:** Regression trees cannot predict values outside the range seen in training. They will flatline. (e.g., if max training price was \$500, it can never predict \$600).
2.  **Instability:** Changing one training point can change the entire tree structure. (Solved by Random Forests).
3.  **Axis-Aligned Splits:** Trees split orthogonally ($X > 5$). They struggle with diagonal boundaries ($X > Y$).
    - *Fix:* Use PCA to rotate data first.
