# SVM: The High-Dimensional Specialist

**The Cancer Genomics Challenge:**

Researchers have 50 cancer patients and 20,000 gene measurements per patient.

**Problem:** Predict which patients will respond to immunotherapy.

**Random Forest attempt:**
```python
rf = RandomForestClassifier()
rf.fit(X_train, y_train)  # 40 patients, 20,000 features
# Result: 100% training accuracy, 50% test accuracy (random guessing)
```

**Why it failed:** 20,000 features, 40 samples → massive overfitting

**SVM with RBF kernel:**
```python
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_scaled_train, y_train)
# Result: 75% training, 72% test (generalizes!)
```

**Why SVM works:**
- Only uses "support vectors" (edge cases) → fewer parameters than samples
- Kernel trick handles non-linear boundaries without explicit feature expansion
- Regularization (C parameter) prevents overfitting

**Clinical deployment:** Model deployed in 12 hospitals for treatment planning

---

## Production Use Cases

**Handwritten digit recognition (pre-deep learning):**
- 784 pixel features (28×28 images)
- Linear SVM: 94% accuracy
- Why: High-dimensional, sparse data

**Text classification (spam detection):**
- 100,000 word features
- Linear SVM: 98.7% accuracy, <5ms inference
- Beats neural networks on speed

---

## 1. The Intuition: The "Widest Street"
SVM tries to find a decision boundary (hyperplane) that maximizes the **margin** (distance) between the two classes.

- **Support Vectors:** The specific data points closer to the boundary that "support" or define it. (Moving other points doesn't change the model!)

---

## 2. The Kernel Trick (Magic!)
What if data isn't separable by a straight line? (e.g., Concentric circles).

SVM projects data into a **higher-dimensional space** (Z-axis) where it *becomes* linearly separable, without actually calculating the coordinates (computationally cheap).

**Common Kernels:**
- **Linear:** Fast, good for text classification (high dim).
- **RBF (Radial Basis Function):** The default. Creates "circles" around data.
- **Poly:** Polynomial boundaries.

---

## 3. Key Parameters: C and Gamma

### C (Regularization Parameter)
- **High C:** Strict. Tries to classify every training point correctly. Risk of **Overfitting**.
- **Low C:** Loose. Allows some misclassification to find a wider margin. Smoother boundary.

### Gamma (RBF Kernel Coefficient)
- "How far the influence of a single training example reaches."
- **High Gamma:** Only nearby points matter. Boundary hugs data tightly. **Overfitting**.
- **Low Gamma:** Distant points matter. Boundary is smooth/linear. Underfitting.

> [!TIP]
> Always tune C and Gamma together using GridSearch.

---

## 4. Implementation

> [!IMPORTANT]
> **SVM REQUIRES SCALING!** Since it calculates distances, if one feature is in millions and another in decimals, the model breaks.

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Pipeline is best practice
clf = make_pipeline(
    StandardScaler(),
    SVC(
        kernel='rbf', 
        C=1.0, 
        gamma='scale', 
        probability=True  # Slower, but needed if you want predict_proba()
    )
)

clf.fit(X_train, y_train)
```

---

## 5. Support Vector Regression (SVR)
SVM works for regression too. It tries to fit a "tube" around the data where errors within epsilon ($\epsilon$) are ignored.

```python
from sklearn.svm import SVR

reg = make_pipeline(
    StandardScaler(),
    SVR(kernel='rbf', C=100, epsilon=0.1)
)
reg.fit(X_train, y_train)
```
