<div align="center">

# ⚙️ Module 6: Feature Engineering

### *Transform Raw Data into Powerful Features*

![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![Difficulty](https://img.shields.io/badge/Difficulty-Intermediate-yellow?style=flat-square)
![Topics](https://img.shields.io/badge/Topics-7-orange?style=flat-square)

**The secret sauce that turns good models into great ones**

[🔧 Encoding](#-encoding-techniques) • [📐 Scaling](#-scaling-methods) • [🎯 Selection](#-feature-selection)

</div>

---

## 💡 Why Feature Engineering?

> **"Better features beat better algorithms."**

**Reality:**
- Raw data rarely works as-is
- Good features can 10× model performance
- More important than hyperparameter tuning

---

## 🔧 Encoding Techniques

**Handle categorical variables**

<table>
<tr>
<td width="50%">

### Standard Methods

**One-Hot Encoding**
```python
Gender: [Male, Female] →
Male_1, Male_0
Female_0, Female_1
```
✅ Use when: Low cardinality (<10 categories)  
❌ Avoid: High cardinality (explodes features)

**Label Encoding**
```python
Size: [S, M, L, XL] → [0, 1, 2, 3]
```
✅ Use: Ordinal categories  
❌ Avoid: Nominal (implies order)

</td>
<td width="50%">

### Advanced Methods

**Target Encoding**
```python
City → mean(target) per city
NYC: 0.65 (65% conversion)
LA:  0.52
```
✅ Handles high cardinality  
⚠️ Watch for overfitting

**Frequency Encoding**
```python
City → count / total
NYC: 0.35 (35% of data)
```
✅ Simple, effective

**Binary Encoding**
- Converts to binary
- <features than one-hot

</td>
</tr>
</table>

---

## 📐 Scaling Methods

**Normalize numerical features**

| Method | Formula | When to Use |
|--------|---------|-------------|
| **StandardScaler** | `(x - μ) / σ` | Most algorithms (SVM, KNN, Neural Nets) |
| **MinMaxScaler** | `(x - min) / (max - min)` | Bounded range [0,1] needed |
| **RobustScaler** | Uses median & IQR | Outliers present |
| **Normalizer** | Scale to unit norm | Text data (L1/L2 norm) |

**⚠️ Important:** Trees (Random Forest, XGBoost) DON'T need scaling!

---

## 🎯 Feature Selection

**Choose the best features, remove noise**

<table>
<tr>
<td width="33%">

### Filter Methods
**Before modeling**

**Correlation**
- Remove highly correlated  
- Threshold: |r| > 0.9

**Mutual Information**
- Measures dependency
- Works with non-linear

**Chi-Square**
- For categorical
- Statistical test

</td>
<td width="33%">

### Wrapper Methods
**Use model feedback**

**Recursive Feature Elimination (RFE)**
```python
from sklearn.feature_selection import RFE
selector = RFE(model, n_features=10)
selector.fit(X, y)
```

**Forward/Backward Selection**
- Add/remove iteratively
- Greedy search

</td>
<td width="33%">

### Embedded Methods
**During training**

**L1 Regularization (Lasso)**
- Shrinks coefficients to 0
- Automatic selection

**Tree Feature Importance**
```python
importances = model.feature_importances_
```

**SelectFromModel**
- sklearn wrapper

</td>
</tr>
</table>

---

## 🛠️ Feature Engineering Toolkit

### Numerical Transformations

```python
# Log Transform (right-skewed data)
df['price_log'] = np.log1p(df['price'])

# Polynomial Features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Binning
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 100])
```

### Feature Creation

```python
# Interactions
df['price_per_sqft'] = df['price'] / df['sqft']

# Aggregations
df['total_bedrooms'] = df['bed'] + df['bath']

# Time Features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6])
```

---

## 💡 What You'll Master

<table>
<tr>
<td width="50%">

### 🔧 Transformation Skills
- ✅ One-hot & label encoding
- ✅ Target & frequency encoding
- ✅ StandardScaler & alternatives
- ✅ Log & polynomial transforms
- ✅ Binning strategies

</td>
<td width="50%">

### 🎯 Selection Skills
- ✅ Correlation analysis
- ✅ Mutual information
- ✅ Recursive elimination (RFE)
- ✅ L1 regularization
- ✅ Tree importance

</td>
</tr>
</table>

---

## 🚨 Common Pitfalls

❌ **Scaling before train/test split** → Data leakage!  
❌ **One-hot encoding high cardinality** → Curse of dimensionality  
❌ **Ignoring feature scaling** → Poor SVM/KNN performance  
❌ **No domain knowledge** → Missing obvious features  
❌ **Over-engineering** → Complexity without gain  

---

## 🎯 Complete Pipeline Example

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Define transformers
numeric_features = ['age', 'income', 'credit_score']
categorical_features = ['city', 'gender', 'occupation']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier())
])

# Train
pipeline.fit(X_train, y_train)
```

---

<div align="center">

**Engineer Features, Engineer Success** ⚙️

*7 topics • Encoding + Scaling + Selection*

[⬅️ Model Evaluation](../05_evaluation/) • [🏠 Home](../../README.md) • [➡️ Unstructured Data](../07_unstructured_data/)

</div>
