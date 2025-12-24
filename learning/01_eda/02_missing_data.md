# 2. Handling Missing Data

## The Reality of Data
Missing data is inevitable. How you handle it can significantly impact your analysis. Ignoring it causes errors; handling it poorly introduces bias.

---

## 2.1 Identifying Missing Data

Before fixing it, you must find it and understand its extent.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno

# 1. Basic Count
print(df.isnull().sum())  # Count per column

# 2. Percentage
missing_pct = df.isnull().sum() / len(df) * 100
print(missing_pct[missing_pct > 0]) # Show only columns with missing data

# 3. Visualization matrix (Best for seeing patterns)
msno.matrix(df)
plt.show()
```

---

## 2.2 Why is it Missing? (IMPORTANT)

Classifying the mechanism of missingness determines your strategy.

1.  **Missing Completely at Random (MCAR):**
    *   No pattern. The probability of missingness is unrelated to any data (observed or missing).
    *   **Example:** A sensor glitched randomly.
    *   **Strategy:** Safe to delete rows or use simple imputation.

2.  **Missing at Random (MAR):**
    *   Missingness is related to *other* observed variables, but not the missing value itself.
    *   **Example:** Women are less likely to report weight (if gender is recorded).
    *   **Strategy:** Advanced imputation (KNN, Regression) works best. Simple deletion can bias results.

3.  **Missing Not at Random (MNAR):**
    *   Missingness depends on the value itself.
    *   **Example:** Top earners don't disclose income (missing because income is high).
    *   **Strategy:** Most critical. Requires domain knowledge. Deletion/Imputation will introduce severe bias. Often requires modeling the missingness.

---

## 2.3 Handling Strategies

### A. Deletion

**1. Listwise Deletion (Drop Rows)**
```python
# Drop any row with ANY missing value
df_clean = df.dropna()

# Drop rows where critical target variable is missing
df_clean = df.dropna(subset=['target_variable'])
```
*   **When:** < 5% missing, and data is MCAR.
*   **Risk:** Loss of statistical power.

**2. Feature Deletion (Drop Columns)**
```python
# Drop columns with > 50% missing values
limit = len(df) * 0.5
df_clean = df.dropna(axis=1, thresh=limit)
```
*   **When:** Variable is mostly empty and not critical.

### B. Simple Imputation

**1. Mean/Median (Numerical)**
```python
from sklearn.impute import SimpleImputer

# Use Median (robust to outliers)
imputer = SimpleImputer(strategy='median')
df['age'] = imputer.fit_transform(df[['age']])
```

**2. Mode (Categorical)**
```python
# Fill with most frequent value
df['category'].fillna(df['category'].mode()[0], inplace=True)

# Or fill with a new category "Unknown" (Sometimes safer!)
df['category'].fillna('Unknown', inplace=True)
```

### C. Advanced Imputation

**1. K-Nearest Neighbors (KNN)**
Finds 'k' most similar rows based on other features and averages their value.
```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
df_imputed = imputer.fit_transform(df) # Returns numpy array
```
*   **Pros:** Very accurate for MAR data.
*   **Cons:** Computationally expensive (O(N^2)).

**2. Time-Series Fill**
```python
# Forward Fill (use previous value)
df['stock_price'].fillna(method='ffill', inplace=True)

# Interpolation (linear fitting)
df['temp'].interpolate(method='linear', inplace=True)
```

**3. MICE (Multivariate Imputation by Chained Equations)**
Models each feature with missing values as a function of valid features.
```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(random_state=0)
df_imputed = imputer.fit_transform(df)
```
