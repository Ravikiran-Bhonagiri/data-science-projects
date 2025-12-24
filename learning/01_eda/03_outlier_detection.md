# 3. Detecting & Handling Outliers

## What is an Outlier?
An outlier is an observation that lies an abnormal distance from other values in a random sample.
- **Errors:** Data entry mistakes (Age = 200), sensor failures.
- **Natural:** Rare but valid events (Bill Gates' income, Usain Bolt's speed).

**Critical Rule:** Never remove an outlier just because it's high or low. Only remove it if you know it's an error. If it's valid, it might be the most interesting data point (fraud detection, anomaly detection).

---

## 3.1 Detection Methods

### 1. Z-Score Method (Parametric)
Quantifies how many standard deviations a data point is from the mean.
*   **Assumption:** Data is Normally Distributed.

```python
from scipy import stats
import numpy as np

# Calculate Z-scores
z_scores = np.abs(stats.zscore(df['salary']))

# Threshold: typically 3 (covers 99.7% of data)
outliers = df[z_scores > 3]
print(f"Detected {len(outliers)} outliers using Z-score")
```

### 2. IQR Method (Non-Parametric)
The most robust method, based on quartiles. Used in boxplots.
*   **Assumption:** None (works for skewed data).

```python
Q1 = df['salary'].quantile(0.25)
Q3 = df['salary'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['salary'] < lower_bound) | (df['salary'] > upper_bound)]
print(f"Detected {len(outliers)} outliers using IQR")
```

### 3. Isolation Forest (Machine Learning)
An unsupervised algorithm that isolates anomalies. Efficient for high-dimensional data.

```python
from sklearn.ensemble import IsolationForest

# Contamination is the expected propotion of outliers
clf = IsolationForest(contamination=0.01, random_state=42)
preds = clf.fit_predict(df[['salary', 'age', 'debt']])

# -1 indicates outlier, 1 indicates normal
df['is_outlier'] = preds
print(df[df['is_outlier'] == -1])
```

---

## 3.2 Handling Strategies

Once detected, what do you do?

### 1. Remove
*   **When:** You are certain it is a data error (e.g., negative age) or it is irrelevant to your specific problem.
```python
df_clean = df[(df['salary'] >= lower_bound) & (df['salary'] <= upper_bound)]
```

### 2. Cap (Winsorization)
*   **When:** You want to keep the data point but reduce its impact on statistical models (like linear regression).
*   **How:** Replace values > 99th percentile with the 99th percentile value.

```python
upper_limit = df['salary'].quantile(0.99)
lower_limit = df['salary'].quantile(0.01)

df['salary'] = np.where(df['salary'] > upper_limit, upper_limit,
               np.where(df['salary'] < lower_limit, lower_limit, df['salary']))
```

### 3. Log Transformation
*   **When:** Data is highly right-skewed (e.g., income, house prices). This compresses the scale and pulls outliers closer to the mean.

```python
import numpy as np

# Log(1+x) avoids error if value is 0
df['log_salary'] = np.log1p(df['salary'])
```

### 4. Treat as Separate Group
*   **When:** The outlier represents a specific, meaningful segment (e.g., "Whales" in gaming apps).
*   **Action:** Create a separate model for these users or a binary flag feature `is_high_value`.
