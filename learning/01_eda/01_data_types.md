# 1. Understanding Data Types

## Why It Matters
Understanding your features is essential for choosing the right analysis techniques and models. Treating categorical data as numerical (e.g., ZIP codes) or vice versa can lead to disastrous model performance.

---

## 1.1 Numerical (Quantitative) Data
Data represented by numbers with meaningful mathematical operations.

### Continuous
- **Definition:** Can take any value within a range (infinite possibilities).
- **Examples:** 
  - Height (170.5 cm)
  - Temperature (98.6°F)
  - Stock Price ($142.37)
- **Common Operations:** Mean, Standard Deviation, Regression
- **Visualization:** Histograms, Box Plots, Density Plots, Scatter Plots

### Discrete
- **Definition:** Integer values, typically counts.
- **Examples:** 
  - Number of children (0, 1, 2...)
  - Items purchased (1, 5, 10)
  - Web page visits
- **Common Operations:** Count, Mode, Frequency, Poisson distribution analysis
- **Visualization:** Bar Charts, Count Plots

---

## 1.2 Categorical (Qualitative) Data
Data that represents categories or groups.

### Nominal
- **Definition:** Categories without inherent order.
- **Examples:** 
  - Color: Red, Blue, Green
  - Country: USA, Canada, UK
  - Product Type: Electronics, Clothing, Food
- **Encoding:** 
  - One-Hot Encoding (`pd.get_dummies`)
  - Label Encoding (only for tree-based models)
- **Visualization:** Bar Charts, Pie Charts

### Ordinal
- **Definition:** Categories with a meaningful order or ranking.
- **Examples:**
  - Education: High School < Bachelor's < Master's < PhD
  - Rating: Poor < Fair < Good < Excellent
  - T-Shirt Size: XS < S < M < L < XL
- **Encoding:** Ordinal Encoding (mapping values like 1, 2, 3...)
- **Visualization:** Ordered Bar Charts

---

## 1.3 Python Implementation

```python
import pandas as pd
import numpy as np

# Load example data
df = pd.read_csv('data.csv')

# --- 1. Identify Data Types ---
print(df.dtypes)
print(df.info()) 

# --- 2. Convert Data Types ---
# Convert string to integer
df['age'] = df['age'].astype(int)

# Convert string to category (saves memory, enables categorical operations)
df['grade'] = df['grade'].astype('category')

# Convert object column to datetime
df['date'] = pd.to_datetime(df['date'])

# --- 3. Statistical Inspection ---
# For Numerical
print(df.describe())

# For Categorical
print(df.describe(include=['O', 'category']))
print(df['status'].value_counts())
```

### Pro Tip:
Always check cardinality (number of unique values) for categorical variables. High cardinality (e.g., User ID, ZIP code) can cause issues for some encoding methods and models.
