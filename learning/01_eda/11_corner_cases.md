# 11. EDA Corner Cases & Gotchas

A collection of edge cases, traps, and "gotchas" that experienced data scientists watch for during EDA.

---

## 1. High Cardinality Trap

**Problem:** Column has too many unique categories.

**Detection:**
```python
def check_cardinality(df, column):
    n_unique = df[column].nunique()
    n_rows = len(df)
    pct = (n_unique / n_rows) * 100
    
    if pct > 10:
        print(f"⚠️ {column}: {n_unique} unique ({pct:.1f}%) - HIGH CARDINALITY")
```

**Rules:**
- **< 10 categories:** Safe for One-Hot Encoding
- **10-50 categories:** Use Label Encoding or Target Encoding
- **> 50 or > 10% of rows:** Group rare values or extract patterns

**Fix:**
```python
# Group rare categories
top_N = df['City'].value_counts().nlargest(20).index
df['City_Grouped'] = df['City'].where(df['City'].isin(top_N), 'Other')
```

---

## 2. Data Leakage (The Silent Killer)

**Problem:** Information from the future/test set "leaks" into training.

### Type A: Target Leakage
Feature contains information **only available AFTER** the target is known.

**Example:**
```python
# BAD: 'default_date' only exists AFTER customer defaults
df['days_since_default'] = (today - df['default_date']).dt.days
```

**How to detect:**
- Feature has 100% correlation with target
- Feature is missing for all non-target cases
- Ask: "Would I know this value BEFORE making the prediction?"

### Type B: Train-Test Contamination
Fitting transformations on the entire dataset before splitting.

**Bad:**
```python
# WRONG! Imputer sees test data statistics
imputer.fit(df)  # Fitted on ALL data
X_train, X_test = train_test_split(df)
```

**Correct:**
```python
X_train, X_test = train_test_split(df)
imputer.fit(X_train)  # Only fit on training
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)
```

---

## 3. Temporal Ordering Violation

**Problem:** For time-series, random train/test split breaks temporal logic.

**Bad:**
```python
# Randomly splits 2020 and 2023 data into both train/test
X_train, X_test = train_test_split(df, test_size=0.2)
```

**Correct:**
```python
# Train on past, test on future
split_date = '2022-12-31'
train = df[df['date'] <= split_date]
test = df[df['date'] > split_date]
```

---

## 4. Duplicate Rows with Different Targets

**Problem:** Same features, different target values.

**Detection:**
```python
# Find duplicates in features but different target
duplicates = df[df.duplicated(subset=feature_cols, keep=False)]
conflicting = duplicates.groupby(feature_cols)['target'].nunique()
conflicts = conflicting[conflicting > 1]

if len(conflicts) > 0:
    print(f"⚠️ {len(conflicts)} feature sets have conflicting targets!")
```

**Causes:**
- Data collection error
- Non-deterministic target (e.g., user behavior)
- Missing important features

**Fix:**
- Investigate root cause
- Aggregate (majority vote, average)
- Remove if too few

---

## 5. Quasi-Constant Features

**Problem:** Feature has 99%+ the same value (useless variance).

**Detection:**
```python
# Check if one value dominates
for col in df.columns:
    top_freq = df[col].value_counts(normalize=True).iloc[0]
    if top_freq > 0.95:
        print(f"⚠️ {col}: {top_freq:.1%} are the same value")
```

**Fix:**
```python
# Drop columns with >95% same value
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
df_reduced = selector.fit_transform(df)
```

---

## 6. Mixed Data Types in Columns

**Problem:** Column labeled "int64" but contains strings or nulls.

**Example:**
```python
# Column 'Age' stored as object because of:
df['Age'] = ['25', '30', 'Unknown', '22', np.nan]
```

**Detection:**
```python
# Find object columns that should be numeric
for col in df.select_dtypes(include='object').columns:
    try:
        pd.to_numeric(df[col])
        print(f"⚠️ {col} can be converted to numeric!")
    except:
        pass
```

**Fix:**
```python
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')  # NaN for 'Unknown'
```

---

## 7. The "99999" Placeholder Trap

**Problem:** Missing values encoded as sentinel values (9999, -1, 0) instead of NaN.

**Detection:**
```python
# Look for suspiciously round numbers
df.describe()  # Check max values for 999, 9999, -1
```

**Fix:**
```python
df['Income'].replace([9999, -1, 0], np.nan, inplace=True)
```

---

## 8. Dates Stored as Strings/Objects

**Problem:** '2023-01-01' is useless as a string; model can't learn from it.

**Detection:**
```python
# Find columns that look like dates
for col in df.select_dtypes(include='object').columns:
    if df[col].str.match(r'\d{4}-\d{2}-\d{2}').sum() > 100:
        print(f"⚠️ {col} might be a date!")
```

**Fix:**
```python
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['is_weekend'] = df['date'].dt.dayofweek > 4
```

---

## 9. Feature and Target from Same Source

**Problem:** Feature is mathematically derived from the target.

**Example:**
```python
# BAD: Total = Monthly * Tenure (perfect multicollinearity)
df['TotalCharges'] = df['MonthlyCharges'] * df['Tenure']
```

**Detection:** Very high correlation (>0.99) or perfect VIF.

**Fix:** Drop one of the redundant features.

---

## 10. Outliers That Are Actually Correct

**Problem:** Removing valid extreme values (e.g., CEO salary in company data).

**Rule:** Never auto-delete outliers. Investigate first.

**Questions to ask:**
1. Is this a data entry error? (Age = 200?)
2. Is this a special segment? (VIP customers?)
3. Does it represent fraud/anomaly? (Keep it!)

---

## 11. Stratification Ignored in Imbalanced Data

**Problem:** Random split might put all minority class in train OR test.

**Bad:**
```python
# With 1% fraud, test set might have 0 fraud cases
X_train, X_test = train_test_split(X, y, test_size=0.2)
```

**Correct:**
```python
# Ensures both sets have ~1% fraud
X_train, X_test = train_test_split(X, y, test_size=0.2, stratify=y)
```

---

## 12. Special Characters in Column Names

**Problem:** Spaces, dots, or special chars break code.

**Example:**
```python
df['Customer.Name']  # Works
df.Customer.Name     # Breaks! (dot is attribute access)
```

**Fix:**
```python
# Standardize column names
df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('.', '_')
```

---

## 13. Silent Nulls (Empty Strings)

**Problem:** Missing values hidden as `''` or `' '`.

**Detection:**
```python
# Count empty strings
(df == '').sum()       # Empty
(df == ' ').sum()      # Single space
df.str.strip().eq('').sum()  # Whitespace only
```

**Fix:**
```python
df.replace(['', ' ', 'N/A', 'NULL'], np.nan, inplace=True)
```

---

## 14. Index Not Reset After Filtering

**Problem:** After dropping rows, index is non-consecutive [0, 1, 5, 8...].

**Issue:** This breaks iloc operations.

**Fix:**
```python
df = df[df['Age'] > 18].reset_index(drop=True)
```

---

## 15. The "Test Set is Perfect" Illusion

**Problem:** Test accuracy way higher than train → Test set leaked into transformations.

**Detection:**
```python
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

if test_score > train_score + 0.05:
    print("⚠️ POSSIBLE DATA LEAKAGE!")
```

---

## Checklist: Before Modeling

- [ ] No data leakage (checked temporal order, fit/transform sequence)
- [ ] Train/test split stratified for imbalanced targets
- [ ] High cardinality features grouped or encoded properly
- [ ] Dates converted and components extracted
- [ ] Outliers investigated (not auto-deleted)
- [ ] Missing value sentinels (9999, -1) replaced with NaN
- [ ] Column names standardized (no spaces/dots)
- [ ] Index reset after filtering
- [ ] Duplicate rows with conflicting targets handled
- [ ] Quasi-constant features removed
