# 5. EDA Workflow Checklist

Follow this systematic process for every new dataset to ensure nothing is missed.

## Step 1: Preliminary Inspection
- [ ] **Load Data:** `pd.read_csv()`
- [ ] **Shape:** `df.shape` (How many rows/cols?)
- [ ] **Data Types:** `df.info()` (Are numbers stored as strings?)
- [ ] **Sample:** `df.head(5)` and `df.sample(5)` (Does the data look distinct?)
- [ ] **Duplicates:** `df.duplicated().sum()` (Remove duplicates early!)

## Step 2: Data Cleaning (The "Pre-EDA")
- [ ] **Fix Types:** Convert dates to datetime, numeric strings to floats.
- [ ] **Standardize Text:** Lowercase, remove whitespace from categorical strings.
- [ ] **Handle Missing:** Check `df.isnull().sum()`. Decide to drop or impute.

## Step 3: Univariate Analysis (Understand Feature individually)
- [ ] **Target Variable:** If supervised learning, analyze the target FIRST.
    - Is it balanced? (Classification)
    - Is it skewed? (Regression)
- [ ] **Numerical Features:** Check histograms. Look for skews.
- [ ] **Categorical Features:** Check value counts. Look for rare categories.

## Step 4: Bivariate Analysis (Find Relations)
- [ ] **Target Relationships:** How does each feature correlate with the target?
    - `groupby()` statistics.
    - Boxplots (Cat vs Num).
    - Scatterplots (Num vs Num).
- [ ] **Feature Correlations:** Check correlation matrix.
    - Identify highly correlated features (Redundant? Multicollinearity?).

## Step 5: Anomaly Detection
- [ ] **Outliers:** Check Z-scores or Boxplots.
- [ ] **Logical Checks:** Are there negative ages? Start dates after end dates?

## Step 6: Ask Questions
EDA isn't just code; it's an investigation.
- Is the data representative of reality?
- Are there missing segments?
- What features seem most predictive?
- Do we need to create new features? (Feature Engineering)

---

## Example Snippet

```python
def initial_eda(df):
    print("SHAPE:", df.shape)
    print("\nINFO:")
    print(df.info())
    print("\nMISSING VALUES:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    print("\nDUPLICATES:", df.duplicated().sum())
    
    # Statistical Summary
    print("\nSUMMARY STATS:")
    print(df.describe())
```
