# 6. EDA Strategy by Problem Type

Your EDA strategy changes significantly depending on whether you are solving a **Regression** problem (predicting a number) or a **Classification** problem (predicting a label).

---

## 🏗️ 1. EDA for Regression
**Goal:** Predict a continuous quantity (e.g., Home Price, Temperature, Sales).
**Key Focus:** Linearity, Correlation, and Outliers.

### A. Analyzing the Target (Y)
*   **Distribution Check:** Is the target Normally Distributed? Linear Regression works best with normal targets.
    *   **tool:** Histogram / KDE plot.
    *   **Action:** If right-skewed (long tail), apply `np.log1p(y)` transformation.
    *   **Action:** If bimodal, maybe you need two separate models?

```python
sns.histplot(df['price'], kde=True)
print(f"Skewness: {df['price'].skew()}")
```

### B. Feature vs Target Analysis
*   **Linearity:** Do features have a linear relationship with the target?
    *   **Tool:** Scatter plots (`sns.scatterplot(x='sqft', y='price')`).
    *   **Action:** If curve is exponential, try log-transforming the feature.
*   **Correlation:** How strong is the linear relationship?
    *   **Tool:** Pearson Correlation (`df.corr()['price']`).
*   **Homoscedasticity check:** Does variance change as feature increases? (Look for "cone" shapes in scatter plots).

### C. Red Flags for Regression
1.  **Outliers:** Extreme values in the target pull the regression line massively. *Must* handle them (cap or remove).
2.  **Collinearity:** Two features highly correlated (e.g., `temp_C` and `temp_F`) break regression math. Check VIF.

---

## 🚦 2. EDA for Classification
**Goal:** Predict a class label (e.g., Churn/Stay, Fraud/Legal, Cat/Dog).
**Key Focus:** Separability and Class Balance.

### A. Analyzing the Target (Y)
*   **Class Balance:** Is one class dominant?
    *   **Tool:** Countplot (`sns.countplot(x='churn')`).
    *   **Risk:** If Churn is only 1%, model will just guess "No Churn" and get 99% accuracy.
    *   **Action:** Plan for Resampling (SMOTE) or use appropriate metrics (F1-score, Recall).

### B. Feature vs Target Analysis
We want to know: *Does this feature help separate the classes?*

*   **Numerical Features:** Compare distributions for each class.
    *   **Tool:** Boxplot or Violin Plot split by target.
    *   **Example:** `sns.boxplot(x='churn', y='monthly_bill')`
    *   **Insight:** If the medians are far apart, the feature is predictive! If boxes overlap perfectly, the feature is useless.
*   **Categorical Features:** Check proportional differences.
    *   **Tool:** Stacked Bar Chart / Cross-tabulation.
    *   **Example:** `pd.crosstab(df['gender'], df['churn'], normalize='index').plot(kind='bar', stacked=True)`
    *   **Insight:** If 50% of males churn but only 20% of females churn, Gender is predictive.

### C. Visualizing Separability
Can we draw a line to separate classes?
*   **Tool:** Pairplot with hue.
```python
sns.pairplot(df, hue='target', vars=['feature1', 'feature2'])
```
*   **Insight:** Look for distinct clusters of colors. If colors are completely mixed, the problem is hard (non-linear).

---

## ⚔️ Identifying the Problem Type Cheat Sheet

| Feature | Regression | Classification |
| :--- | :--- | :--- |
| **Target Variable** | Continuous (Price, Age) | Categorical (Yes/No, Spam/Ham) |
| **Primary Plot** | Scatter Plot (x vs y) | Boxplot (y grouped by x) |
| **Key Statistic** | Pearson Correlation | Information Gain / Chi-Square |
| **Biggest Enemy** | Outliers | Class Imbalance |
| **Transformation** | Log-transform Target | SMOTE / Class Weights |
