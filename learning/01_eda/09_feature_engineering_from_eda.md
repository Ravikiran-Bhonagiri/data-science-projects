# 9. Feature Engineering from EDA

EDA is not just about *looking* at data; it's about *fixing* it to help the model learn. This bridge between "visualization" and "modeling" is Feature Engineering.

---

## 1. Insight: "The relationship is non-linear"
**Observation:** In a scatter plot of `Age` vs `Health_Cost`, you see costs spike after age 60. A straight line (Linear Regression) helps, but misses the curve.

**Action: Polynomial/Log Features**
```python
# Capture the curve
df['Age_Squared'] = df['Age'] ** 2
df['Age_Log'] = np.log1p(df['Age'])
```

---

## 2. Insight: "This category is too granular"
**Observation:** `Country` has 150 values. 100 of them appear only once.
**Problem:** Model will overfit to these rare countries.

**Action: Binning / Grouping (Rare Label Encoding)**
Group rare values into "Other".
```python
# Keep top 10 countries, label rest as 'Other'
top_10 = df['Country'].value_counts().nlargest(10).index
df['Country_Grouped'] = df['Country'].where(df['Country'].isin(top_10), 'Other')
```

---

## 3. Insight: "Continuous variable has clear clusters"
**Observation:** `Time_of_Day` affects traffic, but it's not linear (1 AM is similar to 2 AM, but 8 AM is peak). The histogram shows peaks at Morning (8am), Lunch (1pm), and Evening (6pm).

**Action: Discretization (Binning)**
Turn numbers into logical categories.
```python
# Create meaningful buckets
df['Time_Period'] = pd.cut(df['Hour'], 
                           bins=[0, 6, 12, 18, 24], 
                           labels=['Night', 'Morning', 'Afternoon', 'Evening'])
```

---

## 4. Insight: "These two features define the outcome together"
**Observation:** High `Price` doesn't cause churn... but High `Price` with Low `Quality` definitely does.
**Problem:** Linear models treat Price and Quality separately.

**Action: Interaction Features**
Multiply them to let the model see the combination.
```python
df['Price_per_Quality'] = df['Price'] / df['Quality_Score']
df['Price_x_Tenure'] = df['Price'] * df['Tenure']
```

---

## 5. Insight: "Dates are useless strings"
**Observation:** `2023-01-01` is hard for a model to read.

**Action: Date Component Extraction**
Extract cyclical patterns.
```python
df['dt'] = pd.to_datetime(df['date_str'])

df['Month'] = df['dt'].dt.month
df['Is_Weekend'] = df['dt'].dt.dayofweek > 4
df['Year'] = df['dt'].dt.year  # Catch long term trend
```

---

## Checklist: From Plot to Code

| What you see in Plot | Feature Engineering Action |
| :--- | :--- |
| Skewed Distribution | Log Transform (`np.log1p`) |
| Bimodal Distribution | Binning / Categorization |
| Rare Categories | Group into "Other" |
| Cyclical Data (Hours/Months) | Sine/Cosine Transform or Binning |
| Non-Linear Scatter | Polynomial Features (`x^2`) |
