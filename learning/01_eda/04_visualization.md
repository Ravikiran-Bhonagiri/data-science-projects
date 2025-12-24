# 4. Data Visualization for EDA

Visualization is the fastest way to understand data. Humans are visual creatures; we spot patterns in plots that statistics summary tables hide (see Anscombe's Quartet).

---

## 4.1 Univariate Analysis (One Variable)
Analyzing the distribution of a single feature.

### Numerical Features
**1. Histogram**
*   **Use:** See the shape of distribution (Normal, Skewed, Bimodal).
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(data=df, x='age', kde=True, bins=30)
plt.title('Age Distribution')
plt.show()
```

**2. Box Plot**
*   **Use:** Spot outliers and see quartiles clearly.
```python
sns.boxplot(x=df['salary'])
plt.title('Salary Spread')
plt.show()
```

### Categorical Features
**1. Count Plot (Bar Chart)**
*   **Use:** See frequency or imbalance of classes.
```python
# Order bars by frequency
sns.countplot(data=df, x='department', order=df['department'].value_counts().index)
plt.xticks(rotation=45)
plt.show()
```

---

## 4.2 Bivariate Analysis (Two Variables)
Analyzing relationships between variables.

### Numerical vs Numerical
**1. Scatter Plot**
*   **Use:** Check for linear/non-linear relationships and clusters.
```python
sns.scatterplot(data=df, x='age', y='salary', hue='department', alpha=0.6)
plt.title('Age vs Salary by Department')
plt.show()
```

**2. Correlation Heatmap**
*   **Use:** Rapidly identify correlated features involved in multicollinearity.
```python
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()
```

### Numerical vs Categorical
**1. Grouped Box Plot / Violin Plot**
*   **Use:** Compare distributions across groups.
```python
# Violin plot shows density better than box plot
sns.violinplot(data=df, x='department', y='salary')
plt.title('Salary Distribution per Department')
plt.show()
```

### Categorical vs Categorical
**1. Stacked Bar / Crosstab Heatmap**
*   **Use:** See how two categories interact.
```python
ct = pd.crosstab(df['department'], df['gender'])
sns.heatmap(ct, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Gender Count by Department')
plt.show()
```

---

## 4.3 Multivariate Analysis (3+ Variables)
Use "Hue", "Size", and "Style" to add dimensions to 2D plots.

```python
# x=Age, y=Salary, Hue=Gender, Size=YearsExperience
sns.relplot(data=df, x='age', y='salary', hue='gender', size='experience', 
            kind='scatter', height=6, aspect=1.5)
```

**Pair Plot:**
Automatically creates scatter plots for all numerical pairs.
```python
sns.pairplot(df, hue='churn_status')
```
*Note: Be careful running pairplot on very large datasets with many columns.*
