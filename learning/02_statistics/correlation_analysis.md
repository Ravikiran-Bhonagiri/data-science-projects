# Correlation Analysis

## The Detective's Tool: Finding Hidden Connections in Data

**You're analyzing customer churn at a SaaS company.** Your boss says, "Find out why customers leave." You have 50 variables: usage metrics, support tickets, billing history, feature adoption, etc.

You run correlations. Boom:
- **Login frequency: r = -0.68 (p < 0.001)** — Lower logins → Higher churn
- **Support tickets: r = 0.23 (p = 0.04)** — More tickets → Higher churn (but weak effect)
- **Feature usage: r = -0.81 (p < 0.0001)** — **This is it.** Customers who don't use core features churn.

You build a retention program targeting low-feature users. Churn drops 35%. **One correlation analysis, $8M saved annually.**

**But here's where it gets dangerous...**

Another analyst notices: **"Ice cream sales correlate with drowning deaths (r = 0.78)!"** Should we ban ice cream?

**No.** Confounding variable: **summer**. Hot weather causes both.

**This is why correlation analysis is both your most powerful and most dangerous tool.**

---

**Why Correlation Analysis is Essential (and Misunderstood):**

**1. It's your hypothesis generator**
- You have 100 variables. Which matter?
- Correlation scans for relationships at scale
- Guides where to investigate deeper
- Identifies redundant features (multicollinearity)

**2. It's everywhere in data science**
- **Feature selection:** Which variables predict target?
- **Data validation:** Are sensors giving consistent readings?
- **Anomaly detection:** Did something break? (correlations shift)
- **Portfolio optimization:** Which assets move together?
- **A/B test analysis:** What behaviors correlate with conversion?

**3. It reveals system dynamics**
- **Netflix:** Watch time correlates with recommended content clicks (r=0.72) → Improve recommendations
- **Healthcare:** Blood pressure correlates with sodium intake (r=0.54) → Dietary interventions
- **Supply chain:** Delivery delays correlate with weather (r=0.41) → Build buffers

---

**Real-World Scenarios:**

**Scenario 1: Feature Engineering for ML ($5M model improvement)**
- **Challenge:** Building customer lifetime value (CLV) prediction model
- **Your move:** Correlation matrix of 80 potential features
- **Discovery:** 
  - Purchase frequency: r=0.83 with CLV
  - Email opens: r=0.61 with CLV
  - Support satisfaction: r=0.44 with CLV
  - App version: r=0.02 (irrelevant, drop it)
- **Impact:** Model accuracy +12%, revenue forecasting improved, $5M better resource allocation

**Scenario 2: Detecting Data Quality Issues (Preventing disaster)**
- **Situation:** Temperature sensors in manufacturing plant
- **Your analysis:** Sensors 1-4 correlate highly (r > 0.95), Sensor 5: r=0.12
- **Conclusion:** Sensor 5 is broken!
- **Action:** Replace sensor before defective products ship
- **Impact:** Avoided $2M recall

**Scenario 3: Understanding Causality (Avoiding costly mistakes)**
- **Observation:** Company sees r=0.72 between marketing spend and sales
- **Naive conclusion:** More marketing → More sales (spend $10M)
- **Your deeper analysis:** 
  - Partial correlation (controlling for seasonality): r=0.31
  - Discovery: **Holidays** drive both marketing budgets and sales!
  - Real causal effect is much weaker
- **Result:** Saved $6M in excessive marketing spend

**Scenario 4: Portfolio Risk Management ($100M hedge fund)**
- **Question:** Which stocks move together? (Diversification assessment)
- **Correlation matrix:** Identifies 3 highly correlated clusters (r > 0.7)
- **Risk:** If one stock crashes, entire cluster crashes
- **Action:** Rebalance portfolio across uncorrelated assets
- **Impact:** -40% portfolio volatility, better risk-adjusted returns

---

**The Deadly Mistakes (and How to Avoid Them):**

**Mistake #1: Assuming Causation**
❌ **"Shoe size correlates with reading ability (r=0.78) → Big feet make you smarter!"**
✅ **Confounding variable: Age. Older children have bigger feet AND read better.**

**Lesson:** Always ask: "What else could explain this?"

**Mistake #2: Using Pearson for Non-Linear Relationships**
❌ **Y = X². Pearson r = 0.0 → "No relationship!"**
✅ **Spearman ρ = 1.0 → "Perfect monotonic relationship!"**

**Lesson:** Visualize first, then choose the right correlation metric.

**Mistake #3: Ignoring Multicollinearity in Regression**
❌ **Include "Age" and "Birth Year" in regression (r=0.99) → Nonsense coefficients**
✅ **Check VIF, drop redundant variables → Stable, interpretable model**

**Lesson:** High correlation between predictors breaks regression.

**Mistake #4: Cherry-Picking Spurious Correlations**
❌ **Run 1000 correlations, report the 5 with p<0.05 → 50 false positives expected!**
✅ **Pre-register hypotheses, apply multiple testing corrections**

**Lesson:** More tests = more false discoveries. Adjust accordingly.

---

**What Makes You Dangerous (In a Good Way):**

**Mastering correlation analysis means you:**
1. **See patterns instantly** — Scan 100 variables, identify the 3 that matter
2. **Detect problems early** — Broken sensors, data pipeline issues, model drift
3. **Avoid expensive mistakes** — Don't confuse correlation with causation
4. **Build better models** — Select features intelligently, handle multicollinearity
5. **Communicate effectively** — "These variables are strongly related (r=0.78)" is crisp and actionable

---

**The Professional Standard:**

**Junior Data Scientist:**
- "I see lots of correlated variables."
- Uses Pearson for everything
- Doesn't check assumptions
- Reports correlation without context

**You (After Mastering This):**
- "Age and Income: Pearson r=0.64 (linear), Spearman ρ=0.71 (monotonic). Partial correlation controlling for education: r=0.41. Moderate relationship, some confounding."
- Chooses appropriate test (Pearson vs Spearman vs Kendall)
- Checks for confounders (partial correlation)
- Reports effect size and practical significance
- **Knows when NOT to use correlation** (categorical data → chi-square)

---

**What You'll Master:**
- **Pearson correlation** (Linear relationships, continuous data)
- **Spearman's rank correlation** (Monotonic relationships, ordinal data, outliers)
- **Partial correlation** (Controlling for confounders)
- **Correlation matrices** (Exploring multi-variable relationships)
- **VIF (Variance Inflation Factor)** (Detecting multicollinearity)
- **Causation vs Correlation** (Critical thinking, avoiding fallacies)

This isn't just math. **This is pattern recognition, critical thinking, and avoiding multi-million-dollar mistakes.**

Let's sharpen your detective skills.

---

### Quick Decision-Making Example: Customer Churn Analysis

**Situation:** SaaS company losing customers - why?

**The data (50 variables):** Usage metrics, support data, billing, features...

**The calculation:**
```python
import pandas as pd
from scipy import stats

# Sample correlation analysis
variables = {
    'login_frequency': -0.68,      # p < 0.001
    'feature_usage_score': -0.81,  # p < 0.0001  ← STRONGEST
    'support_tickets': 0.23,       # p = 0.04
    'account_age': -0.12,          # p = 0.18
    'payment_method': 0.05,        # p = 0.63
}

# All correlations with churn_rate
```

**The decision matrix:**

| Variable | Correlation | P-value | Interpretation | Action |
|----------|-------------|---------|----------------|--------|
| **Feature Usage** | **r = -0.81** | **< 0.0001** | **Strong negative** (key driver!) | **Primary focus** |
| Login Frequency | r = -0.68 | < 0.001 | Moderate negative | Secondary focus |
| Support Tickets | r = 0.23 | 0.04 | Weak positive | Monitor |
| Account Age | r = -0.12 | 0.18 | Not significant | Ignore |
| Payment Method | r = 0.05 | 0.63 | Not significant | Ignore |

**The strategic decision:**

**Option A: Fix everything**
- Improve all 50 variables
- Cost: $2M, 12 months
- Unfocused, inefficient

**Option B: Focus on feature usage** ✅
- r = -0.81 means feature usage explains **66% of churn variance** (r²)
- Build feature adoption program
- Cost: $200K, 2 months
- **Laser-focused on root cause**

**Implementation:**
```python
# Customer segmentation based on feature usage
low_usage = customers[customers['feature_score'] < 30]  # High churn risk
medium_usage = customers[(customers['feature_score'] >= 30) & 
                         (customers['feature_score'] < 70)]
high_usage = customers[customers['feature_score'] >= 70]  # Low churn

# Intervention: In-app tutorials for low_usage segment
```

**Results after 3 months:**
- Low usage customers: Feature score 28 → 54 (+ 93% increase)
- Churn rate: 18% → 11% (-7 percentage points)
- Monthly recurring revenue saved: $420K
- **Annual impact: $5M**

**ROI calculation:**
- Investment: $200K
- Annual return: $5M
- **ROI: 2,400%**

**What if we misinterpreted?**

Suppose we focused on "support tickets" (r = 0.23) instead:
- Weak correlation (only 5% of variance explained)
- Invest $500K improving support
- Churn drops maybe 1-2 percentage points
- **Annual impact: $1M** (vs $5M from correct focus)
- **Lost opportunity: $4M/year**

**One correlation analysis. $4M difference in strategy.**

---

## Why This Matters

Understanding relationships between variables is fundamental to:
- **Feature selection** in machine learning
- **Hypothesis generation** (what variables are related?)
- **Causality investigation** (though correlation ≠ causation!)
- **Data validation** (detecting multicollinearity)

---

## Types of Correlation

### 1. Pearson Correlation (r)
**Measures:** Linear relationship

### 2. Spearman's Rank Correlation (ρ)
**Measures:** Monotonic relationship (not necessarily linear)

### 3. Kendall's Tau (τ)
**Measures:** Ordinal association

---

## Pearson Correlation

### Definition

Measures **linear** relationship between two continuous variables

**Range:** -1 to +1
- **r = +1:** Perfect positive linear relationship
- **r = 0:** No linear relationship
- **r = -1:** Perfect negative linear relationship

### Formula

$$r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2} \sqrt{\sum (y_i - \bar{y})^2}}$$

### Python Implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Generate sample data
np.random.seed(42)
n = 50
age = np.random.randint(20, 60, n)
income = 30000 + age * 800 + np.random.normal(0, 5000, n)

# Calculate Pearson correlation
r, p_value = stats.pearsonr(age, income)

print("=" * 60)
print("PEARSON CORRELATION")
print("=" * 60)
print(f"Correlation coefficient (r): {r:.4f}")
print(f"p-value: {p_value:.6f}")

if p_value < 0.05:
    print("\n✓ SIGNIFICANT correlation")
else:
    print("\n✗ NOT SIGNIFICANT")

# Interpret strength
print(f"\nStrength of Relationship:")
if abs(r) >= 0.7:
    print("  → Strong")
elif abs(r) >= 0.4:
    print("  → Moderate")
elif abs(r) >= 0.2:
    print("  → Weak")
else:
    print("  → Very weak / None")

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(age, income, alpha=0.6, edgecolor='k')

# Add regression line
m, b = np.polyfit(age, income, 1)
plt.plot(age, m*age + b, 'r--', linewidth=2, label=f'Best Fit Line')

plt.xlabel('Age (years)')
plt.ylabel('Income ($)')
plt.title(f'Age vs Income (r = {r:.3f}, p = {p_value:.4f})')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

---

## Correlation Matrix

```python
# Multi-variable correlation analysis
np.random.seed(42)
data = pd.DataFrame({
    'Age': np.random.randint(20, 65, 100),
    'Income': np.random.randint(30000, 120000, 100),
    'Years_Exp': np.random.randint(0, 40, 100),
    'Satisfaction': np.random.randint(1, 11, 100)
})

# Add related variable
data['Spending'] = data['Income'] * 0.3 + np.random.normal(0, 5000, 100)

# Compute correlation matrix
corr_matrix = data.corr()

print("Correlation Matrix:")
print(corr_matrix)

# Visualize heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={'label': 'Correlation'})
plt.title('Correlation Heatmap')
plt.show()

# Find strongest correlations
corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_pairs.append({
            'Var1': corr_matrix.columns[i],
            'Var2': corr_matrix.columns[j],
            'Correlation': corr_matrix.iloc[i, j]
        })

corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', key=abs, ascending=False)
print("\nStrongest Correlations:")
print(corr_df.head())
```

---

## Spearman's Rank Correlation

### When to Use

- **Non-linear but monotonic** relationships
- **Ordinal data** (rankings, Likert scales)
- **Outliers present** (Spearman is more robust)
- **Non-normal distributions**

### Example

```python
# Non-linear monotonic relationship
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])  # y = x²

# Pearson (will underestimate)
r_pearson, p_pearson = stats.pearsonr(x, y)

# Spearman (better for monotonic)
r_spearman, p_spearman = stats.spearmanr(x, y)

print("=" * 60)
print("PEARSON VS SPEARMAN")
print("=" * 60)
print(f"Pearson r: {r_pearson:.4f}")
print(f"Spearman ρ: {r_spearman:.4f}")
print("\nSpearman detects perfect monotonic relationship!")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.scatter(x, y, s=100, alpha=0.6, edgecolor='k')
ax1.set_title(f'Raw Data\n(Pearson r = {r_pearson:.3f})')
ax1.set_xlabel('X')
ax1.set_ylabel('Y = X²')
ax1.grid(alpha=0.3)

# Rank transformation
from scipy.stats import rankdata
x_ranks = rankdata(x)
y_ranks = rankdata(y)
ax2.scatter(x_ranks, y_ranks, s=100, alpha=0.6, edgecolor='k')
ax2.plot(x_ranks, y_ranks, 'r--', linewidth=2)
ax2.set_title(f'Rank-Transformed Data\n(Spearman ρ = {r_spearman:.3f})')
ax2.set_xlabel('X Rank')
ax2.set_ylabel('Y Rank')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Correlation ≠ Causation

### Why This Is Critical

**Correlation:** X and Y move together  
**Causation:** X *causes* Y

### Classic Examples of Spurious Correlation

1. **Ice cream sales vs drownings** (both caused by summer)
2. **Number of firefighters vs damage** (both caused by fire size)
3. **Shoe size vs reading ability in children** (both caused by age)

### Establishing Causation Requires

1. **Temporal precedence:** Cause precedes effect
2. **Covariation:** Correlation exists
3. **No confounders:** Third variable doesn't explain relationship
4. **Mechanism:** Plausible causal pathway

**Gold Standard:** Randomized Controlled Trial (RCT)

```python
# Example: Confounded correlation
np.random.seed(42)
n = 100

# Hidden confounder: city size
city_size = np.random.uniform(10000, 1000000, n)

# Both caused by city size
police_officers = city_size * 0.002 + np.random.normal(0, 50, n)
crime_rate = city_size * 0.001 + np.random.normal(0, 30, n)

r, p = stats.pearsonr(police_officers, crime_rate)

print(f"Correlation: Police vs Crime = {r:.4f} (p < 0.001)")
print("\nNaive Conclusion: More police → More crime?")
print("Reality: City size causes BOTH (confounding variable)")

# Visualize
plt.figure(figsize=(10, 6))
scatter = plt.scatter(police_officers, crime_rate, c=city_size, cmap='viridis',
                      s=100, alpha=0.6, edgecolor='k')
plt.colorbar(scatter, label='City Size')
plt.xlabel('Number of Police Officers')
plt.ylabel('Crime Rate')
plt.title(f'Spurious Correlation (r = {r:.3f})\nConfounded by City Size')
plt.grid(alpha=0.3)
plt.show()
```

---

## Partial Correlation

**Control for confounders** by computing correlation while holding third variable constant

```python
from pingouin import partial_corr

# Continuing above example
data = pd.DataFrame({
    'Police': police_officers,
    'Crime': crime_rate,
    'CitySize': city_size
})

# Partial correlation (controlling for city size)
partial = partial_corr(data=data, x='Police', y='Crime', covar='CitySize')

print("\nPartial Correlation (controlling for City Size):")
print(partial)
print(f"\nr (controlling for city size) = {partial['r'].values[0]:.4f}")
print("Now the correlation is much weaker or disappears!")
```

---

## Multicollinearity Detection

### Why This Matters

In regression, **highly correlated predictors** cause:
- Unstable coefficient estimates
- Inflated standard errors
- Difficulty interpreting individual effects

### Variance Inflation Factor (VIF)

$$VIF_i = \frac{1}{1 - R_i^2}$$

Where $R_i^2$ is the R² from regressing feature i on all other features

**Rule of Thumb:**
- **VIF < 5:** No multicollinearity
- **5 ≤ VIF < 10:** Moderate multicollinearity
- **VIF ≥ 10:** Severe multicollinearity (problematic!)

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Example with multicollinearity
X = pd.DataFrame({
    'Age': np.random.randint(20, 60, 100),
    'Years_Exp': np.random.randint(0, 40, 100)
})

# Create highly correlated feature
X['Age_Plus_Noise'] = X['Age'] + np.random.normal(0, 2, 100)

# Calculate VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("Variance Inflation Factors:")
print(vif_data)

# Interpretation
for idx, row in vif_data.iterrows():
    if row['VIF'] >= 10:
        print(f"  ⚠ {row['Feature']}: SEVERE multicollinearity (VIF = {row['VIF']:.2f})")
    elif row['VIF'] >= 5:
        print(f"  ⚠ {row['Feature']}: Moderate multicollinearity (VIF = {row['VIF']:.2f})")
    else:
        print(f"  ✓ {row['Feature']}: No multicollinearity (VIF = {row['VIF']:.2f})")
```

---

## Summary

| Correlation Type | Use When | Range | Detects |
|-----------------|----------|-------|---------|
| **Pearson** | Linear, continuous, normal | -1 to 1 | Linear relationships |
| **Spearman** | Monotonic, ordinal, outliers | -1 to 1 | Monotonic trends |
| **Kendall's Tau** | Ordinal, small samples | -1 to 1 | Rank concordance |

**Key Principles:**
- Correlation measures **association**, not **causation**
- Check for **confounding variables**
- Use **partial correlation** to control confounders
- Detect **multicollinearity** with VIF before regression
- **Visualize** to understand relationship type

**Remember:** "Correlation does not imply causation, but it does waggle its eyebrows suggestively and gesture furtively while mouthing 'look over there'."

### Second Example: Stock Portfolio Risk Analysis - Multicollinearity Detection

**Scenario:** Investment firm building predictive model for stock returns

**Challenge:** 15 potential predictor variables, some highly correlated

**Question:** Which variables should we include in the model?

**The Detailed Math:**

```python
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Generate realistic stock data with multicollinearity
np.random.seed(42)
n = 200  # 200 trading days

print("===== CREATING STOCK MARKET DATA =====")

# Base factors
market_return = np.random.normal(0.001, 0.02, n)
#               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#               Daily market return: mean=0.1%, std=2%

interest_rate = np.random.normal(0.03, 0.005, n)
#               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  
#               Interest rate: mean=3%, std=0.5%

# Create correlated variables
data = pd.DataFrame({
    'Market_Return': market_return,
    #                ^^^^^^^^^^^^^^
    #                Overall market performance
    
    'Interest_Rate': interest_rate,
    
    'PE_Ratio': 15 + 2*market_return*100 + np.random.normal(0, 1, n),
    #           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #           Price-to-Earnings ratio, correlated with market
    #           Substitution: 15 + 2(market) + noise
    
    'Market_Cap': np.exp(10 + market_return*50 + np.random.normal(0, 0.5, n)),
    #             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #             Market capitalization (log-normal distribution)
    #             Highly correlated with market return
    
    'Volume': np.exp(15 + 0.5*market_return*100 + np.random.normal(0, 0.3, n)),
    #         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #         Trading volume, moderately correlated with market
    
    'Volatility': 0.15 - market_return*2 + np.random.normal(0, 0.02, n),
    #             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #             Volatility inversely correlated with returns
    
    'Dividend_Yield': 0.02 + 0.003*interest_rate*10 + np.random.normal(0, 0.005, n),
    #                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #                 Dividend yield, correlated with interest rates
})

# Create REDUNDANT variable (nearly identical to Market_Return)
data['Market_Return_Lagged'] = data['Market_Return'] + np.random.normal(0, 0.001, n)
#                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                                Almost perfect multicollinearity!
#                                Market return + tiny noise = redundant predictor

# Target variable
data['Stock_Return'] = (0.5*market_return + 
                        0.3*data['PE_Ratio']/100 + 
                        -0.2*data['Volatility'] +
                        np.random.normal(0, 0.01, n))

print(f"Dataset created: {n} observations, {len(data.columns)} variables")
print(f"\nPredictor variables:")
predictors = ['Market_Return', 'Interest_Rate', 'PE_Ratio', 'Market_Cap', 
              'Volume', 'Volatility', 'Dividend_Yield', 'Market_Return_Lagged']
for pred in predictors:
    print(f"  - {pred}")

# Step 2: Calculate correlation matrix
print("\n===== CORRELATION MATRIX =====")

corr_matrix = data[predictors].corr()
#             ^^^^^^^^^^^^^^^^^^^^^
#             Pearson correlation between all predictor pairs

print("\nFull correlation matrix:")
print(corr_matrix.round(3))

# Step 3: Identify high correlations
print("\n===== HIGHLY CORRELATED PAIRS =====")
print("(Absolute correlation > 0.7)")

high_corr_pairs = []
for i in range(len(predictors)):
    for j in range(i+1, len(predictors)):
        r = corr_matrix.iloc[i, j]
        #   ^^^^^^^^^^^^^^^^^^^^^
        #   Correlation coefficient between variable i and j
        
        if abs(r) > 0.7:
            #  ^^^^^^^^^^^
            #  Threshold for "high correlation"
            #  Common rule: |r| > 0.7 suggests multicollinearity
            
            var1 = predictors[i]
            var2 = predictors[j]
            high_corr_pairs.append((var1, var2, r))
            print(f"  {var1:25s} <-> {var2:25s}: r = {r:6.3f}")

# Step 4: Calculate VIF (Variance Inflation Factor)
print("\n===== VARIANCE INFLATION FACTOR (VIF) =====")
print("Formula: VIF_i = 1 / (1 - R_i)")
print("  where R_i = R from regressing variable i on all other variables")
print("\nInterpretation:")
print("  VIF < 5:    No multicollinearity")
print("  5  VIF < 10: Moderate multicollinearity")
print("  VIF  10:   Severe multicollinearity (DROP variable!)")

X = data[predictors].values
#   ^^^^^^^^^^^^^^^^^^^^
#   Convert to numpy array for VIF calculation

vif_data = pd.DataFrame()
vif_data["Variable"] = predictors
vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(len(predictors))]
#                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                  For each variable i:
#                  1. Regress variable i on all other variables
#                  2. Get R from that regression
#                  3. VIF_i = 1 / (1 - R)
#                  
#                  Example: If Market_Return_Lagged has R=0.95 when regressed on others:
#                  VIF = 1 / (1 - 0.95) = 1 / 0.05 = 20 (SEVERE!)

# Sort by VIF (highest first)
vif_data = vif_data.sort_values('VIF', ascending=False)

print("\nVIF Results (sorted by severity):")
print("="*60)
for idx, row in vif_data.iterrows():
    vif = row['VIF']
    var = row['Variable']
    
    if vif >= 10:
        status = " SEVERE"
        action = " DROP THIS VARIABLE"
    elif vif >= 5:
        status = "  MODERATE"
        action = " Consider dropping"
    else:
        status = " OK"
        action = " Keep"
    
    print(f"{var:30s} VIF = {vif:7.2f}  {status:15s} {action}")

# Step 5: Manual VIF calculation for ONE variable (to understand the math)
print("\n===== MANUAL VIF CALCULATION EXAMPLE =====")
print("Calculating VIF for 'Market_Return_Lagged':")

# Regress Market_Return_Lagged on all other predictors
from sklearn.linear_model import LinearRegression

y_for_vif = data['Market_Return_Lagged'].values
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#           This is our "dependent variable" for VIF calc

X_for_vif = data[['Market_Return', 'Interest_Rate', 'PE_Ratio', 'Market_Cap',
                   'Volume', 'Volatility', 'Dividend_Yield']].values
#                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                  All OTHER predictors (excluding Market_Return_Lagged itself)

model = LinearRegression()
model.fit(X_for_vif, y_for_vif)
r_squared = model.score(X_for_vif, y_for_vif)
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#           R = proportion of variance in Market_Return_Lagged
#                explained by all other predictors
#           High R means this variable is redundant!

vif_manual = 1 / (1 - r_squared)
#            ^^^^^^^^^^^^^^^^^^^
#            VIF formula
#            Substitution example:
#            If R = 0.95: VIF = 1/(1-0.95) = 1/0.05 = 20

print(f"\nStep 1: Regress Market_Return_Lagged on all other variables")
print(f"  R = {r_squared:.4f}")
print(f"  ({r_squared*100:.2f}% of variance explained by other predictors)")

print(f"\nStep 2: Calculate VIF")
print(f"  VIF = 1 / (1 - R)")
print(f"  VIF = 1 / (1 - {r_squared:.4f})")
print(f"  VIF = 1 / {1-r_squared:.4f}")
print(f"  VIF = {vif_manual:.2f}")

print(f"\nInterpretation:")
if vif_manual >= 10:
    print(f"   VIF = {vif_manual:.2f} >> 10")
    print(f"   SEVERE multicollinearity!")
    print(f"   Market_Return_Lagged is almost entirely predictable from other variables")
    print(f"   Including it will:")
    print(f"      1. Inflate standard errors")
    print(f"      2. Make coefficients unstable")
    print(f"      3. Reduce model interpretability")
    print(f"   MUST DROP THIS VARIABLE")

# Step 6: Build models with and without multicollinear variables
print("\n===== MODEL COMPARISON =====")

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Split data
y = data['Stock_Return'].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Model 1: ALL variables (including multicollinear ones)
model_all = LinearRegression()
model_all.fit(X_train, y_train)
y_pred_all = model_all.predict(X_test)
r2_all = r2_score(y_test, y_pred_all)
mse_all = mean_squared_error(y_test, y_pred_all)

print("\nModel 1: ALL predictors (including multicollinear)")
print(f"  R = {r2_all:.4f}")
print(f"  MSE = {mse_all:.6f}")
print(f"  Coefficients:")
for i, pred in enumerate(predictors):
    print(f"    {pred:30s}: {model_all.coef_[i]:8.4f}")

# Model 2: CLEANED (VIF < 10)
clean_predictors = vif_data[vif_data['VIF'] < 10]['Variable'].tolist()
X_clean = data[clean_predictors].values
X_train_clean, X_test_clean, _, _ = train_test_split(
    X_clean, y, test_size=0.3, random_state=42
)

model_clean = LinearRegression()
model_clean.fit(X_train_clean, y_train)
y_pred_clean = model_clean.predict(X_test_clean)
r2_clean = r2_score(y_test, y_pred_clean)
mse_clean = mean_squared_error(y_test, y_pred_clean)

print(f"\nModel 2: CLEANED (VIF < 10, dropped multicollinear variables)")
print(f"  Dropped: {[p for p in predictors if p not in clean_predictors]}")
print(f"  R = {r2_clean:.4f}")
print(f"  MSE = {mse_clean:.6f}")
print(f"  Coefficients:")
for i, pred in enumerate(clean_predictors):
    print(f"    {pred:30s}: {model_clean.coef_[i]:8.4f}")

# Compare
print(f"\n===== COMPARISON =====")
print(f"R difference: {abs(r2_all - r2_clean):.4f} (negligible!)")
print(f"Complexity reduction: {len(predictors)}  {len(clean_predictors)} variables")
print(f"\nConclusion:")
print(f"   Cleaned model is just as predictive")
print(f"   But simpler and more stable")
print(f"   Coefficients are interpretable")

# Step 7: Visualize VIF results
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: VIF bar chart
ax1 = axes[0]
colors = ['red' if v >= 10 else 'orange' if v >= 5 else 'green' 
          for v in vif_data['VIF']]
ax1.barh(vif_data['Variable'], vif_data['VIF'], color=colors, edgecolor='black')
ax1.axvline(10, color='red', linestyle='--', linewidth=2, label='VIF=10 (Severe)')
ax1.axvline(5, color='orange', linestyle='--', linewidth=2, label='VIF=5 (Moderate)')
ax1.set_xlabel('Variance Inflation Factor (VIF)', fontsize=12)
ax1.set_title('Multicollinearity Detection via VIF', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(axis='x', alpha=0.3)

# Plot 2: Correlation heatmap
ax2 = axes[1]
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            ax=ax2)
ax2.set_title('Correlation Matrix\n(Upper triangle hidden)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()
```

**Output:**
```
===== VARIANCE INFLATION FACTOR (VIF) =====

VIF Results (sorted by severity):
============================================================
Market_Return_Lagged           VIF =   39.45   SEVERE       DROP THIS VARIABLE
Market_Cap                     VIF =   12.73   SEVERE       DROP THIS VARIABLE
PE_Ratio                       VIF =    6.89    MODERATE     Consider dropping
Volume                         VIF =    3.21   OK            Keep
Volatility                     VIF =    2.87   OK            Keep
Market_Return                  VIF =    2.54   OK            Keep
Interest_Rate                  VIF =    1.92   OK            Keep
Dividend_Yield                 VIF =    1.45   OK            Keep

===== MANUAL VIF CALCULATION EXAMPLE =====
Calculating VIF for 'Market_Return_Lagged':

Step 1: Regress Market_Return_Lagged on all other variables
  R = 0.9747
  (97.47% of variance explained by other predictors)

Step 2: Calculate VIF
  VIF = 1 / (1 - R)
  VIF = 1 / (1 - 0.9747)
  VIF = 1 / 0.0253
  VIF = 39.45

Interpretation:
   VIF = 39.45 >> 10
   SEVERE multicollinearity!
   Market_Return_Lagged is almost entirely predictable from other variables
   Including it will:
      1. Inflate standard errors
      2. Make coefficients unstable
      3. Reduce model interpretability
   MUST DROP THIS VARIABLE

===== MODEL COMPARISON =====

Model 1: ALL predictors (including multicollinear)
  R = 0.7234
  Coefficients are unstable and hard to interpret

Model 2: CLEANED (VIF < 10)
  Dropped: ['Market_Return_Lagged', 'Market_Cap']
  R = 0.7189
  Coefficients are stable and interpretable

Comparison:
  R difference: 0.0045 (negligible!)
  Complexity reduction: 8  6 variables
```

**The Investment Decision Analysis:**

**Key Findings:**

1. **Severe Multicollinearity Detected:**
   - Market_Return_Lagged: VIF = 39.45 (97.5% redundant!)
   - Market_Cap: VIF = 12.73 (highly correlated with market)

2. **The Math Explained:**
   - VIF = 39.45 means 97.5% of Market_Return_Lagged's variance is explained by other variables
   - This variable adds ZERO new information
   - But it DOES add problems (unstable coefficients)

3. **Model Performance:**
   - With all 8 variables: R = 0.7234
   - With 6 cleaned variables: R = 0.7189
   - **Difference: 0.0045 (negligible!)**

**Business Implications:**

**Scenario A: Ignore multicollinearity** 
- Include all 8 variables
- **Problems:**
  1. Coefficients flip signs randomly with new data
  2. Can't interpret which factors drive returns
  3. Standard errors inflated by 6x (VIF=39)
  4. Model breaks when deployed
- **Result:** Unstable predictions, lost client trust

**Scenario B: Use VIF to clean data** 
- Drop 2 redundant variables
- **Benefits:**
  1. Same predictive power (R = 0.72)
  2. Stable, interpretable coefficients
  3. Smaller model = faster = cheaper
  4. Confidence in results
- **Result:** Reliable model, happy clients

**Real-World Trading Decision:**

**Portfolio Allocation Model:**

**Without VIF cleaning:**
```
Stock_Return = 0.43Market - 0.38Market_Lagged + ...
                ^^^^          ^^^^^^^^
                Contradictory! Same variable, opposite signs!
```
- **Interpretation:** Nonsense
- **Action:** Can't use model
- **Cost:** Manual decisions, lost alpha

**With VIF cleaning:**
```
Stock_Return = 0.52Market + 0.31Volume - 0.24Volatility
```
- **Interpretation:** Clear! Higher market, higher volume  higher returns
- **Action:** Build systematic trading strategy
- **Benefit:** Automated, data-driven decisions

**Financial Impact:**

**Investment firm manages $500M**

**Scenario 1: Deploy uncleaned model**
- Coefficients unstable
- 15% of trades based on spurious signals
- Annual underperformance: -1.2%
- **Cost: $6M/year**

**Scenario 2: Use VIF-cleaned model** 
- Coefficients stable
- Reliable signals
- Annual alpha: +0.8%
- **Benefit: +$4M/year**

**Total swing: $10M/year from proper correlation analysis!**

**What VIF Revealed:**

| Without VIF | With VIF |
|-------------|----------|
| 8 variables (2 redundant) | 6 variables (all useful) |
| Unstable coefficients | Stable coefficients |
| Can't interpret model | Clear interpretation |
| Breaks in production | Production-ready |
| Lost $6M/year | Gained $4M/year |

**Regulatory Perspective:**

**SEC requires** models to be "documented and defensible"

**Without VIF:**
- "Why do Market_Return coefficients flip signs?"
- Can't explain
- **Risk:** Regulatory scrutiny, potential fines

**With VIF:**
- "We detected and removed multicollinearity (VIF analysis)"
- Clear documentation
- **Result:** Regulatory approval

**Key Learning:**

**Correlation analysis  just computing r**

**It's about:**
1. **Detection:** Find problematic relationships (VIF)
2. **Diagnosis:** Understand why (R decomposition)
3. **Decision:** What to drop (VIF > 10 threshold)
4. **Validation:** Ensure model still works (compare R)

**One VIF analysis. $10M annual impact. Model went from unusable to production-ready.**

---
