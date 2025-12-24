# Linear Regression: When Simplicity Wins

**The Zillow $900M Mistake:**

Zillow's "Zestimate" (automated home valuation) used complex neural networks.

**2021 Problem:**
- Model predicted home prices
- Zillow used predictions to buy houses directly ("iBuying")
- Result: Bought 7,000 homes for $2.8B
- **Actual sale price:** $1.9B
- **Loss:** $900 million

**What went wrong:**
- Over-complicated model captured noise
- Assumed pandemic housing boom would continue
- Ignored simple linear trends (interest rates ↑ → prices ↓)

**A simple linear model tracking:**
```python
price = β₀ + β₁(sqft) + β₂(bedrooms) + β₃(interest_rate) + β₄(inventory)
```

Would have shown: "Interest rates rising = prices falling. Stop buying."

---

## Production Use Cases

### 1. **Advertising Budget Allocation**
**Goal:** Predict sales based on ad spend

```python
# Data: Past marketing campaigns
# Features: TV_spend, Digital_spend, Print_spend
# Target: Revenue

model = LinearRegression()
model.fit(X_train, y_train)

# Interpretation: 
print(f"$1 in TV ads → ${model.coef_[0]:.2f} revenue")
print(f"$1 in Digital → ${model.coef_[1]:.2f} revenue")
```

**Business value:** Know exactly which channel to invest in.

### 2. **Salary Prediction (HR Analytics)**
**Problem:** Setting fair compensation

- Feature: Years_experience, Education_level, Location
- Target: Salary
- **Why linear:** Explainability required for legal compliance

---

## 1. The Core Equation
$$y = \beta_0 + \beta_1 x_1 + ... + \beta_n x_n + \epsilon$$

- **Goal:** Minimize SSE (Sum of Squared Errors).
- **Optimizer:** OLS (Ordinary Least Squares) or Gradient Descent.

---

## 2. Assumptions (The "LINE" Test)
Linear models fail if these assumptions are violated.

1.  **L**inearity: Relationship is linear.
2.  **I**ndependence: No correlation between residuals (Durbin-Watson test).
3.  **N**ormality: Residuals are normally distributed (Q-Q Plot, Shapiro-Wilk).
4.  **E**qual Variance (Homoscedasticity): Residuals have constant variance (Breusch-Pagan test).

### Diagnostic Code (Statsmodels)
Scikit-learn predicts well, but `statsmodels` gives you the p-values and diagnostics needed for rigorous science.

```python
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Add intercept (Statsmodels doesn't add it by default)
X_train_const = sm.add_constant(X_train)
model_sm = sm.OLS(y_train, X_train_const).fit()

print(model_sm.summary())
# Look for: 
# - Adj. R-squared (Fit quality)
# - Prob (F-statistic) < 0.05 (Model significance)
# - P>|t| (Feature significance)
# - Durbin-Watson (~2.0 is good)
```

---

## 3. Regularization: Ridge vs Lasso
When you have **high multicollinearity** or **too many features**, standard OLS overfits.

### A. Ridge Regression (L2)
- Adds penalty equal to square of magnitude of coefficients.
- **Shrinks** coefficients towards zero, but not exactly zero.
- **Use when:** You want to keep all features but reduce overfitting.

$$Cost = SSE + \alpha \sum \beta^2$$

### B. Lasso Regression (L1)
- Adds penalty equal to absolute value of coefficients.
- **Eliminates** features (sets coefficients to zero).
- **Use when:** You want automatic feature selection.

$$Cost = SSE + \alpha \sum |\beta|$$

### C. ElasticNet
- Combination of both. Best of both worlds.

### Implementation
```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler

# 1. Scaling is MANDATORY for Regularization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Ridge (L2)
ridge = Ridge(alpha=1.0)  # Alpha = Strength of regularization
ridge.fit(X_train_scaled, y_train)

# 3. Lasso (L1)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)
print(f"Features eliminated by Lasso: {np.sum(lasso.coef_ == 0)}")
```

---

## 4. Polynomial Regression
When data is non-linear, you can still use Linear Regression by transforming features.

$$y = \beta_0 + \beta_1 x + \beta_2 x^2$$

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Create non-linear features (x^2, x^3)
poly = PolynomialFeatures(degree=2, include_bias=False)
model = make_pipeline(poly, LinearRegression())

model.fit(X_train, y_train)
```

---

## 5. Metrics for Regression

| Metric | Name | Interpretation |
|--------|------|----------------|
| **MAE** | Mean Absolute Error | Average error in real units ($). Robust to outliers. |
| **MSE** | Mean Squared Error | Penalizes large errors heavily. Hard to interpret units ($^2$). |
| **RMSE** | Root MSE | Average error in real units, but sensitive to large errors. |
| **R²** | Coeff. of Determination | % of variance explained (0 to 1). |
| **Adj R²**| Adjusted R² | R² penalized for useless features. Use this for feature selection. |

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
```
