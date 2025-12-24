# Regression Metrics: When You're Always Wrong

**The Real Estate Pricing Model:**

You're building a model to estimate house values for a lending company.

**Model 1: Low Error on Average**
```
House A: Actual $500k, Predicted $520k (error: +$20k)
House B: Actual $450k, Predicted $430k (error: -$20k)
House C: Actual $2.5M, Predicted $900k (error: -$1.6M)

Mean Absolute Error (MAE): ($20k + $20k + $1.6M) / 3 = $547k
```

**What happened:** The model catastrophically failed on the luxury property but looked "okay" on average.

**Business impact:** Bank approves $2M loan on house worth $900k. Foreclosure. $1.1M loss.

---

## The Three Core Metrics

### 1. MAE (Mean Absolute Error): The Forgiving Metric

**Scenario:** Sales forecasting for inventory planning

```python
# Actual daily sales: [100, 105, 98, 102, 500]  # Last day was Black Friday
# Predicted: [98, 107, 95, 99, 120]

mae = mean_absolute_error(actual, predicted)
# MAE = (2 + 2 + 3 + 3 + 380) / 5 = 78 units
```

**Why it's used:**
- Outliers (Black Friday) don't dominate the metric
- Interpretable: "On average, we're off by 78 units"
- **Good for:** Planning where occasional big misses are acceptable

**Limitation:** Underpenalizes catastrophic errors

### 2. RMSE (Root Mean Squared Error): The Punisher

**Same data:**
```python
# Squares the errors before averaging
rmse = sqrt(mean_squared_error(actual, predicted))
# RMSE = sqrt((4 + 4 + 9 + 9 + 144,400) / 5) = 170 units
```

**Why it's used:**
- Heavily penalizes large errors (380² = 144,400)
- **Good for:** Systems where big errors are catastrophic (medical dosing, structural engineering)
- Standard in Kaggle competitions (forces you to fix outliers)

**Limitation:** Hard to interpret ("Root of squared units"?)

### 3. R² (R-Squared): The Comparison Metric

**Interpretation:** "How much better am I than just predicting the average?"

- R² = 1.0: Perfect predictions
- R² = 0.0: You're as good as predicting the mean every time
- R² < 0.0: **You're worse than predicting the mean** (you're actively harmful)

```python
# Baseline (predict mean): $500k for every house
# Your model varies predictions
r2_score(y_true, y_pred)  # 0.73

# Meaning: Your model explains 73% of the variance
```

**Use:** Comparing models. "Model B (R²=0.81) beats Model A (R²=0.73)"

## 1. The Big Three

### A. MAE (Mean Absolute Error)
$$MAE = \frac{1}{n} \sum |y - \hat{y}|$$
*   **Interpretation:** "On average, I am wrong by \$5."
*   **Pro:** Very robust. Outliers don't destroy it.
*   **Con:** Harder to use in calculus (gradient descent) because $|x|$ is not differentiable at 0.

### B. RMSE (Root Mean Squared Error)
$$RMSE = \sqrt{\frac{1}{n} \sum (y - \hat{y})^2}$$
*   **Interpretation:** Standard Deviation of the error.
*   **Pro:** Penalizes large errors **Quadratically**.
    *   Error 2 $\rightarrow$ Penalty 4.
    *   Error 10 $\rightarrow$ Penalty 100.
*   **Use when:** Large errors are unacceptable (e.g., Safety systems).

### C. R-Squared ($R^2$)
$$R^2 = 1 - \frac{\text{Unexplained Variance}}{\text{Total Variance}}$$
*   **Interpretation:** "I can explain 80% of the movement in the target."
*   **Range:** -Infinity to 1.
*   **Warning:** You can get negative $R^2$ if your model is worse than drawing a horizontal line.

---

## 2. Advanced: RMSLE (Log Error)

**Root Mean Squared Logarithmic Error.**
Basically, take the log of the numbers before calculating RMSE.

$$RMSLE = \sqrt{\frac{1}{n} \sum (\log(y+1) - \log(\hat{y}+1))^2}$$

*   **Feature 1:** It cares about **Ratio**. Predicting 100 instead of 50 is the same error as predicting 1000 instead of 500.
*   **Feature 2:** It penalizes **Underestimation** more than Overestimation.
*   **Use Case:** Forecasting sales, inventory, website traffic.

```python
from sklearn.metrics import mean_squared_log_error
import numpy as np

# y_true, y_pred
print(np.sqrt(mean_squared_log_error(y_test, y_pred)))
```

---

## 3. Which one to choose?

| Scenario | Metric | Why? |
|----------|--------|------|
| **General Purpose** | **MAE** | Easy to explain to boss. |
| **"Don't be hugely wrong"** | **RMSE** | Punishes outliers. |
| **"Predicting Growth/Sales"** | **RMSLE** | Cares about % growth. |
| **"Comparing Models"** | **$R^2$** | Normalized score (0-1). |
