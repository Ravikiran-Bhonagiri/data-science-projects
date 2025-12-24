# Time Series EDA

Time series data has a **temporal ordering** that must be respected. Standard EDA techniques break when applied to sequential data. This guide covers specialized techniques for analyzing data over time.

---

## When to Use This
- Stock prices, crypto trading
- IoT sensor readings (temperature, pressure)
- Sales/revenue forecasting
- Website traffic analytics
- Medical vital signs monitoring

---

## 1. Stationarity Testing

**Why it matters:** Most time series models (ARIMA, GARCH) require stationary data.

**Stationary = No trend, constant variance, constant autocorrelation**

### Visual Check
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load time series
df = pd.read_csv('stock_prices.csv', parse_dates=['Date'], index_col='Date')

# Plot
df['Close'].plot(figsize=(12, 6))
plt.title('Stock Price Over Time')
plt.ylabel('Price ($)')
plt.show()
```
**What to look for:**
- Upward/downward trend? → Non-stationary
- Increasing variance? → Non-stationary
- Seasonal patterns? → Non-stationary

### Statistical Tests

**1. Augmented Dickey-Fuller (ADF) Test**
- **Null Hypothesis (H0):** Series has a unit root (is non-stationary).
- **Alternate Hypothesis (H1):** Series is stationary.

```python
from statsmodels.tsa.stattools import adfuller

result = adfuller(df['Close'])
print(f'ADF Statistic: {result[0]:.4f}')
print(f'p-value: {result[1]:.4f}')

if result[1] < 0.05:
    print("✅ Stationary (reject H0)")
else:
    print("❌ Non-Stationary (fail to reject H0)")
```

**2. KPSS Test**
- **Null Hypothesis (H0):** Series is trend-stationary (stationary around a deterministic trend).
- **Alternate Hypothesis (H1):** Series is non-stationary.

```python
from statsmodels.tsa.stattools import kpss

result = kpss(df['Close'], regression='c')
print(f'KPSS Statistic: {result[0]:.4f}')
print(f'p-value: {result[1]:.4f}')

if result[1] > 0.05:
    print("✅ Stationary (fail to reject H0)")
else:
    print("❌ Non-Stationary (reject H0)")
```

> [!TIP]
> **ADF vs KPSS Decision Table:**
> - Both say Stationary: Series is definitely stationary.
> - Both say Non-Stationary: Series is definitely non-stationary.
> - ADF Stationary, KPSS Non-Stationary: Difference Stationary (use differencing).
> - ADF Non-Stationary, KPSS Stationary: Trend Stationary (remove trend).

---

## 2. Autocorrelation & White Noise

### Autocorrelation plots (ACF/PACF)
```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_acf(df['Close_diff'].dropna(), lags=40, ax=axes[0])
plot_pacf(df['Close_diff'].dropna(), lags=40, ax=axes[1])
plt.show()
```

### Ljung-Box Test (Testing for White Noise)
If a series is "White Noise", it is completely random and contains no information for modeling.
```python
from statsmodels.stats.diagnostic import acorr_ljungbox

res = acorr_ljungbox(df['Close_diff'].dropna(), lags=[10], return_df=True)
print(res)
# If p-value > 0.05, the series is likely White Noise.
```

---

## 3. Handling Missing Values in Time Series
In time series, we cannot use simple mean/median imputation because it breaks the temporal sequence.

```python
# Create gaps
df_gap = df.copy()
# Advanced Interpolation Methods
df_gap['linear'] = df_gap['Close'].interpolate(method='linear')
df_gap['quadratic'] = df_gap['Close'].interpolate(method='quadratic')
df_gap['spline'] = df_gap['Close'].interpolate(method='spline', order=2)

# Visualization of Interpolation
df_gap[['linear', 'quadratic', 'spline']].plot(figsize=(12, 6))
```

---

## 4. Seasonal Decomposition (STL)
STL (Seasonal-Trend decomposition using LOESS) is more robust than standard decomposition.

```python
from statsmodels.tsa.seasonal import STL

stl = STL(df['Close'], period=30)
res = stl.fit()
res.plot()
plt.show()
```

---

## 5. Multivariate Time Series: Granger Causality
Does variable X "cause" (predict) variable Y?

```python
from statsmodels.tsa.stattools import grangercausalitytests

# Does 'Volume' cause 'Price'?
data = df[['Close', 'Volume']].dropna()
gc_res = grangercausalitytests(data, maxlag=5, verbose=True)
```

---

## 6. Time Series Cross-Validation
Standard k-fold cross-validation is illegal in time series (prevents leakage from the future).

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(df):
    print("TRAIN:", train_index, "TEST:", test_index)
```

---

## 4. Rolling Statistics

Track how mean and std dev change over time.

```python
# 30-day rolling mean and std
df['Rolling_Mean'] = df['Close'].rolling(window=30).mean()
df['Rolling_Std'] = df['Close'].rolling(window=30).std()

plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Original', alpha=0.7)
plt.plot(df['Rolling_Mean'], label='30-Day Rolling Mean', color='red')
plt.fill_between(df.index, 
                 df['Rolling_Mean'] - df['Rolling_Std'],
                 df['Rolling_Mean'] + df['Rolling_Std'], 
                 alpha=0.2, color='red')
plt.legend()
plt.title('Rolling Statistics')
plt.show()
```

---

## 5. Detecting Change Points

Find when the statistical properties of the series change abruptly.

```python
import ruptures as rpt

# Detect change points
signal_data = df['Close'].values
algo = rpt.Pelt(model="rbf").fit(signal_data)
result = algo.predict(pen=10)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df['Close'])
for cp in result[:-1]:
    plt.axvline(x=df.index[cp], color='r', linestyle='--', alpha=0.7)
plt.title('Change Point Detection')
plt.show()
```

---

## 6. Checking for Irregular Intervals

```python
# Check if timestamps are evenly spaced
time_diff = df.index.to_series().diff()
print(f"Most common interval: {time_diff.mode()[0]}")
print(f"Missing intervals: {(time_diff != time_diff.mode()[0]).sum()}")

# Resample to regular frequency
df_resampled = df.resample('D').mean()  # Daily
df_resampled = df_resampled.interpolate(method='linear')  # Fill gaps
```

---

## Checklist for Time Series EDA

- [ ] Plot the raw series (identify trends, seasonality)
- [ ] Test for stationarity (ADF, KPSS)
- [ ] Make stationary if needed (differencing, log transform)
- [ ] Check ACF/PACF (identify AR/MA components)
- [ ] Decompose (separate trend, seasonal, residual)
- [ ] Calculate rolling statistics
- [ ] Detect change points
- [ ] Check for irregular intervals
- [ ] Identify outliers (spikes, drops)
