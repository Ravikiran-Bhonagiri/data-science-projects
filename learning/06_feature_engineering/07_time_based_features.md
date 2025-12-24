# Time-Based Features: The Timestamp Goldmine

**The E-Commerce Disaster:**

You're predicting product sales for inventory planning.

**Model 1: Ignoring Time**
```python
features = ['price', 'category', 'brand']
model.fit(X_train, y_train)
# Test RMSE: 487 units
```

**Model 2: Adding Hour of Day**
```python
df['hour'] = df['timestamp'].dt.hour
features = ['price', 'category', 'brand', 'hour']
# Test RMSE: 201 units
```

**Why the massive improvement?** Sales patterns:
- 3 AM: 12 units/hour (people sleeping)
- 12 PM: 340 units/hour (lunch break browsing)
- 9 PM: 680 units/hour (couch shopping)

**The hour feature captured the entire daily cycle.**

---

## The Critical Mistake: Linear Time Encoding

**Wrong approach:**
```python
df['month'] = df['timestamp'].dt.month  # Jan=1, Dec=12
```

**The problem:** Model thinks December (12) is 12× more than January (1).
Worse: December and January are actually **neighbors** (holiday shopping season).

---

## 1. Cyclical Encoding (Sin/Cos Transform)

**The fix:** Represent time as points on a circle.

```python
import numpy as np

# Hour of day (0-23)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Day of week (0-6)
df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

# Month (1-12)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
```

**Why it works:**
- Hour 23 (11 PM) and Hour 0 (12 AM) are now close in feature space
- December and January are neighbors
- Preserves the cyclical nature

**Production use cases:**
- Retail (daily/weekly sales patterns)
- Traffic prediction (rush hour)
- Energy consumption (seasonal)

---

## 2. Lag Features (Past Values)

**Credit Card Fraud Detection:**

```python
# Transaction amount right now means nothing
# What matters: Is this DIFFERENT from your normal pattern?

# Create lag features
df['amount_lag_1'] = df.groupby('user_id')['amount'].shift(1)  # Previous transaction
df['amount_lag_7'] = df.groupby('user_id')['amount'].shift(7)  # 7 transactions ago

# Deviation from history
df['amount_change'] = df['amount'] - df['amount_lag_1']
df['amount_pct_change'] = df['amount'] / df['amount_lag_1']
```

**Real pattern detection:**
- User normally spends $40/transaction
- Suddenly: $2,400 purchase
- `amount_pct_change = 60x` → **Flag as fraud**

---

## 3. Rolling Window Features (Moving Aggregates)

**Stock Trading Algorithm:**

```python
# Last 7 days of prices
df['price_rolling_7d_mean'] = df['price'].rolling(window=7).mean()
df['price_rolling_7d_std'] = df['price'].rolling(window=7).std()

# Volatility indicator
df['price_volatility'] = df['price_rolling_7d_std'] / df['price_rolling_7d_mean']

# Trend indicator
df['price_trend'] = df['price'] - df['price_rolling_7d_mean']
```

**Business logic:**
- High volatility + positive trend = Momentum buying opportunity
- Low volatility + price < rolling_mean = Accumulation zone

**Production result:** Algorithmic trading fund uses this for entry signals.

---

## 4. Time Since Event

**Subscription Churn Prediction:**

```python
df['days_since_signup'] = (df['current_date'] - df['signup_date']).dt.days
df['days_since_last_purchase'] = (df['current_date'] - df['last_purchase_date']).dt.days
df['days_since_support_ticket'] = (df['current_date'] - df['last_ticket_date']).dt.days
```

**Key patterns:**
- 0-30 days since signup: 2% churn (honeymoon period)
- 60-90 days since signup: 18% churn (critical evaluation period)
- 180+ days since last purchase: 45% churn (they forgot about you)

**Action:** Target users at day 55 with re-engagement campaign.

---

## 5. Special Day Flags (Binary Indicators)

**Retail Sales Forecasting:**

```python
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['is_month_end'] = (df['day'] > 25).astype(int)
df['is_payday'] = df['day'].isin([1, 15]).astype(int)  # Bi-weekly payday

# Holidays
df['is_black_friday'] = ((df['month'] == 11) & (df['day'] >= 23) & (df['day'] <= 29)).astype(int)
df['is_cyber_monday'] = ...
```

**Impact:** Black Friday sales are 15× normal. Without the flag, model averages it out and fails catastrophically.

---

## Production Checklist

| Time Pattern | Feature Type | Use Case |
|--------------|--------------|----------|
| **Daily cycle** | Sin/Cos Hour | Web traffic, sales, energy |
| **Weekly cycle** | Sin/Cos Day-of-Week | B2B sales, subscriptions |
| **Seasonal** | Sin/Cos Month | Retail, agriculture, tourism |
| **Recent history** | Lag (1-7 periods) | Stock prices, fraud detection |
| **Trends** | Rolling mean/std | Forecasting, anomaly detection |
| **User lifecycle** | Days since event | Churn, upsell targeting |
| **Special events** | Binary flags | Holidays, promotions |
