# XGBoost: When Milliseconds and Accuracy Both Matter

**Uber's Surge Pricing Engine:**

Challenge: Predict demand in real-time across 10,000 geographic zones.

**Requirements:**
- Update every 30 seconds
- Process 1M features (weather, events, historical patterns, time)
- Accuracy critical (wrong price → lost drivers OR angry customers)
- Latency <100ms

**Why not Random Forest:**
- Too slow (500 trees × sequential evaluation)
- Lower accuracy on complex interactions

**XGBoost solution:**
```python
xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=6,
    tree_method='gpu_hist'  # GPU acceleration
)
```

**Results:**
- Inference: 23ms average (vs 180ms Random Forest)
- RMSE: 0.14 (vs 0.19 RF)
- Revenue optimization: $340M additional annual revenue
- Driver satisfaction: ↑18% (better supply/demand matching)

**Key:** Gradient boosting learns errors sequentially → more accurate. GPU parallelization → fast enough for real-time.

---

## Production Architecture

**Kaggle dominance (2015-2020):**
- 80% of winning solutions used XGBoost/LightGBM
- Outperforms Random Forest on tabular data consistently
- Faster than neural networks with similar accuracy

---

## 1. The Boosting Intuition (Functional Gradient Descent)

Unlike Random Forest (which averages independent trees), Boosting builds trees **sequentially**. Each tree aims to correct the mistakes of the previous one.

### The Algorithm
1.  **Initialize:** Predict the mean (for regression) or log-odds (for classification) for everyone.
2.  **Calculate Residuals:** $Error = Actual - Predicted$.
3.  **Train Tree 1:** Fit a tree *to the residuals* (not the target!).
    - Logic: "If I add this tree's prediction to the baseline, I get closer to the truth."
4.  **Update:** $Prediction_{new} = Prediction_{old} + (\eta \times Output_{Tree1})$
    - $\eta$ (Eta): Learning Rate. Shrinks the contribution of each tree to prevent overfitting.
5.  **Repeat:** Calculate new residuals, fit Tree 2, update.

> [!NOTE]
> We are performing **Gradient Descent**, but instead of tweaking weights (like in Neural Nets), we are tweaking functions (trees).

---

## 2. The Big Three: XGBoost vs LightGBM vs CatBoost

| Feature | XGBoost (2014) | LightGBM (2017) | CatBoost (2017) |
|---------|----------------|-----------------|-----------------|
| **Splitting** | Level-wise (balanced) | Leaf-wise (unbalanced, deep) | Symmetric (oblivious) |
| **Speed** | Fast | **Fastest** (Histogram-based) | Slower training, Fast inference |
| **Categoricals**| One-Hot (Manual) | Integer encoding (Native) | **Best** (Target Encoding) |
| **Accuracy** | Excellent | Excellent | Excellent |
| **Tuning** | Hard | Medium | Easy (robust defaults) |

---

## 3. XGBoost: The Classic Workhorse

**Key Feature:** Regularization (L1/L2) built into the objective function.

```python
import xgboost as xgb
from sklearn.metrics import accuracy_score

# Data matrices (DMatrix) are optimized for XGBoost speed
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Hyperparameters
params = {
    'max_depth': 6,             # Controls complexity
    'eta': 0.05,                # Learning rate (Critical!)
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',   # Monitoring metric
    'subsample': 0.8,           # Stochastic boosting (prevent overfit)
    'colsample_bytree': 0.8,    # Feature fraction
    'scale_pos_weight': 1       # Handle imbalance (sum(neg)/sum(pos))
}

# Training with Early Stopping
# Stop if validation score doesn't improve for 50 rounds
model_xgb = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,       # High Max
    evals=[(dtest, "Test")],
    early_stopping_rounds=50,
    verbose_eval=100
)

# Predict
y_pred_probs = model_xgb.predict(dtest)
y_pred = (y_pred_probs > 0.5).astype(int)
```

---

## 4. LightGBM: The Speed King

**Key Feature:** Leaf-wise growth. It expands the leaf with the max loss reduction, leading to deeper, asymmetrical trees. **Must limit `max_depth` to prevent overfit.**

```python
import lightgbm as lgb

# Dataset object preserves memory
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params_lgb = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,           # Primary control (instead of max_depth)
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

model_lgb = lgb.train(
    params_lgb,
    train_data,
    num_boost_round=1000,
    valid_sets=[test_data],
    early_stopping_rounds=50,
    verbose_eval=100
)
```

---

## 5. CatBoost: The Categorical Wizard

**Key Feature:** Handles categorical columns automatically using "Ordered Target Statistics" (averaging target value for the category, but careful to avoid leakage).

```python
from catboost import CatBoostClassifier

# Define which columns are categorical indices
cat_features = [0, 2, 5] 

model_cb = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='Logloss',
    eval_metric='Accuracy',
    verbose=100,
    random_seed=42
)

# No need for OHE! Just pass the raw features
model_cb.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_test, y_test),
    early_stopping_rounds=50
)
```

---

## 6. Comprehensive Tuning Guide (Cheat Sheet)

| Concept | Parameter (XGB) | Parameter (LGBM) | Parameter (Cat) | Action to Reduce Overfitting |
|---------|-----------------|------------------|-----------------|------------------------------|
| **Number of Trees** | `num_boost_round` | `num_boost_round` | `iterations` | Rely on Early Stopping. |
| **Step Size** | `eta` (0.01-0.3) | `learning_rate` | `learning_rate` | **Decrease** (and increase rounds). |
| **Tree Depth** | `max_depth` | `num_leaves` | `depth` | **Decrease** (Common: 3-8). |
| **Row Sampling** | `subsample` | `bagging_fraction` | `subsample` | Decrease to 0.7-0.9. |
| **Col Sampling** | `colsample_bytree` | `feature_fraction`| `rsm` | Decrease to 0.7-0.9. |
| **Min Split Loss** | `gamma` | `min_split_gain` | `l2_leaf_reg` | **Increase** (Conservative). |

> [!TIP]
> **Tuning Strategy:**
> 1. Set `eta` to 0.1.
> 2. Tune `max_depth` (or `num_leaves`) and structural params.
> 3. Lower `eta` to 0.01 and increase `num_boost_round`.
