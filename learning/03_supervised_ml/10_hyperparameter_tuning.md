# Hyperparameter Tuning: Bayesian Optimization & The Art of AutoML

Stop guessing numbers.
"Should `learning_rate` be 0.1 or 0.05?"
"Should `max_depth` be 5 or 8?"

In robust Machine Learning, **Automated Hyperparameter Optimization (HPO)** is not a luxury; it is a mathematical necessity. A 3% gain in AUC from proper tuning can determine which team wins a Kaggle competition or which trading algorithm is profitable.

---

## 1. The Theory: Why "Grid Search" is Obsolete

### The Methods
1.  **Grid Search:** Try ALL combinations. $O(N^k)$. Scales exponentially. Impossible for >4 parameters.
2.  **Random Search:** Try Random combinations.
    - *Math Fact:* In high dimensions, random search finds a good solution faster than grid search because hyperparameters often have low "effective dimensionality" (meaning only 1 or 2 actually matter).

### The King: Bayesian Optimization
Grid and Random search are **uninformed**. They don't learn from the past.
("I just tried `lr=0.5` and it was terrible. Why am I now trying `lr=0.49`? I should try `0.01`!")

**Bayesian Optimization** builds a probabilistic model (surrogate) of the objective function.
$$P(Score | Hyperparameters)$$

#### The TPE Algorithm (Tree-structured Parzen Estimator)
Used by **Optuna** and **Hyperopt**. Instead of modeling $y = f(x)$, it models two densities:
1.  $l(x)$: The distribution of params $x$ that yielded **Good** scores (Top 20%).
2.  $g(x)$: The distribution of params $x$ that yielded **Bad** scores.

The algorithm chooses the next $x$ to maximize the ratio $l(x) / g(x)$.
*   "Show me params likely to be in the Good group and unlikely to be in the Bad group."

---

## 2. Advanced Technique: Pruning (Hyperband)

We don't just want to pick parameters; we want to stop bad training runs EARLY.

**Successive Halving / Hyperband:**
1.  Start 100 trials. Train for only 10 epochs.
2.  Kill the bottom 50%.
3.  Train the survivors for 20 epochs.
4.  Kill the bottom 50%.
5.  Repeat until one winner remains.

This allocates resources exponentially to the most promising candidates.

---

## 3. Implementation: Optuna with Pruning

```python
import optuna
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load Data
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)

def objective(trial):
    # 1. Suggest Parameters (The "Prior")
    # Log uniform helps explore 0.001 and 0.1 with equal weight
    param = {
        'objective': 'binary:logistic',
        'metric': 'logloss',
        'verbosity': 0,
        'booster': 'gbtree',
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.2, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0),
        'max_depth': trial.suggest_int('max_depth', 1, 9),
        'eta': trial.suggest_float('eta', 1e-8, 1.0, log=True),
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
    }

    # 2. Pruning Integration (Manually stepping through boosting rounds)
    # We break training into chunks to allow Optuna to inspect progress
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-logloss")
    
    model = xgb.train(
        param, 
        dtrain, 
        num_boost_round=1000,
        evals=[(dtest, "validation")],
        callbacks=[pruning_callback], # THE MAGIC LINE
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    # 3. Return Final Metric
    preds = model.predict(dtest)
    pred_labels = [round(value) for value in preds]
    return accuracy_score(y_test, pred_labels)

# 4. Create Study with specific Pruner
# MedianPruner: Prune if trial is worse than the median of previous trials at this step
study = optuna.create_study(
    direction='maximize', 
    sampler=optuna.samplers.TPESampler(),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
)

study.optimize(objective, n_trials=100)

print(f"Best Accuracy: {study.best_value}")
print(f"Best Params: {study.best_params}")
```

---

## 4. Multi-Objective Optimization (Pareto Fronts)

Sometimes you have two conflicting goals:
1.  Maximize **Accuracy**.
2.  Minimize **Inference Latency** (for mobile deployment).

Optuna can optimize both simultaneously to find the **Pareto Front**—the set of optimal trade-offs.

```python
def multi_objective(trial):
    params = ... # suggest params
    model.fit(X_train, y_train)
    
    acc = model.score(X_test, y_test)
    
    # Calculate Latency
    import time
    start = time.time()
    model.predict(X_test[:1000]) # Predict on batch
    latency = time.time() - start
    
    return acc, latency

study = optuna.create_study(directions=['maximize', 'minimize'])
study.optimize(multi_objective, n_trials=100)

# Visualize the Trade-off
optuna.visualization.plot_pareto_front(study, target_names=["Accuracy", "Latency"])
```

---

## 5. Storage (Don't lose your work)

You run a tuning job for 3 days. Your computer crashes. It's all gone.
**Always use a database backend.**

```python
# Creates a local SQLite file. You can stop/resume anytime.
study = optuna.create_study(
    study_name="my_xgboost_experiment", 
    storage="sqlite:///db.sqlite3", 
    load_if_exists=True
)
```
