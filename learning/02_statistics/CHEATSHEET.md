# Statistics Cheatsheet for Data Science

## Quick Reference Guide

---

## 1. Probability Distributions

### Key Concepts

| Term | Definition | When to Use |
|------|------------|-------------|
| **Normal Distribution** | Bell-shaped, symmetric distribution | Natural phenomena, continuous data, CLT applies |
| **Binomial Distribution** | Count of successes in n trials | Success/failure scenarios, fixed trials |
| **Poisson Distribution** | Count of events in fixed interval | Rare events, arrivals, defects |
| **Exponential Distribution** | Time between events | Wait times, time to failure |
| **Central Limit Theorem** | Sample means → Normal (large n) | Justifies normal-based tests even with non-normal data |

### Key Formulas

**Normal Distribution:**
- PDF: $f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}$
- Parameters: μ (mean), σ (std dev)
- **68-95-99.7 Rule:** 68% within 1σ, 95% within 2σ, 99.7% within 3σ

**Binomial Distribution:**
- PMF: $P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$
- Mean: μ = n × p
- Variance: σ² = n × p × (1-p)

**Poisson Distribution:**
- PMF: $P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$
- Mean = Variance = λ

**Central Limit Theorem:**
- $\bar{X} \sim N(\mu, \frac{\sigma^2}{n})$
- Standard Error: SE = σ/√n

### Python Syntax

```python
from scipy import stats
import numpy as np

# Normal
stats.norm.pdf(x, loc=μ, scale=σ)          # Probability density
stats.norm.cdf(x, loc=μ, scale=σ)          # Cumulative probability
stats.norm.ppf(p, loc=μ, scale=σ)          # Inverse CDF (quantile)

# Binomial
stats.binom.pmf(k, n, p)                   # P(X = k)
stats.binom.cdf(k, n, p)                   # P(X ≤ k)

# Poisson
stats.poisson.pmf(k, mu=λ)                 # P(X = k)
stats.poisson.cdf(k, mu=λ)                 # P(X ≤ k)

# Check normality
stats.shapiro(data)                        # Shapiro-Wilk test
stats.probplot(data, dist="norm", plot=plt)  # Q-Q plot
```

---

## 2. Hypothesis Testing

### Key Concepts

| Term | Definition | Formula/Value |
|------|------------|---------------|
| **Null Hypothesis (H₀)** | Status quo, no effect | What we test against |
| **Alternative Hypothesis (H₁)** | What we want to prove | Research hypothesis |
| **p-value** | P(data \| H₀ is true) | **p < α → Reject H₀** |
| **Significance Level (α)** | Type I error rate | Usually 0.05 (5%) |
| **Type I Error** | False Positive | Reject true H₀ (α) |
| **Type II Error** | False Negative | Fail to reject false H₀ (β) |
| **Power** | 1 - β | P(Reject H₀ \| H₀ false) |

### Decision Rule

```
If p-value < α  →  REJECT H₀ (Significant)
If p-value ≥ α  →  FAIL TO REJECT H₀ (Not significant)
```

### T-Tests

| Test | Use Case | Formula |
|------|----------|---------|
| **One-Sample** | Compare sample mean to known value | $t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}$ |
| **Two-Sample** | Compare two independent groups | $t = \frac{\bar{x}_1 - \bar{x}_2}{SE_{pooled}}$ |
| **Paired** | Same group, different times | $t = \frac{\bar{d}}{s_d / \sqrt{n}}$ |

**Degrees of Freedom:**
- One-sample: df = n - 1
- Two-sample: df = n₁ + n₂ - 2
- Paired: df = n - 1

### Python Syntax

```python
from scipy import stats

# One-sample t-test
t_stat, p_value = stats.ttest_1samp(data, popmean=μ₀)

# Two-sample t-test
t_stat, p_value = stats.ttest_ind(group1, group2)

# Paired t-test
t_stat, p_value = stats.ttest_rel(before, after)

# Effect size (Cohen's d)
def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std
```

---

## 3. Confidence Intervals

### Key Concepts

| Term | Meaning |
|------|---------|
| **95% CI** | If we repeated study 100x, ~95 CIs would contain true parameter |
| **Margin of Error** | ± range around point estimate |
| **Width ∝ 1/√n** | Quadruple sample size → halve CI width |

### Formulas

**Mean (known σ or large n):**
$$CI = \bar{x} \pm z_{\alpha/2} \times \frac{\sigma}{\sqrt{n}}$$

**Mean (unknown σ, small n):**
$$CI = \bar{x} \pm t_{\alpha/2, df} \times \frac{s}{\sqrt{n}}$$

**Proportion:**
$$CI = \hat{p} \pm z_{\alpha/2} \times \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$$

**Common z-values:**
- 90% CI: z = 1.645
- 95% CI: z = 1.96
- 99% CI: z = 2.576

### Python Syntax

```python
from scipy import stats

# Mean (t-distribution)
mean = data.mean()
std = data.std(ddof=1)
n = len(data)
confidence = 0.95

t_critical = stats.t.ppf((1 + confidence) / 2, df=n-1)
margin_error = t_critical * (std / np.sqrt(n))
ci = (mean - margin_error, mean + margin_error)

# Proportion (Wilson Score - more accurate)
from statsmodels.stats.proportion import proportion_confint
ci_lower, ci_upper = proportion_confint(count, nobs, alpha=0.05, method='wilson')

# Bootstrap CI
from sklearn.utils import resample
bootstrap_means = [np.mean(resample(data)) for _ in range(10000)]
ci = np.percentile(bootstrap_means, [2.5, 97.5])
```

---

## 4. ANOVA (Analysis of Variance)

### Key Concepts

| Term | Definition | Formula |
|------|------------|---------|
| **F-statistic** | Between-group var / Within-group var | $F = \frac{MS_{between}}{MS_{within}}$ |
| **Eta-squared (η²)** | Proportion of variance explained | $\eta^2 = \frac{SS_{between}}{SS_{total}}$ |
| **Post-hoc tests** | Which groups differ? | Tukey HSD, Bonferroni |

**Effect Size (η²):**
- < 0.01: Small
- 0.01-0.06: Medium
- ≥ 0.06: Large

### When to Use

- **One-Way ANOVA:** 3+ groups, 1 factor
- **Two-Way ANOVA:** Multiple factors, test interactions
- **Kruskal-Wallis:** Non-parametric alternative (non-normal data)

### Python Syntax

```python
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# One-way ANOVA
f_stat, p_value = stats.f_oneway(group1, group2, group3, group4)

# Post-hoc (Tukey HSD)
all_data = np.concatenate([group1, group2, group3])
all_groups = ['A']*len(group1) + ['B']*len(group2) + ['C']*len(group3)
tukey = pairwise_tukeyhsd(all_data, all_groups, alpha=0.05)
print(tukey)

# Kruskal-Wallis (non-parametric)
h_stat, p_value = stats.kruskal(group1, group2, group3)

# Check assumptions
stats.levene(group1, group2, group3)  # Homogeneity of variances
```

---

## 5. Chi-Square Tests

### Key Concepts

| Test | Use Case | Formula |
|------|----------|---------|
| **Test of Independence** | Are 2 categorical variables related? | $\chi^2 = \sum \frac{(O - E)^2}{E}$ |
| **Goodness of Fit** | Does data match expected distribution? | Same formula |
| **Cramér's V** | Effect size for association | $V = \sqrt{\frac{\chi^2}{n(k-1)}}$ |

**Expected Frequency:**
$$E_{ij} = \frac{(\text{Row Total}_i) \times (\text{Column Total}_j)}{\text{Grand Total}}$$

**Cramér's V Interpretation:**
- < 0.1: Negligible
- 0.1-0.3: Weak
- 0.3-0.5: Moderate
- ≥ 0.5: Strong

### Assumptions

- **All expected frequencies ≥ 5**
- If violated: Use Fisher's Exact Test (2×2 tables)

### Python Syntax

```python
from scipy.stats import chi2_contingency, fisher_exact
import pandas as pd

# Chi-square test of independence
contingency_table = pd.crosstab(df['var1'], df['var2'])
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

# Cramér's V
n = contingency_table.sum().sum()
min_dim = min(contingency_table.shape) - 1
cramers_v = np.sqrt(chi2 / (n * min_dim))

# Fisher's Exact Test (2x2 only)
odds_ratio, p_value = fisher_exact([[a, b], [c, d]])

# Goodness of fit
observed = [45, 52, 48, 50, 47, 58]
expected = [50, 50, 50, 50, 50, 50]
chi2, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
```

---

## 6. Correlation Analysis

### Key Concepts

| Correlation | Use Case | Range | Measures |
|-------------|----------|-------|----------|
| **Pearson (r)** | Linear relationship, continuous data | -1 to +1 | Linear association |
| **Spearman (ρ)** | Monotonic, ordinal, outliers | -1 to +1 | Rank correlation |
| **Kendall's τ** | Ordinal, small samples | -1 to +1 | Rank concordance |

**Interpretation:**
- |r| < 0.2: Very weak
- 0.2 ≤ |r| < 0.4: Weak
- 0.4 ≤ |r| < 0.7: Moderate
- |r| ≥ 0.7: Strong

### Important

**r² = Proportion of variance explained**
- r = 0.8 → r² = 0.64 (64% of variance explained)

**Correlation ≠ Causation**
- Always check for confounding variables
- Use partial correlation to control confounders

### Multicollinearity (VIF)

**Variance Inflation Factor:**
- VIF < 5: No problem
- 5 ≤ VIF < 10: Moderate
- VIF ≥ 10: Severe (drop variable)

### Python Syntax

```python
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Pearson correlation
r, p_value = stats.pearsonr(x, y)

# Spearman (rank-based)
rho, p_value = stats.spearmanr(x, y)

# Correlation matrix
corr_matrix = df.corr()

# Multicollinearity (VIF)
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Partial correlation (control for Z)
from pingouin import partial_corr
partial_corr(data=df, x='X', y='Y', covar='Z')
```

---

## 7. Power Analysis & Sample Size

### Key Concepts

**Four Interconnected Parameters:**
1. **Sample Size (n)** - What we calculate
2. **Effect Size (d)** - How big is the difference?
3. **Significance (α)** - Usually 0.05
4. **Power (1-β)** - Usually 0.80 (80%)

**Know any 3 → Calculate the 4th**

### Effect Sizes

**Cohen's d (t-tests):**
- 0.2: Small
- 0.5: Medium
- 0.8: Large

**Eta-squared (ANOVA):**
- 0.01: Small
- 0.06: Medium
- 0.14: Large

**Cramér's V (Chi-square):**
- 0.1: Small
- 0.3: Medium
- 0.5: Large

### Python Syntax

```python
from statsmodels.stats.power import TTestIndPower, zt_ind_solve_power

# T-test power analysis
power_analysis = TTestIndPower()

# Calculate required sample size
n = power_analysis.solve_power(effect_size=0.5, power=0.80, alpha=0.05)

# Calculate achieved power
power = power_analysis.solve_power(effect_size=0.5, nobs1=100, alpha=0.05)

# Calculate minimum detectable effect
mde = power_analysis.solve_power(nobs1=100, power=0.80, alpha=0.05)

# For proportions
n = zt_ind_solve_power(effect_size=effect_size, alpha=0.05, power=0.80)
```

### Multiple Testing Corrections

| Method | Use Case | Formula |
|--------|----------|---------|
| **Bonferroni** | Conservative, guarantees FWER | α_adj = α / m |
| **FDR (Benjamini-Hochberg)** | Less conservative, more power | Controls false discovery rate |

```python
from statsmodels.stats.multitest import multipletests

reject, pvals_adj, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')
reject, pvals_adj, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
```

---

## 8. Test Selection Guide

### Flow Chart

```
┌─────────────────────────────────────┐
│ What type of data?                  │
└─────────────────────────────────────┘
         │
         ├─── Numerical ────────────────┐
         │                              │
         │    How many groups?          │
         │    ├─ 1 group                │
         │    │  → One-sample t-test    │
         │    ├─ 2 groups               │
         │    │  ├─ Independent         │
         │    │  │  → Two-sample t-test │
         │    │  └─ Paired              │
         │    │     → Paired t-test     │
         │    └─ 3+ groups              │
         │       → ANOVA                │
         │                              │
         └─── Categorical ──────────────┐
                                        │
              Are variables related?    │
              → Chi-square test         │
```

### Quick Reference Table

| Question | Test | Python |
|----------|------|--------|
| Is mean = 10? | One-sample t-test | `stats.ttest_1samp(data, 10)` |
| Group A vs B? | Two-sample t-test | `stats.ttest_ind(A, B)` |
| Before vs After? | Paired t-test | `stats.ttest_rel(before, after)` |
| Compare 4 groups? | ANOVA | `stats.f_oneway(g1, g2, g3, g4)` |
| Gender vs Product? | Chi-square | `chi2_contingency(crosstab)` |
| Age vs Income? | Pearson correlation | `stats.pearsonr(age, income)` |
| How many samples? | Power analysis | `power.solve_power(...)` |

---

## 9. Common Mistakes to Avoid

| Mistake | Why It's Wrong | Correct Approach |
|---------|----------------|------------------|
| **"Accept H₀"** | Can't prove null | Say "Fail to reject H₀" |
| **p-hacking** | Invalidates test | Pre-register hypothesis |
| **Ignoring effect size** | Significant ≠ important | Always report effect size + CI |
| **Multiple tests without correction** | Inflated Type I error | Use Bonferroni or FDR |
| **Assuming causation from correlation** | Confounding variables | Use experiments or causal methods |
| **Violating assumptions** | Unreliable results | Check assumptions, use alternatives |

---

## 10. Common Imports

```python
# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical tests
from scipy import stats
from scipy.stats import (
    norm, binom, poisson, expon,          # Distributions
    ttest_1samp, ttest_ind, ttest_rel,    # T-tests
    f_oneway, kruskal,                     # ANOVA
    chi2_contingency, fisher_exact,        # Chi-square
    pearsonr, spearmanr,                   # Correlation
    shapiro, levene                        # Assumption tests
)

# Advanced analysis
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multitest import multipletests

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
```

---

## 11. Interpretation Guidelines

### P-values

| P-value | Interpretation |
|---------|----------------|
| **p < 0.001** | Very strong evidence against H₀ |
| **0.001 ≤ p < 0.01** | Strong evidence |
| **0.01 ≤ p < 0.05** | Moderate evidence (conventionally "significant") |
| **0.05 ≤ p < 0.10** | Weak evidence (sometimes reported as "marginally significant") |
| **p ≥ 0.10** | Little to no evidence against H₀ |

### Reporting Results

**Template:**
```
"[Test name] revealed that [what you found] ([test statistic] = X.XX, 
p = 0.XXX, [effect size measure] = X.XX). This indicates [interpretation]."
```

**Example:**
```
"An independent samples t-test revealed that the treatment group (M = 15.3, 
SD = 2.1) had significantly higher scores than the control group (M = 12.1, 
SD = 1.9), t(148) = 8.42, p < .001, Cohen's d = 1.62. This indicates a 
large treatment effect."
```

---

## Quick Decision Tree

```
START
  │
  ├─ Comparing means? ─→ YES ─→ Numerical data ─→ t-test or ANOVA
  │                                                  
  ├─ Testing relationship? ─→ YES ─→ Both categorical? ─→ Chi-square
  │                                 Both numerical? ─→ Correlation
  │                                  
  ├─ Need sample size? ─→ YES ─→ Power analysis
  │
  └─ Quantifying uncertainty? ─→ YES ─→ Confidence interval
```

---

**Remember:** 
- Always check assumptions
- Report effect sizes, not just p-values
- Use visualizations
- Think about practical significance
- Document your decisions
