# Hypothesis Testing

## The $1 Billion Question: Can You Prove It?

**Picture this:** You're in a boardroom at Amazon. A product manager says, "This new checkout design will increase conversions by 15%—I can feel it." Your CEO turns to you: "Is that true, or are we about to waste $50 million rolling out something that doesn't work?"

You run an A/B test. 10,000 users see the old design, 10,000 see the new one. New design: 12.3% conversion. Old design: 11.8%. The PM celebrates. But you calculate the p-value: 0.14. **"Stop,"** you say. **"This could just be random noise. We can't be confident it's real."**

You just saved $50 million and prevented a company-wide disaster. That's the power of hypothesis testing.

**Why hypothesis testing is your superpower:**
- **You make million-dollar decisions with confidence** — No more gut feelings. You quantify certainty.
- **You protect against costly mistakes** — Launching bad products, approving ineffective drugs, making wrong investments.
- **You speak the language of science and business** — p-values, confidence, significance—these are how the world makes decisions.
- **You turn data into defensible truth** — "I believe" becomes "I can prove with 95% confidence."

**Real scenarios where hypothesis testing saves the day:**
- **Pharmaceuticals:** Does this drug work better than placebo? (Billions in R&D, lives at stake)
- **Tech:** Should we change our recommendation algorithm? (User engagement, revenue impact)
- **Finance:** Is this trading strategy profitable or lucky? (Investment decisions, risk management)
- **Manufacturing:** Has quality decreased? (Recalls, reputation, lawsuits)

**The brutal reality:** Without hypothesis testing, you're guessing. With it, you're making evidence-based decisions that stand up to scrutiny from CEOs, regulators, and the scientific community.

This isn't just math. **This is your credibility. Your career. Your impact.**

Let's master it.

---

### Quick Decision-Making Example: The $50M A/B Test

**Situation:** New checkout design shows 12.3% conversion vs 11.8% control

**The calculation:**
```python
# Two-proportion z-test
control_conversions = 1180  # out of 10,000
treatment_conversions = 1230  # out of 10,000

# Calculate p-value
from statsmodels.stats.proportion import proportions_ztest
count = [1230, 1180]
nobs = [10000, 10000]

z_stat, p_value = proportions_ztest(count, nobs)
# Result: p = 0.14
```

**The decision tree:**
- **If p < 0.05:** "95% confident improvement is real" → **Deploy** ($50M investment)
- **If p ≥ 0.05:** "Could be random noise" → **Don't deploy** (avoid risk)

**Actual result:** p = 0.14 → **STOP. Don't deploy.**

**Why this matters:**
- Deploying would cost $50M
- 14% chance of seeing this difference by pure luck
- Not confident enough to bet $50M on it

**What saved the company:** One p-value calculation. 30 seconds of math. $50M saved.

---

### Second Example: Drug Effectiveness Clinical Trial

**Scenario:** Pharmaceutical company testing new diabetes medication

**Results:**
- **Placebo group (n=150):** Average blood sugar reduction = 8 mg/dL (std = 12 mg/dL)
- **Drug group (n=150):** Average blood sugar reduction = 15 mg/dL (std = 14 mg/dL)

**Question:** Is the drug actually better, or could this be randomchance?

**The Detailed Math:**

```python
import numpy as np
from scipy import stats

# Step 1: Define the data
placebo_mean = 8.0     # Average reduction in placebo group (mg/dL)
placebo_std = 12.0     # Standard deviation in placebo group
placebo_n = 150        # Number of patients in placebo group

drug_mean = 15.0       # Average reduction in drug group (mg/dL)
drug_std = 14.0        # Standard deviation in drug group
drug_n = 150           # Number of patients in drug group

print("===== CLINICAL TRIAL DATA =====")
print(f"Placebo: {placebo_mean} mg/dL reduction (n={placebo_n})")
print(f"Drug:    {drug_mean} mg/dL reduction (n={drug_n})")
print(f"Observed difference: {drug_mean - placebo_mean} mg/dL")

# Step 2: Set up hypotheses
# H₀ (Null): Drug has NO effect (μ_drug = μ_placebo)
# H₁ (Alternative): Drug DOES have effect (μ_drug ≠ μ_placebo)
alpha = 0.05  # Significance level (5% Type I error rate)

print(f"\n===== HYPOTHESIS TESTING =====")
print(f"H₀: μ_drug = μ_placebo (no difference)")
print(f"H₁: μ_drug ≠ μ_placebo (drug works)")
print(f"Significance level (α): {alpha}")

# Step 3: Calculate pooled standard error
# Formula: SE = √(s₁²/n₁ + s₂²/n₂)
se_placebo = (placebo_std ** 2) / placebo_n
#            ^^^^^^^^^^^^^^^^^^    ^^^^^^^^^^^
#            |                     |
#            |                     +-- Divide by sample size
#            +-- Square the standard deviation (variance)
#            Substitution: 12² / 150 = 144 / 150 = 0.96

se_drug = (drug_std ** 2) / drug_n
#         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#         Substitution: 14² / 150 = 196 / 150 = 1.307

pooled_se = np.sqrt(se_placebo + se_drug)
#           ^^^^^^^^^^^^^^^^^^^^^^^^^
#           Formula: √(SE₁ + SE₂)
#           Substitution: √(0.96 + 1.307) = √2.267 = 1.506

print(f"\n===== STANDARD ERROR CALCULATION =====")
print(f"SE_placebo = s²/n = {placebo_std}² / {placebo_n} = {se_placebo:.3f}")
print(f"SE_drug = s²/n = {drug_std}² / {drug_n} = {se_drug:.3f}")
print(f"Pooled SE = √(SE₁ + SE₂) = √({se_placebo:.3f} + {se_drug:.3f}) = {pooled_se:.3f}")

# Step 4: Calculate t-statistic
# Formula: t = (x̄₁ - x̄₂) / SE_pooled
t_statistic = (drug_mean - placebo_mean) / pooled_se
#              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#              (mean difference) / (standard error)
#              Substitution: (15 - 8) / 1.506 = 7 / 1.506 = 4.648

print(f"\n===== T-STATISTIC CALCULATION =====")
print(f"t = (x̄_drug - x̄_placebo) / SE_pooled")
print(f"t = ({drug_mean} - {placebo_mean}) / {pooled_se:.3f}")
print(f"t = {drug_mean - placebo_mean} / {pooled_se:.3f}")
print(f"t = {t_statistic:.3f}")
print(f"\nInterpretation: The observed difference is {t_statistic:.2f} standard errors away from zero")

# Step 5: Calculate degrees of freedom
# For two-sample t-test: df = n₁ + n₂ - 2
df = placebo_n + drug_n - 2
#    ^^^^^^^^^^^^^^^^^^^^^^^^
#    Substitution: 150 + 150 - 2 = 298

print(f"\nDegrees of freedom: {placebo_n} + {drug_n} - 2 = {df}")

# Step 6: Calculate p-value
# Two-tailed test: probability of seeing |t| this extreme or more
p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df))
#         ^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#         |   |
#         |   +-- P(T > |t|) for one tail
#         +-- Multiply by 2 for two-tailed test
#
# stats.t.cdf gives cumulative probability P(T ≤ t)
# 1 - cdf gives survival probability P(T > t)
# We use abs(t) to handle negative t-values

print(f"\n===== P-VALUE CALCULATION =====")
print(f"p-value = 2 × P(T > |{t_statistic:.3f}|)")
print(f"p-value = {p_value:.6f}")
print(f"As percentage: {p_value * 100:.4f}%")

# Step 7: Calculate critical t-value for comparison
t_critical = stats.t.ppf(1 - alpha/2, df)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#            For α=0.05, two-tailed: need t where P(T > t) = 0.025
#            ppf(0.975, 298) gives the critical value

print(f"\nCritical t-value (α={alpha}, two-tailed): ±{t_critical:.3f}")
print(f"Our t-statistic: {t_statistic:.3f}")

# Step 8: Make decision
print(f"\n===== DECISION =====")
if p_value < alpha:
    print(f"✓ REJECT H₀ (p = {p_value:.6f} < {alpha})")
    print(f"  The drug IS significantly better than placebo")
    result_text = "SIGNIFICANT"
else:
    print(f"✗ FAIL TO REJECT H₀ (p = {p_value:.6f} ≥ {alpha})")
    print(f"  Insufficient evidence that drug works")
    result_text = "NOT SIGNIFICANT"

if abs(t_statistic) > t_critical:
    print(f"  |t| = {abs(t_statistic):.3f} > {t_critical:.3f} (critical value)")
else:
    print(f"  |t| = {abs(t_statistic):.3f} ≤ {t_critical:.3f} (critical value)")
```

**Output:**
```
===== CLINICAL TRIAL DATA =====
Placebo: 8.0 mg/dL reduction (n=150)
Drug:    15.0 mg/dL reduction (n=150)
Observed difference: 7.0 mg/dL

===== HYPOTHESIS TESTING =====
H₀: μ_drug =  μ_placebo (no difference)
H₁: μ_drug ≠ μ_placebo (drug works)
Significance level (α): 0.05

===== STANDARD ERROR CALCULATION =====
SE_placebo = s²/n = 12.0² / 150 = 0.960
SE_drug = s²/n = 14.0² / 150 = 1.307
Pooled SE = √(SE₁ + SE₂) = √(0.960 + 1.307) = 1.506

===== T-STATISTIC CALCULATION =====
t = (x̄_drug - x̄_placebo) / SE_pooled
t = (15.0 - 8.0) / 1.506
t = 7.0 / 1.506
t = 4.648

Interpretation: The observed difference is 4.65 standard errors away from zero

Degrees of freedom: 150 + 150 - 2 = 298

===== P-VALUE CALCULATION =====
p-value = 2 × P(T > |4.648|)
p-value = 0.000005
As percentage: 0.0005%

Critical t-value (α=0.05, two-tailed): ±1.968
Our t-statistic: 4.648

===== DECISION =====
✓ REJECT H₀ (p = 0.000005 < 0.05)
  The drug IS significantly better than placebo
  |t| = 4.648 > 1.968 (critical value)
```

**The Decision Analysis:**

**What the numbers tell us:**

1. **T-statistic = 4.648**
   - The 7 mg/dL difference is 4.65 standard errors away from zero
   - Extremely unlikely to see this by chance

2. **P-value = 0.000005 (0.0005%)**
   - Only 0.0005% chance of seeing this difference if drug didn't work
   - 99.9995% confident the drug works

3. **Critical value = 1.968**
   - Our |t| = 4.648 is WAY beyond the critical threshold
   - Clear rejection of null hypothesis

**Regulatory Decision (FDA Approval):**

| Criterion | Threshold | Our Result | Status |
|-----------|-----------|------------|--------|
| **P-value** | < 0.05 | 0.000005 | ✓ PASS |
| **Clinical Significance** | ≥ 5 mg/dL reduction | 7 mg/dL | ✓ PASS |
| **Safety Profile** | Acceptable | (assume yes) | ✓ PASS |

**FDA Decision: APPROVE DRUG**

**Business Impact:**

**Scenario A: Without proper hypothesis testing** (relying on gut feeling)
- "7 mg/dL seems good, but is it real?"
- Hesitate to seek FDA approval
- Competitor launches first
- **Lost market:** $500M annually

**Scenario B: With hypothesis testing** ✓
- p < 0.000005 = Overwhelming evidence
- Confident FDA submission
- Approval granted
- **Market capture:** $500M annually
- Patient lives improved: Tens of thousands

**The Math Changed:**
- Uncertainty → Certainty
- Hesitation → Action
- Lost opportunity → $500M business

**Key Formulas Used:**

1. **Pooled Standard Error:**
   $$SE = \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}$$

2. **T-Statistic:**
   $$t = \frac{\bar{x}_1 - \bar{x}_2}{SE}$$

3. **Degrees of Freedom:**
   $$df = n_1 + n_ 2 - 2$$

4. **P-value (two-tailed):**
   $$p = 2 \times P(T > |t_{obs}|)$$

**One hypothesis test. Millions of dollars. Thousands of lives.**

---

## Why This Matters

Hypothesis testing is the cornerstone of data-driven decision making. It provides a structured framework to:
- **Validate business decisions** (Is the new feature better?)
- **Test scientific claims** (Does this drug work?)
- **Make decisions with quantified risk** (95% confidence)
- **Avoid jumping to conclusions** from random noise

---

## The Hypothesis Testing Framework

### Step 1: State Hypotheses

**Null Hypothesis (H₀):** The status quo, "no effect"
- Example: "The new design has NO effect on conversion rate"

**Alternative Hypothesis (H₁ or Hₐ):** What we want to test
- Example: "The new design DOES affect conversion rate"

###Types of Alternative Hypotheses:
- **Two-tailed:** H₁: μ ≠ μ₀ (different, but don't know direction)
- **Right-tailed:** H₁: μ > μ₀ (specifically greater)
- **Left-tailed:** H₁: μ < μ₀ (specifically less)

### Step 2: Choose Significance Level (α)

**α = Probability of Type I Error** (False Positive)
- Standard: **α = 0.05** (5% chance of rejecting true H₀)
- Conservative: **α = 0.01** (stricter)
- Exploratory: **α = 0.10** (more lenient)

### Step 3: Calculate Test Statistic

Transform your data into a standardized value that follows a known distribution.

### Step 4: Compute p-value

**p-value:** Probability of observing your data (or more extreme) if H₀ is true

### Step 5: Make Decision

- **p < α:** Reject H₀ (result is statistically significant)
- **p ≥ α:** Fail to reject H₀ (insufficient evidence)

---

## Error Types

| Reality Truth ↓ <br> Decision → | **Reject H₀** | **Fail to Reject H₀** |
|---|---|---|
| **H₀ is True** | **Type I Error** <br> (False Positive) <br> α | Correct <br> (1-α) |
| **H₀ is False** | Correct <br> (Power = 1-β) | **Type II Error** <br> (False Negative) <br> β |

**Type I Error (α):** Rejecting a true null hypothesis
- Example: Approving ineffective drug

**Type II Error (β):** Failing to reject a false null hypothesis
- Example: Rejecting effective drug

**Power (1-β):** Probability of correctly rejecting false H₀
- Higher power = Lower chance of Type II error

---

## T-Tests

### When to Use

Comparing means when:
- Sample size is small to moderate (n < 30 typically)
- Population standard deviation is unknown
- Data is approximately normal (or n ≥ 30 due to CLT)

---

### 1. One-Sample T-Test

**Purpose:** Test if sample mean differs from a known population value

**Hypotheses:**
- H₀: μ = μ₀
- H₁: μ ≠ μ₀

**Test Statistic:**

$$t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}$$

**Degrees of Freedom:** df = n - 1

**Python Example:**

```python
from scipy import stats
import numpy as np

# Example: Is average height different from 170 cm?
heights = np.array([165, 172, 168, 175, 170, 169, 171, 174, 168, 172,
                   166, 173, 170, 171, 169, 172, 168, 174, 171, 170])

population_mean = 170

# Perform one-sample t-test
t_stat, p_value = stats.ttest_1samp(heights, population_mean)

print(f"Sample Mean: {heights.mean():.2f} cm")
print(f"Sample Std: {heights.std(ddof=1):.2f} cm")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Degrees of Freedom: {len(heights)-1}")

alpha = 0.05
if p_value < alpha:
    print(f"\n✓ REJECT H₀: Mean is significantly different from {population_mean} cm")
else:
    print(f"\n✗ FAIL TO REJECT H₀: No significant difference from {population_mean} cm")
```

---

### 2. Independent Two-Sample T-Test

**Purpose:** Compare means of two independent groups

**Hypotheses:**
- H₀: μ₁ = μ₂
- H₁: μ₁ ≠ μ₂

**Assumptions:**
1. Independence of observations
2. Normality (or n ≥ 30 per group)
3. **Equal variances** (homoscedasticity)
   - If violated, use Welch's t-test (default in scipy)

**Python Example:**

```python
# Example: Do men and women have different salaries?
men_salaries = np.array([55000, 60000, 58000, 62000, 57000, 61000, 59000, 63000])
women_salaries = np.array([52000, 54000, 53000, 56000, 51000, 55000, 54000, 57000])

# Test for equal variances first (Levene's test)
stat, p_levene = stats.levene(men_salaries, women_salaries)
print(f"Levene's Test p-value: {p_levene:.4f}")
if p_levene > 0.05:
    print("  → Variances are equal, use standard t-test")
    equal_var = True
else:
    print("  → Variances are NOT equal, use Welch's t-test")
    equal_var = False

# Perform two-sample t-test
t_stat, p_value = stats.ttest_ind(men_salaries, women_salaries, equal_var=equal_var)

print(f"\nMen Mean: ${men_salaries.mean():,.0f}")
print(f"Women Mean: ${women_salaries.mean():,.0f}")
print(f"Difference: ${men_salaries.mean() - women_salaries.mean():,.0f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("\n✓ REJECT H₀: Significant salary difference exists")
else:
    print("\n✗ FAIL TO REJECT H₀: No significant salary difference")

# Calculate effect size (Cohen's d)
pooled_std = np.sqrt(((len(men_salaries)-1)*men_salaries.std()**2 + 
                      (len(women_salaries)-1)*women_salaries.std()**2) / 
                     (len(men_salaries) + len(women_salaries) - 2))
cohens_d = (men_salaries.mean() - women_salaries.mean()) / pooled_std
print(f"\nCohen's d (effect size): {cohens_d:.3f}")
print(f"  → Interpretation: ", end="")
if abs(cohens_d) < 0.2:
    print("Small effect")
elif abs(cohens_d) < 0.5:
    print("Medium effect")
else:
    print("Large effect")
```

---

### 3. Paired T-Test

**Purpose:** Compare means from the same group at different times

**Hypotheses:**
- H₀: μ_diff = 0
- H₁: μ_diff ≠ 0

**When to Use:**
- Before/After studies
- Matched pairs
- Repeated measures

**Python Example:**

```python
# Example: Did training improve test scores?
before = np.array([75, 80, 72, 85, 78, 82, 77, 79, 76, 81])
after = np.array([78, 83, 75, 88, 81, 85, 80, 82, 79, 84])

differences = after - before

# Perform paired t-test
t_stat, p_value = stats.ttest_rel(before, after)

print(f"Before Mean: {before.mean():.2f}")
print(f"After Mean: {after.mean():.2f}")
print(f"Mean Improvement: {differences.mean():.2f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("\n✓ REJECT H₀: Training significantly improved scores")
else:
    print("\n✗ FAIL TO REJECT H₀: No significant improvement")

# Visualize
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Before/After comparison
ax1.plot(range(len(before)), before, 'o-', label='Before', linewidth=2, markersize=8)
ax1.plot(range(len(after)), after, 'o-', label='After', linewidth=2, markersize=8)
ax1.set_xlabel('Student')
ax1.set_ylabel('Score')
ax1.set_title('Before vs After Training')
ax1.legend()
ax1.grid(alpha=0.3)

# Differences
ax2.hist(differences, bins=8, edgecolor='black', alpha=0.7)
ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='No Change')
ax2.axvline(differences.mean(), color='green', linestyle='--', linewidth=2, 
            label=f'Mean Diff = {differences.mean():.1f}')
ax2.set_xlabel('Score Improvement')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of Improvements')
ax2.legend()

plt.tight_layout()
plt.show()
```

---

## Z-Tests

### When to Use

- **Large sample size** (n ≥ 30)
- **Known population standard deviation** (rare in practice)
- Essentially same as t-test but uses normal distribution

**Example:**

```python
# One-sample z-test
from statsmodels.stats.weightstats import ztest

data = np.random.normal(105, 15, 100)  # Large sample
pop_mean = 100

z_stat, p_value = ztest(data, value=pop_mean)
print(f"z-statistic: {z_stat:.4f}")
print(f"p-value: {p_value:.4f}")
```

---

## Practical Workflow: A/B Test

```python
# Complete A/B Test Example
np.random.seed(42)

# Simulate conversion data
n_control = 1000
n_treatment = 1000
control_conversions = np.random.binomial(1, 0.10, n_control)  # 10% baseline
treatment_conversions = np.random.binomial(1, 0.12, n_treatment)  # 12% new design

# Summary statistics
control_rate = control_conversions.mean()
treatment_rate = treatment_conversions.mean()
lift = (treatment_rate - control_rate) / control_rate

print("=" * 60)
print("A/B TEST RESULTS")
print("=" * 60)
print(f"Control Conversion Rate: {control_rate:.2%}")
print(f"Treatment Conversion Rate: {treatment_rate:.2%}")
print(f"Absolute Lift: {treatment_rate - control_rate:.2%}")
print(f"Relative Lift: {lift:.1%}")

# Statistical test (two-proportion z-test)
from statsmodels.stats.proportion import proportions_ztest

count = np.array([treatment_conversions.sum(), control_conversions.sum()])
nobs = np.array([n_treatment, n_control])

z_stat, p_value = proportions_ztest(count, nobs, alternative='larger')

print(f"\nz-statistic: {z_stat:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("\n✓ STATISTICALLY SIGNIFICANT")
    print(f"  The new design performs {lift:.1%} better (p < 0.05)")
    print("  RECOMMENDATION: Deploy new design")
else:
    print("\n✗ NOT STATISTICALLY SIGNIFICANT")
    print("  Insufficient evidence of improvement")
    print("  RECOMMENDATION: Continue testing or abandon")

#Visualize
labels = ['Control', 'Treatment']
rates = [control_rate, treatment_rate]
colors = ['#e74c3c', '#2ecc71']

plt.figure(figsize=(10, 6))
bars = plt.bar(labels, rates, color=colors, edgecolor='black', linewidth=2)
plt.ylabel('Conversion Rate')
plt.title(f'A/B Test Results (p = {p_value:.4f})')
plt.ylim(0, max(rates) * 1.2)

# Add value labels
for bar, rate in zip(bars, rates):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{rate:.2%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.grid(axis='y', alpha=0.3)
plt.show()
```

---

## Common Mistakes

### 1. Confusing "Fail to Reject" with "Accept H₀"
**Wrong:** "We accept the null hypothesis"
**Right:** "We fail to reject the null hypothesis (insufficient evidence)"

### 2. P-hacking
Running multiple tests until finding p < 0.05
**Solution:** Pre-register hypothesis, use Bonferroni correction

### 3. Ignoring Effect Size
p < 0.05 doesn't mean practically important
**Solution:** Always report effect size + confidence intervals

### 4. Violating Assumptions
Using t-test on non-normal data with small n
**Solution:** Check assumptions, use non-parametric alternatives

---

## Summary

| Test | Compares | Use When | Python Function |
|------|----------|----------|-----------------|
| **One-Sample T** | Sample mean vs known value | n < 30, σ unknown | `stats.ttest_1samp()` |
| **Two-Sample T** | Two independent groups | Two groups, n < 30 | `stats.ttest_ind()` |
| **Paired T** | Same group, different times | Before/After | `stats.ttest_rel()` |
| **Z-Test** | Sample mean vs population | n ≥ 30, σ known | `ztest()` |

**Key Principle:** p-value tells you probability of data given H₀, NOT probability H₀ is true!
