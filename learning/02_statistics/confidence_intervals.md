# Confidence Intervals

## The Honesty That Builds Trust: Quantifying Uncertainty

**Scenario:** You're a data scientist at a political polling firm. Election night is 3 days away. Your analysis shows Candidate A at 51% support. The campaign manager asks, "So we're going to win?"

A junior analyst says: **"Yes! 51% beats 50%!"**

You say: **"Our 95% confidence interval is [48%, 54%]. We cannot confidently call this race."**

The campaign adjusts strategy, focuses on key districts, and wins by a narrow margin. Your honesty about uncertainty—communicated through confidence intervals—made the difference.

**Why confidence intervals are more powerful than point estimates:**

**Consider these statements:**
- ❌ **"Average customer lifetime value is $127."** (Sounds precise, but is it?)
- ✅ **"Average CLV is $127, 95% CI: [$115, $139]."** (Now we know the precision!)

The second statement tells you:
- How confident you should be
- Whether a $120 decision threshold is risky
- If you need more data
- If differences between segments are real or noise

**Where confidence intervals save companies millions:**

1. **Product launches:** "Our new feature increases engagement by 8% [95% CI: 3%, 13%]" → Clear ROI, confident investment
2. **Pricing decisions:** "Optimal price point is $29 [95% CI: $27, $31]" → Safe pricing strategy
3. **Quality control:** "Defect rate increased to 2.3% [95% CI: 1.8%, 2.8%]" → Investigate or normal variation?
4. **Clinical trials:** "Blood pressure reduced by 15 mmHg [95% CI: 10, 20]" → FDA approval evidence

**The hidden power:** Confidence intervals communicate **intellectual honesty**. When you present results with CIs:
- Executives trust you more (you acknowledge uncertainty)
- Scientists respect you (proper statistical communication)
- Regulators approve you (required for FDA, academic publishing)
- You make better decisions (you know when NOT to act)

**Here's what separates great data scientists from mediocre ones:**
- **Mediocre:** "Conversion rate increased from 10% to 12%—success!"
- **Great:** "Conversion rate: 12% [10.5%, 13.5%]. This is a **significant, reliable improvement**. Deploy."

One inspires blind action. The other inspires confident, informed action.

**The math you'll learn here isn't theoretical—it's the language of trust, precision, and professional credibility.**

Let's build that credibility.

---

### Quick Decision-Making Example: Pricing Strategy Math

**Situation:** Testing optimal price point for SaaS product

**The calculation:**
```python
# Sample data from pricing experiment
import numpy as np
from scipy import stats

revenue_per_customer = np.array([127, 142, 118, 135, 129, 138, 121, 145, 
                                  131, 140, 125, 136, 128, 143, 132])

# Calculate 95% confidence interval
mean = revenue_per_customer.mean()
std = revenue_per_customer.std(ddof=1)
n = len(revenue_per_customer)

t_critical = stats.t.ppf(0.975, df=n-1)
margin_error = t_critical * (std / np.sqrt(n))

ci_lower = mean - margin_error
ci_upper = mean + margin_error

# Result: Mean = $133, 95% CI: [$127, $139]
```

**The decision:**

**Option A: Price at $125**
- Below CI lower bound ($127)
- **Risk:** Leaving money on table
- **Expected loss:** ~$8 per customer
- With 10,000 customers: **$80,000 annual loss**

**Option B: Price at $135 (within CI)**
- Safely within interval
- **Confidence:** 95% that true value includes this
- **Decision:** ✅ **Safe, optimal pricing**

**Option C: Price at $145**
- Above CI upper bound ($139)
- **Risk:** Price resistance, lost customers
- **Decision:** ❌ **Too risky**

**What the CI tells you:**
- Point estimate ($133) alone → Could pick $125 or $145 (both wrong!)
- CI [$127, $139] → **Actionable range for safe decisions**

**Business impact:** Confidence intervals turned vague data into a $135 optimal price → $1.35M annual revenue (vs $1.25M at $125)

---

### Second Example: Survey Sample Size - Election Polling

**Scenario:** Political campaign wants to know candidate's support level

**Survey Results:** 520 out of 1,000 likely voters support the candidate

**Question:** What's the true support level in the population (with 95% confidence)?

**The Detailed Math:**

```python
import numpy as np
from scipy import stats

# Step 1: Define the data
n = 1000                    # Total number of voters surveyed
support_count = 520        # Number who support candidate
p_hat = support_count / n  # Sample proportion
#       ^^^^^^^^^^^^^^^^^^^^
#       Substitution: 520 / 1000 = 0.52 (52%)

print("===== SURVEY DATA =====")
print(f"Sample size: {n}")
print(f"Support count: {support_count}")
print(f"Sample proportion (p̂): {p_hat:.4f} ({p_hat*100:.1f}%)")

# Step 2: Set confidence level
confidence = 0.95           # 95% confidence interval
alpha = 1 - confidence      # Significance level
#       ^^^^^^^^^^^^^^^^^^
#       Substitution: 1 - 0.95 = 0.05

print(f"\n===== CONFIDENCE LEVEL =====")
print(f"Confidence: {confidence*100:.0f}%")
print(f"Alpha (α): {alpha}")

# Step 3: Calculate standard error
# Formula for proportion: SE = √[p̂(1-p̂) / n]
se = np.sqrt(p_hat * (1 - p_hat) / n)
#    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#    Numerator: p̂(1-p̂) = proportion × (1 - proportion)
#              Substitution: 0.52 × (1 - 0.52) = 0.52 × 0.48 = 0.2496
#    
#    Denominator: n = sample size = 1000
#    
#    SE = √(0.2496 / 1000) = √0.0002496 = 0.0158

print(f"\n===== STANDARD ERROR CALCULATION =====")
print(f"SE = √[p̂(1-p̂) / n]")
print(f"SE = √[{p_hat:.4f} × {(1-p_hat):.4f} / {n}]")
print(f"SE = √[{p_hat*(1-p_hat):.4f} / {n}]")
print(f"SE = √{p_hat*(1-p_hat)/n:.6f}")
print(f"SE = {se:.4f}")
print(f"\nInterpretation: The sample proportion varies by about {se*100:.2f} percentage points")

# Step 4: Find critical z-value
# For 95% CI, we need z where 2.5% is in each tail
z_critical = stats.norm.ppf(1 - alpha/2)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#            ppf = percent point function (inverse of CDF)
#            We want: P(Z ≤ z) = 0.975 (leaves 2.5% in upper tail)
#            Substitution: ppf(0.975) = 1.96

print(f"\n===== CRITICAL VALUE =====")
print(f"For {confidence*100:.0f}% CI, α/2 = {alpha/2}")
print(f"z_critical = norm.ppf({1-alpha/2})")
print(f"z_critical = {z_critical:.4f}")
print(f"\nThis means 95% of data falls within ±{z_critical:.2f} standard deviations")

# Step 5: Calculate margin of error
# Formula: MOE = z × SE
margin_of_error = z_critical * se
#                 ^^^^^^^^^^^^^^^^
#                 Substitution: 1.96 × 0.0158 = 0.031

print(f"\n===== MARGIN OF ERROR =====")
print(f"MOE = z_critical × SE")
print(f"MOE = {z_critical:.4f} × {se:.4f}")
print(f"MOE = {margin_of_error:.4f}")
print(f"MOE as percentage: ±{margin_of_error*100:.2f}%")

# Step 6: Calculate confidence interval
ci_lower = p_hat - margin_of_error
#          ^^^^^^^^^^^^^^^^^^^^^^^^^
#          Substitution: 0.52 - 0.031 = 0.489 (48.9%)

ci_upper = p_hat + margin_of_error
#          ^^^^^^^^^^^^^^^^^^^^^^^^^
#          Substitution: 0.52 + 0.031 = 0.551 (55.1%)

print(f"\n===== 95% CONFIDENCE INTERVAL =====")
print(f"CI = p̂ ± MOE")
print(f"CI = {p_hat:.4f} ± {margin_of_error:.4f}")
print(f"CI = [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"CI as percentage: [{ci_lower*100:.1f}%, {ci_upper*100:.1f}%]")

# Step 7: More accurate Wilson Score Interval
# (Better for extreme proportions, but complex formula)
from statsmodels.stats.proportion import proportion_confint

wilson_lower, wilson_upper = proportion_confint(
    support_count,          # Number of successes
    n,                      # Total trials
    alpha=0.05,            # 1 - confidence
    method='wilson'        # Wilson score method
)

print(f"\n===== WILSON SCORE INTERVAL (More Accurate) =====")
print(f"95% CI: [{wilson_lower:.4f}, {wilson_upper:.4f}]")
print(f"As percentage: [{wilson_lower*100:.1f}%, {wilson_upper*100:.1f}%]")

# Step 8: Visualize the interval
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

# Plot confidence interval
plt.subplot(1, 2, 1)
plt.errorbar(1, p_hat, yerr=margin_of_error, fmt='o', markersize=12, 
             capsize=15, capthick=3, elinewidth=3, color='#2ecc71')
plt.axhline(0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='50% (Tie)')
plt.fill_between([0.5, 1.5], ci_lower, ci_upper, alpha=0.2, color='blue')
plt.xlim(0.5, 1.5)
plt.ylim(0.4, 0.65)
plt.ylabel('Support Proportion', fontsize=12)
plt.title(f'95% Confidence Interval\n[{ci_lower*100:.1f}%, {ci_upper*100:.1f}%]', 
          fontsize=14, fontweight='bold')
plt.xticks([])
plt.grid(axis='y', alpha=0.3)
plt.legend()

# Plot normal distribution
plt.subplot(1, 2, 2)
x = np.linspace(p_hat - 4*se, p_hat + 4*se, 1000)
y = stats.norm.pdf(x, p_hat, se)
plt.plot(x, y, linewidth=2, color='blue')
plt.fill_between(x, y, where=(x >= ci_lower) & (x <= ci_upper), 
                 alpha=0.3, color='green', label='95% CI')
plt.axvline(p_hat, color='red', linestyle='--', linewidth=2, label=f'Sample: {p_hat*100:.1f}%')
plt.axvline(0.5, color='orange', linestyle=':', linewidth=2, label='50% (Tie)')
plt.xlabel('Support Proportion', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.title('Sampling Distribution', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

**Output:**
```
===== SURVEY DATA =====
Sample size: 1000
Support count: 520
Sample proportion (p̂): 0.5200 (52.0%)

===== CONFIDENCE LEVEL =====
Confidence: 95%
Alpha (α): 0.05

===== STANDARD ERROR CALCULATION =====
SE = √[p̂(1-p̂) / n]
SE = √[0.5200 × 0.4800 / 1000]
SE = √[0.2496 / 1000]
SE = √0.000250
SE = 0.0158

Interpretation: The sample proportion varies by about 1.58 percentage points

===== CRITICAL VALUE =====
For 95% CI, α/2 = 0.025
z_critical = norm.ppf(0.975)
z_critical = 1.9600

This means 95% of data falls within ±1.96 standard deviations

===== MARGIN OF ERROR =====
MOE = z_critical × SE
MOE = 1.9600 × 0.0158
MOE = 0.0310
MOE as percentage: ±3.10%

===== 95% CONFIDENCE INTERVAL =====
CI = p̂ ± MOE
CI = 0.5200 ± 0.0310
CI = [0.4890, 0.5510]
CI as percentage: [48.9%, 55.1%]

===== WILSON SCORE INTERVAL (More Accurate) =====
95% CI: [0.4892, 0.5505]
As percentage: [48.9%, 55.1%]
```

**The Strategic Decision Analysis:**

**Interpreting the 95% CI [48.9%, 55.1%]:**

1. **Point estimate: 52%**
   - Sample shows candidate leading
   - But is it enough?

2. **Lower bound: 48.9%**
   - Worst-case scenario (still plausible)
   - **Below 50%!** Could be losing

3. **Upper bound: 55.1%**
   - Best-case scenario
   - Comfortable win

4. **The 50% threshold**
   - CI **includes 50%** (spans 48.9% to 55.1%)
   - Cannot confidently say candidate is winning

**Campaign Decisions:**

**Scenario A: Ignore the CI, trust point estimate**
- "We have 52%, we're winning!"
- Reduce campaign spending
- Relax get-out-the-vote efforts
- **Risk:** Could actually be at 49% → **LOSE ELECTION**

**Scenario B: Use the CI properly** ✓
- "CI includes 50% - race is too close to call"
- **Decision:** Treat as competitive race
- Maintain full campaign intensity
- Focus on swing districts
- Maximize voter turnout
- **Outcome:** Win by mobilizing base

**Mathematical Proof of Uncertainty:**

| Scenario | True Support | Within CI? | Election Result |
|----------|--------------|------------|------------------|
| Worst case | 48.9% | ✓ Yes | **LOSE** |
| Point estimate | 52.0% | ✓ Yes | Win |
| Best case | 55.1% | ✓ Yes | Win big |

**All three scenarios are plausible within 95% CI!**

**Sample Size Impact:**

What if we surveyed MORE people?

```python
# With n = 4,000 (4x larger)
se_large = np.sqrt(0.52 * 0.48 / 4000)  # SE = 0.0079
moe_large = 1.96 * se_large              # MOE = 0.0155 (1.55%)
# New CI: [50.45%, 53.55%]
# Now ENTIRELY above 50% → Confident win!
```

**Key Formulas Demonstrated:**

1. **Sample Proportion:**
   $$\hat{p} = \frac{x}{n}$$

2. **Standard Error (Proportion):**
   $$SE = \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$$

3. **Margin of Error:**
   $$MOE = z_{\alpha/2} \times SE$$

4. **Confidence Interval:**
   $$CI = \hat{p} \pm MOE$$

**The Power of Confidence Intervals:**

**What it prevented:**
- False confidence from point estimate alone
- Strategic errors (reducing campaign effort)
- Election loss from complacency

**What it enabled:**
- Honest assessment of race competitiveness
- Data-driven campaign strategy
- Proper resource allocation
- **Election victory**

**One confidence interval. One election saved.**

---

## Why This Matters

Point estimates (single numbers like "mean = 10.5") don't tell the whole story. Confidence intervals:
- **Quantify uncertainty** in estimates
- **More informative** than just p-values
- **Enable decision-making** with known risk
- **Required for scientific publishing**

Instead of saying "conversion rate is 12%", say **"conversion rate is 12% ± 2% (95% CI: [10%, 14%])"**

---

## What is a Confidence Interval?

**Definition:** A range of values that likely contains the true population parameter with a specified level of confidence.

**95% Confidence Interval means:**
- If we repeated this experiment 100 times, about 95 of those intervals would contain the true parameter
- **NOT:** "95% probability the true value is in this interval"

---

## Confidence Interval for Mean

### Formula (Known σ or Large n)

$$CI = \bar{x} \pm z_{\alpha/2} \times \frac{\sigma}{\sqrt{n}}$$

Where:
- $\bar{x}$ = sample mean
- $z_{\alpha/2}$ = critical z-value
- $\sigma$ = population standard deviation
- $n$ = sample size

**Common z-values:**
- 90% CI: z = 1.645
- 95% CI: z = 1.96
- 99% CI: z = 2.576

### Formula (Unknown σ, Small n)

Use **t-distribution** instead:

$$CI = \bar{x} \pm t_{\alpha/2, df} \times \frac{s}{\sqrt{n}}$$

Where:
- $s$ = sample standard deviation
- $df$ = degrees of freedom (n - 1)

---

## Python Implementation

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Sample data: Customer spending
np.random.seed(42)
spending = np.array([45, 52, 38, 61, 49, 55, 42, 58, 47, 51,
                    53, 46, 50, 44, 57, 48, 54, 41, 56, 43])

# Parameters
n = len(spending)
mean = spending.mean()
std = spending.std(ddof=1)  # Sample std deviation
confidence = 0.95

# Method 1: Using t-distribution (preferred for small samples)
t_critical = stats.t.ppf((1 + confidence) / 2, df=n-1)
margin_of_error = t_critical * (std / np.sqrt(n))

ci_lower = mean - margin_of_error
ci_upper = mean + margin_of_error

print("=" * 60)
print("95% CONFIDENCE INTERVAL FOR MEAN")
print("=" * 60)
print(f"Sample Size: {n}")
print(f"Sample Mean: ${mean:.2f}")
print(f"Sample Std: ${std:.2f}")
print(f"Standard Error: ${std/np.sqrt(n):.2f}")
print(f"\nt-critical (df={n-1}): {t_critical:.4f}")
print(f"Margin of Error: ${margin_of_error:.2f}")
print(f"\n95% Confidence Interval: [${ci_lower:.2f}, ${ci_upper:.2f}]")
print(f"\nInterpretation:")
print(f"  We are 95% confident the true mean spending is")
print(f"  between ${ci_lower:.2f} and ${ci_upper:.2f}")

# Visualize
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(spending, bins=10, edgecolor='black', alpha=0.7)
plt.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean = ${mean:.2f}')
plt.xlabel('Spending ($)')
plt.ylabel('Frequency')
plt.title('Distribution of Spending')
plt.legend()
plt.grid(axis='y', alpha=0.3)

plt.subplot(1, 2, 2)
plt.errorbar(1, mean, yerr=margin_of_error, fmt='o', markersize=10, 
             capsize=10, capthick=2, elinewidth=2, color='#2ecc71')
plt.axhline(ci_lower, color='blue', linestyle=':', alpha=0.5)
plt.axhline(ci_upper, color='blue', linestyle=':', alpha=0.5)
plt.fill_between([0.5, 1.5], ci_lower, ci_upper, alpha=0.2, color='blue')
plt.xlim(0.5, 1.5)
plt.ylabel('Spending ($)')
plt.title(f'95% Confidence Interval\n[${ci_lower:.2f}, ${ci_upper:.2f}]')
plt.xticks([])
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Confidence Interval for Proportion

### Formula

$$CI = \hat{p} \pm z_{\alpha/2} \times \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$$

Where:
- $\hat{p}$ = sample proportion
- $n$ = sample size

### Python Example

```python
# Example: Conversion rate confidence interval
total_visitors = 1000
conversions = 120
conversion_rate = conversions / total_visitors

confidence = 0.95
z_critical = stats.norm.ppf((1 + confidence) / 2)
margin_of_error = z_critical * np.sqrt(conversion_rate * (1 - conversion_rate) / total_visitors)

ci_lower = conversion_rate - margin_of_error
ci_upper = conversion_rate + margin_of_error

print(f"Conversion Rate: {conversion_rate:.2%}")
print(f"95% CI: [{ci_lower:.2%}, {ci_upper:.2%}]")
print(f"\nInterpretation:")
print(f"  We are 95% confident the true conversion rate is")
print(f"  between {ci_lower:.2%} and {ci_upper:.2%}")

# Alternative: Wilson Score Interval (better for small samples)
from statsmodels.stats.proportion import proportion_confint

ci_lower_wilson, ci_upper_wilson = proportion_confint(conversions, total_visitors, 
                                                       alpha=1-confidence, method='wilson')

print(f"\nWilson Score Interval (more accurate):")
print(f"  95% CI: [{ci_lower_wilson:.2%}, {ci_upper_wilson:.2%}]")
```

---

## Interpreting Confidence Intervals

### Correct Interpretations

✅ **"We are 95% confident that the interval [10, 14] contains the true mean"**

✅ **"If we repeated this study many times, 95% of calculated intervals would contain the true parameter"**

✅ **"The true value is plausibly anywhere in this range"**

### Common Misinterpretations

❌ **"There's a 95% probability the true mean is in [10, 14]"**
- Wrong: The true mean is fixed (not random). The interval is random.

❌ **"95% of the data falls in [10, 14]"**
- Wrong: CI is about the mean, not individual observations

❌ **"We are 95% sure our sample mean is in [10, 14]"**
- Wrong: We already know the sample mean exactly!

---

## Width of Confidence Interval

### Factors That Affect Width

**Wider CI:**
- Higher confidence level (99% vs 95%)
- Smaller sample size
- Higher variability in data

**Narrower CI:**
- Lower confidence level (90% vs 95%)
- Larger sample size
- Lower variability in data

**Relationship:**

$$\text{Width} \propto \frac{1}{\sqrt{n}}$$

Doubling sample size reduces CI width by √2 ≈ 0.71 (29% reduction)

### Demonstration

```python
# Show how CI width changes with sample size
sample_sizes = [10, 30, 50, 100, 200, 500, 1000]
ci_widths = []

population = np.random.normal(50, 15, 10000)

for n in sample_sizes:
    sample = np.random.choice(population, size=n, replace=False)
    mean = sample.mean()
    std = sample.std(ddof=1)
    t_crit = stats.t.ppf(0.975, df=n-1)
    margin = t_crit * (std / np.sqrt(n))
    width = 2 * margin
    ci_widths.append(width)

plt.figure(figsize=(10, 6))
plt.plot(sample_sizes, ci_widths, marker='o', linewidth=2, markersize=8)
plt.xlabel('Sample Size (n)')
plt.ylabel('95% CI Width')
plt.title('Confidence Interval Width vs Sample Size')
plt.grid(alpha=0.3)
plt.axhline(10, color='red', linestyle='--', alpha=0.5, label='Target Precision')
plt.legend()
plt.show()

print("Sample Size vs CI Width:")
for n, width in zip(sample_sizes, ci_widths):
    print(f"  n = {n:4d}: Width = {width:.2f}")
```

---

## Bootstrap Confidence Intervals

### Why This Matters

**Problem:** What if you don't know the sampling distribution formula?

**Solution:** Bootstrap resampling — use your data to estimate the sampling distribution!

### How It Works

1. Resample with replacement from your data (10,000 times)
2. Calculate statistic (mean, median, etc.) for each resample
3. Use 2.5th and 97.5th percentiles as 95% CI

### Python Implementation

```python
# Bootstrap for ANY statistic (even when no formula exists!)
def bootstrap_ci(data, statistic_func, n_bootstrap=10000, confidence=0.95):
    """
    Calculate bootstrap confidence interval
    
    statistic_func: function to compute statistic (e.g., np.mean, np.median)
    """
    bootstrap_stats = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        resample = np.random.choice(data, size=n, replace=True)
        # Calculate statistic
        stat = statistic_func(resample)
        bootstrap_stats.append(stat)
    
    # Calculate percentiles
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_stats, lower_percentile)
    ci_upper = np.percentile(bootstrap_stats, upper_percentile)
    
    return ci_lower, ci_upper, bootstrap_stats

# Example: CI for median (no simple formula!)
data = np.array([45, 52, 38, 61, 49, 55, 42, 58, 47, 51])

ci_lower, ci_upper, bootstrap_medians = bootstrap_ci(data, np.median)

print(f"Sample Median: {np.median(data):.2f}")
print(f"95% Bootstrap CI: [{ci_lower:.2f}, {ci_upper:.2f}]")

# Visualize bootstrap distribution
plt.figure(figsize=(10, 6))
plt.hist(bootstrap_medians, bins=50, edgecolor='black', alpha=0.7)
plt.axvline(np.median(data), color='red', linestyle='--', linewidth=2, label='Sample Median')
plt.axvline(ci_lower, color='blue', linestyle=':', linewidth=2, label='95% CI')
plt.axvline(ci_upper, color='blue', linestyle=':', linewidth=2)
plt.xlabel('Median')
plt.ylabel('Frequency')
plt.title('Bootstrap Distribution of Median')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

### Bootstrap Advantages

✅ Works for ANY statistic (median, correlation, ratio, etc.)
✅ No assumptions about distribution
✅ Intuitive and flexible

---

##Difference Between Two Means

```python
# CI for difference between two group means
group_a = np.array([75, 80, 72, 85, 78, 82, 77, 79])
group_b = np.array([82, 88, 85, 90, 84, 87, 83, 89])

mean_a = group_a.mean()
mean_b = group_b.mean()
diff = mean_b - mean_a

# Pooled standard error
n_a, n_b = len(group_a), len(group_b)
var_a, var_b = group_a.var(ddof=1), group_b.var(ddof=1)
se_diff = np.sqrt(var_a/n_a + var_b/n_b)

# t-critical value
df = n_a + n_b - 2
t_crit = stats.t.ppf(0.975, df=df)

# CI for difference
margin = t_crit * se_diff
ci_lower = diff - margin
ci_upper = diff + margin

print(f"Group A Mean: {mean_a:.2f}")
print(f"Group B Mean: {mean_b:.2f}")
print(f"Difference (B - A): {diff:.2f}")
print(f"95% CI for Difference: [{ci_lower:.2f}, {ci_upper:.2f}]")

if ci_lower > 0:
    print("\n✓ Group B is significantly higher (CI doesn't include 0)")
elif ci_upper < 0:
    print("\n✓ Group A is significantly higher (CI doesn't include 0)")
else:
    print("\n✗ No significant difference (CI includes 0)")
```

---

## Practical Guidelines

### Sample Size Planning

**Question:** How large a sample do I need for a desired margin of error?

$$n = \left( \frac{z_{\alpha/2} \times \sigma}{MOE} \right)^2$$

```python
# Example: How many customers to survey for ±$5 margin of error?
sigma_estimate = 20  # Estimated std deviation (from pilot study)
moe_desired = 5
confidence = 0.95

z_crit = stats.norm.ppf((1 + confidence) / 2)
n_required = (z_crit * sigma_estimate / moe_desired) ** 2

print(f"Required sample size: {int(np.ceil(n_required))}")
```

---

## Summary

| Interval Type | Formula | Use When |
|---------------|---------|----------|
| **Mean (σ known)** | $\bar{x} \pm z \frac{\sigma}{\sqrt{n}}$ | Large n or σ known |
| **Mean (σ unknown)** | $\bar{x} \pm t \frac{s}{\sqrt{n}}$ | Small n, σ unknown |
| **Proportion** | $\hat{p} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$ | Binary outcomes |
| **Bootstrap** | Resample + percentiles | ANY statistic |

**Key Insights:**
- CI width ∝ 1/√n (need 4x sample for half the width)
- Higher confidence → wider interval
- **CI that doesn't include null value = significant result**
- Bootstrap works when no formula exists
