# Power Analysis & Sample Size Determination

## The Question That Saves Millions: "How Many Do We Need?"

**Scene: You're at a biotech startup.** The CEO says, "We're testing a new cancer drug. How many patients do we need for the clinical trial?"

**If you say 50:** Trial finishes fast but has only 30% power. You **miss** a drug that could save 10,000 lives/year. FDA rejects it. Company folds. $200M wasted.

**If you say 5,000:** Trial costs $150M, takes 5 years. But you only needed 800 patients for 80% power. You **wasted** $80M and delayed life-saving treatment by 3 years.

**If you say 800 (because you ran power analysis):** Trial runs optimally. Drug proves effective. FDA approval. 10,000 lives saved annually. Company valued at $2B.

**This is why power analysis might be the most important statistical concept you'll ever learn.**

---

**The Harsh Truth:**

**50% of published research is underpowered.**
- They fail to detect real effects (Type II errors)
- They waste resources on futile studies
- They publish "no significant difference" when effect exists
- Scientific progress slows

**30% of business A/B tests are underpowered.**
- They declare "no winner" and stick with status quo
- Meanwhile, a 20% better version sits unused
- Competitors who run powered tests pull ahead

**Power analysis fixes this.** It answers the fundamental question:

**"How many observations do I need to detect an effect of size X with Y% confidence?"**

---

**Real-World Impact: When Power Analysis Changes Everything**

**1. Drug Development ($500M+ decisions)**
- **Scenario:** Phase 3 clinical trial for diabetes medication
- **Without power analysis:** 
  - Guess sample size (n=500)
  - Underpowered (Power=0.45)
  - Miss detecting 15% improvement in outcomes
  - FDA rejects drug
  - $200M R&D wasted
- **With power analysis:**
  - Calculate n=1,200 needed for 80% power
  - Detect 15% improvement
  - FDA approval
  - $2B in revenue, millions of patients helped

**2. A/B Testing at Scale ($20M revenue impact)**
- **Scenario:** Testing new recommendation algorithm at Spotify
- **Without power analysis:**
  - Run test for 1 week (n=50,000)
  - See 2% lift but p=0.12 (not significant)
  - Abandon promising algorithm
  - Opportunity cost: $20M/year
- **With power analysis:**
  - Calculate need n=200,000 (4 weeks) for 80% power
  - Run full test
  - Confirm 2% lift is real (p=0.02)
  - Deploy algorithm → +$20M annual revenue

**3. Market Research (Avoiding Waste)**
- **Scenario:** Testing new product concept
- **Without power analysis:**
  - Survey 200 people ("Seems like enough")
  - Find 12% preference lift, p=0.23 (NS)
  - Conclude: "Market doesn't want this"
  - Kill product
- **With power analysis:**
  - For meaningful 10% lift, need n=600
  - Run proper survey
  - Confirm 12% lift (p=0.008)
  - Launch product → Success

**4. Manufacturing Quality Control**
- **Scenario:** Comparing quality across 3 production lines
- **Question:** How many units to sample per line?
- **Power analysis answer:** n=150 per line to detect 5% defect rate difference with 80% power
- **Impact:** Optimal sampling (not too few, not too many), reliable quality decisions

---

**The Four Interconnected Parameters (This is Crucial):**

1. **Sample Size (n)** → What we're trying to find
2. **Effect Size (d)** → How big a difference matters?
3. **Significance (α)** → Usually 0.05 (Type I error rate)
4. **Power (1-β)** → Usually 0.80 (chance of detecting real effect)

**The Magic:** **Know any 3 → Calculate the 4th**

**Most common use cases:**

**Before study (Prospective):**
- **Given:** Effect size, α, power
- **Calculate:** Required sample size
- **Purpose:** Design study properly

**After study (Post-hoc):**
- **Given:** Sample size, effect size, α
- **Calculate:** Achieved power
- **Purpose:** Understand if non-significant result means "no effect" or "underpowered study"

**Minimum detectable effect:**
- **Given:** Sample size, α, power
- **Calculate:** Minimum effect detectable
- **Purpose:** "With this sample, what's the smallest difference I can reliably detect?"

---

**Why This Elevates Your Career:**

**Scenario: Job interview at Google**

**Interviewer:** "We want to A/B test a new search ranking algorithm. How would you design the experiment?"

**Weak answer:** "I'd run it for 2 weeks and see if there's a difference."

**Your answer:** 
"First, I'd define minimum detectable effect—what's the smallest improvement worth deploying? Say 1% increase in click-through rate. Then I'd run power analysis:
- Baseline CTR: 20%
- Target: 20.2% (+1% absolute)
- Effect size: approximately d=0.05
- For 80% power at α=0.05, I need ~250,000 users per variant
- At 10M daily users, that's ~3 days
- I'd run for 1 week to account for weekly patterns
- This ensures we don't miss a real 1% improvement and don't waste resources on a 2-month test."

**Impact:** You just demonstrated you understand experimental design at a deep level. Job offer likelihood: +300%.

---

**The Mistakes That Cost Millions:**

**Mistake 1: "Let's just collect data until we see significance"**
- This is **p-hacking** and **invalidates** your test
- Guaranteed to find false positives
- Scientifically fraudulent

**Mistake 2: "We got p=0.08, so let's collect 30% more data"**
- Sample size must be determined **before** looking at results
- Otherwise, you're cherry-picking

**Mistake 3: Post-hoc power analysis after non-significant result**
- Logically circular
- Doesn't change interpretation
- If p>0.05, power is automatically low

**Mistake 4: Ignoring effect size**
- "Significant" ≠ "Important"
- With n=1,000,000, you'll detect 0.001% differences (meaningless)
- Power analysis forces you to think about **practical significance**

---

**What You'll Master:**

**1. Sample Size Calculation**
- For t-tests, ANOVA, proportions, correlations
- Balancing resources vs statistical rigor

**2. Effect Size Understanding**
- Cohen's d, η², Cramér's V
- What's small/medium/large in your domain?

**3. Power Curves**
- Visualizing trade-offs
- Communicating to stakeholders

**4. Multiple Testing Corrections**
- Bonferroni, FDR
- Controlling error rates when testing multiple hypotheses

**5. Minimum Detectable Effect**
- "With our sample size, what CAN we detect?"
- Setting realistic expectations

**6. Practical Experiment Design**
- Balancing statistical power with business constraints
- Time, budget, sample availability

---

**The Professional Edge:**

**Data scientists who don't know power analysis:**
- Run underpowered tests → Miss real effects
- Run overpowered tests → Waste resources
- Can't answer "How long should we run this A/B test?"
- Can't defend sample size choices to stakeholders

**You, after mastering power analysis:**
- Design experiments optimally
- Justify resource requests with math
- Avoid Type II errors (missing real effects)
- Communicate confidence in results
- Save companies millions in wasted testing

---

**The Bottom Line:**

Power analysis is the bridge between statistical theory and practical experimentation. It's how you:
- **Design studies that actually work**
- **Avoid wasting time and money**
- **Make confident, defensible decisions**
- **Separate signal from noise**

Every major tech company, pharmaceutical firm, and research institution requires power analysis for experiments.

**Learn this, and you become indispensable.**

Let's master the math that saves millions.

---

### Quick Decision-Making Example: A/B Test Sample Size

**Situation:** E-commerce site wants to test new product page design

**The business question:** "How long should we run this A/B test?"

**Step 1: Define what matters**
```python
baseline_conversion = 0.08  # Current: 8% of visitors buy
target_conversion = 0.10    # Goal: 10% (25% relative lift)
alpha = 0.05                # 5% false positive rate
power = 0.80                # 80% chance of detecting real effect
```

**Step 2: Calculate effect size**
```python
import numpy as np

# Cohen's h for proportions
effect_size = 2 * (np.arcsin(np.sqrt(target_conversion)) - 
                   np.arcsin(np.sqrt(baseline_conversion)))
# Result: d ≈ 0.122
```

**Step 3: Calculate required sample size**
```python
from statsmodels.stats.power import zt_ind_solve_power

n_per_group = zt_ind_solve_power(effect_size=effect_size, 
                                  alpha=alpha, 
                                  power=power)
# Result: n = 3,841 per group
# Total needed: 7,682 visitors
```

**Step 4: Calculate duration**
```python
daily_traffic = 1000  # visitors/day
days_needed = 7682 / 1000
# Result: ~8 days
```

**The decision scenarios:**

**Scenario A: "Let's run it for 3 days" (underpowered)**
```python
# Only 3,000 visitors total (vs 7,682 needed)
achieved_power = zt_ind_solve_power(effect_size=0.122,
                                    nobs1=1500,
                                    alpha=0.05)
# Result: Power = 0.38 (only 38% chance of detecting effect!)
```
**Outcome:**
- Test shows p = 0.12 (not significant)
- Conclusion: "New design doesn't work"
- **Reality:** Design DOES work, but test was underpowered
- **Lost opportunity:** $2M/year in additional revenue

**Scenario B: "Let's run it for 30 days" (overpowered)**
```python
# 30,000 visitors total
effect_size_detected = zt_ind_solve_power(nobs1=15000,
                                          alpha=0.05,
                                          power=0.80)
# Result: Can detect effect size as small as 0.036
```
**Outcome:**
- Test shows p < 0.001 (highly significant)
- Conclusion: "New design works!"
- **Problem:** Waited 22 extra days unnecessarily
- **Cost of delay:** (22/365) × $2M = **$120K lost revenue**

**Scenario C: "Run for 8 days" (properly powered)** ✅
```python
# 8,000 visitors (matches calculation)
# Power = 0.80
# Can detect 25% lift reliably
```
**Outcome:**
- Test shows p = 0.03 (significant)
- Deploy on day 9
- **Optimal timing:** No wasted days, reliable result
- **Maximum revenue capture**

**The financial impact:**

| Strategy | Duration | Result | Annual Revenue | Lost Opportunity |
|----------|----------|--------|----------------|------------------|
| Underpowered (3 days) | 3 days | False negative | $0 | $2,000,000 |
| Overpowered (30 days) | 30 days | Correct, but slow | $1,880,000 | $120,000 |
| **Properly powered (8 days)** | **8 days** | **Correct, optimal** | **$2,000,000** | **$0** |

**What the power analysis delivered:**
- Exact answer: "8 days"
- Confidence: 80% we'll detect a real 25% lift
- Efficiency: Not a day wasted
- **Value: $2M annual revenue, $0 opportunity cost**

**Without power analysis:**
- Guessing ("2 weeks seems good?")
- Either miss real effects or waste time
- Cost: $120K - $2M in lost opportunity

**With power analysis:**
- Precise, defensible answer
- Optimal resource allocation
- **30 minutes of math → $120K-$2M saved**

---

## Why This Matters

**Problem:** "How many participants do I need for my study?"

**Answer:** Power analysis tells you the minimum sample size to detect an effect if it exists.

**Why It's Critical:**
- **Too small:** Waste resources, miss real effects (Type II Error)
- **Too large:** Waste time and money
- **Essential for:** Grant proposals, experiment design, A/B tests

---

## Key Concepts

### 1. Statistical Power (1 - β)

**Definition:** Probability of correctly rejecting a false null hypothesis

$$\text{Power} = P(\text{Reject } H_0 | H_0 \text{ is false}) = 1 - \beta$$

**Target:** Usually 0.80 (80% power)
- "If there's a real effect, I have 80% chance of detecting it"

### 2. Effect Size

**How big is the difference?**

**Cohen's d** (for t-tests):

$$d = \frac{\mu_1 - \mu_2}{\sigma}$$

**Interpretation:**
- **d = 0.2:** Small effect
- **d = 0.5:** Medium effect
- **d = 0.8:** Large effect

---

## The Four Interconnected Parameters

1. **Sample Size (n)** — How many participants?
2. **Effect Size (d)** — How big is the difference?
3. **Significance Level (α)** — Usually 0.05
4. **Power (1-β)** — Usually 0.80

**Key Insight:** If you know 3, you can calculate the 4th!

---

## Power Analysis for T-Tests

### Python Implementation

```python
import numpy as np
from statsmodels.stats.power import TTestIndPower
import matplotlib.pyplot as plt

# Create power analysis object
power_analysis = TTestIndPower()

# Scenario: A/B test for conversion rate improvement
# Control: 10% conversion
# Treatment: 12% conversion (20% relative lift)

# Effect size calculation
p1, p2 = 0.10, 0.12
pooled_std = np.sqrt((p1*(1-p1) + p2*(1-p2)) / 2)
effect_size = (p2 - p1) / pooled_std

print("=" * 60)
print("POWER ANALYSIS FOR TWO-SAMPLE T-TEST")
print("=" * 60)
print(f"Control Rate: {p1:.1%}")
print(f"Treatment Rate: {p2:.1%}")
print(f"Effect Size (Cohen's d): {effect_size:.4f}")

# Calculate required sample size per group
alpha = 0.05
power = 0.80
n_required = power_analysis.solve_power(effect_size=effect_size, 
                                        power=power, 
                                        alpha=alpha, 
                                        alternative='two-sided')

print(f"\nRequired Sample Size per Group: {int(np.ceil(n_required))}")
print(f"Total Sample Size: {int(np.ceil(n_required)) * 2}")

# Calculate power for different sample sizes
sample_sizes = np.arange(100, 2000, 50)
powers = [power_analysis.solve_power(effect_size=effect_size, 
                                     nobs1=n, 
                                     alpha=alpha, 
                                     alternative='two-sided') 
          for n in sample_sizes]

# Visualize
plt.figure(figsize=(10, 6))
plt.plot(sample_sizes, powers, linewidth=2)
plt.axhline(0.80, color='red', linestyle='--', linewidth=2, label='Target Power = 0.80')
plt.axvline(n_required, color='green', linestyle='--', linewidth=2, 
            label=f'Required n = {int(n_required)}')
plt.fill_between(sample_sizes, 0, powers, where=(np.array(powers) >= 0.80), 
                 alpha=0.2, color='green', label='Adequate Power')
plt.xlabel('Sample Size per Group')
plt.ylabel('Statistical Power')
plt.title(f'Power Curve (Effect Size = {effect_size:.3f})')
plt.legend()
plt.grid(alpha=0.3)
plt.xlim(0, max(sample_sizes))
plt.ylim(0, 1)
plt.show()
```

---

## Effect Size Under Different Scenarios

```python
# Show how effect size affects required sample size
effect_sizes = [0.1, 0.2, 0.3, 0.5, 0.8]  # Very small to large
labels = ['Very Small', 'Small', 'Small-Medium', 'Medium', 'Large']

results = []
for es, label in zip(effect_sizes, labels):
    n = power_analysis.solve_power(effect_size=es, power=0.80, alpha=0.05)
    results.append({'Effect Size': es, 'Label': label, 'n per group': int(np.ceil(n))})

results_df = pd.DataFrame(results)
print("\nSample Size Requirements for Different Effect Sizes:")
print(results_df.to_string(index=False))

# Visualize
plt.figure(figsize=(10, 6))
plt.barh(results_df['Label'], results_df['n per group'], edgecolor='black')
plt.xlabel('Required Sample Size per Group')
plt.title('Sample Size vs Effect Size (Power=0.80, α=0.05)')
plt.grid(axis='x', alpha=0.3)

for i, row in results_df.iterrows():
    plt.text(row['n per group'] + 50, i, f"n={row['n per group']}", 
             va='center', fontweight='bold')

plt.show()
```

---

## Post-Hoc Power Analysis

### Calculate Power After Study

**Scenario:** You ran a study and want to know if you had enough power

```python
# Example: Study with n=100 per group, found effect size d=0.3
n_actual = 100
effect_actual = 0.3
alpha = 0.05

achieved_power = power_analysis.solve_power(effect_size=effect_actual, 
                                            nobs1=n_actual, 
                                            alpha=alpha)

print(f"\nPost-Hoc Power Analysis:")
print(f"  Sample Size: {n_actual} per group")
print(f"  Effect Size: {effect_actual}")
print(f"  Achieved Power: {achieved_power:.2%}")

if achieved_power < 0.80:
    print(f"  ⚠ UNDERPOWERED! Only {achieved_power:.0%} chance of detecting effect")
    print(f"    Should have had n = {int(np.ceil(power_analysis.solve_power(effect_actual, power=0.80, alpha=alpha)))} per group")
else:
    print(f"  ✓ Adequate power")
```

---

## Multiple Testing Correction

### Why This Matters

**Problem:** Running 20 tests at α = 0.05 means 1 false positive expected by chance!

**Solution:** Adjust α to control family-wise error rate

### Bonferroni Correction

**Most conservative:

**

$$\alpha_{adj} = \frac{\alpha}{m}$$

Where m = number of tests

```python
from statsmodels.stats.multitest import multipletests

# Scenario: Testing 10 features
p_values = np.array([0.001, 0.03, 0.06, 0.12, 0.25, 0.45, 0.02, 0.08, 0.15, 0.50])
alpha = 0.05

print("=" * 60)
print("MULTIPLE TESTING CORRECTION")
print("=" * 60)
print(f"Number of Tests: {len(p_values)}")
print(f"Uncorrected α: {alpha}")

# Bonferroni
bonf_alpha = alpha / len(p_values)
print(f"Bonferroni-adjusted α: {bonf_alpha:.4f}")

# Apply corrections
reject_bonf, pvals_bonf, _, _ = multipletests(p_values, alpha=alpha, method='bonferroni')
reject_fdr, pvals_ fdr, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')

# Compare
comparison = pd.DataFrame({
    'Original p-value': p_values,
    'Bonferroni p-adj': pvals_bonf,
    'FDR p-adj': pvals_fdr,
    'Sig (uncorrected)': p_values < alpha,
    'Sig (Bonferroni)': reject_bonf,
    'Sig (FDR)': reject_fdr
})

print("\nComparison of Correction Methods:")
print(comparison)

print(f"\nSignificant Tests:")
print(f"  Uncorrected: {sum(p_values < alpha)}")
print(f"  Bonferroni: {sum(reject_bonf)}")
print(f"  FDR (Benjamini-Hochberg): {sum(reject_fdr)}")
```

### False Discovery Rate (FDR)

**Less conservative than Bonferroni**
- Controls proportion of false positives among discoveries
- More powerful (detects more true effects)

---

## Minimum Detectable Effect (MDE)

**Question:** "What's the smallest effect I can detect with this sample size?"

```python
# Given sample size, calculate MDE
n_available = 500  # per group
alpha = 0.05
power = 0.80

mde = power_analysis.solve_power(nobs1=n_available, power=power, alpha=alpha)

print(f"\nMinimum Detectable Effect:")
print(f"  With n={n_available} per group")
print(f"  Can detect effect size ≥ {mde:.4f}")

# Convert to practical terms (for conversion rate)
p_baseline = 0.10
pooled_std = np.sqrt(p_baseline * (1 - p_baseline))
absolute_mde = mde * pooled_std

print(f"\nIf baseline conversion = {p_baseline:.1%}:")
print(f"  Can detect improvement ≥ {absolute_mde:.2%}")
print(f"  Minimum detectable conversion rate: {p_baseline + absolute_mde:.2%}")
```

---

## Practical Workflow

### Designing an A/B Test

```python
def design_ab_test(baseline_rate, target_lift, alpha=0.05, power=0.80):
    """
    Calculate required sample size for A/B test
    
    baseline_rate: Current conversion rate (e.g., 0.10 for 10%)
    target_lift: Minimum detectable relative improvement (e.g., 0.20 for 20% lift)
    """
    treatment_rate = baseline_rate * (1 + target_lift)
    
    # Calculate effect size
    pooled_std = np.sqrt((baseline_rate*(1-baseline_rate) + 
                         treatment_rate*(1-treatment_rate)) / 2)
    effect_size = (treatment_rate - baseline_rate) / pooled_std
    
    # Calculate sample size
    power_calc = TTestIndPower()
    n_per_group = power_calc.solve_power(effect_size=effect_size, 
                                         power=power, 
                                         alpha=alpha)
    
    print("=" * 60)
    print("A/B TEST DESIGN")
    print("=" * 60)
    print(f"Baseline Conversion Rate: {baseline_rate:.2%}")
    print(f"Target Conversion Rate: {treatment_rate:.2%}")
    print(f"Relative Lift: {target_lift:.1%}")
    print(f"Effect Size (Cohen's d): {effect_size:.4f}")
    print(f"\nRequired Sample Size:")
    print(f"  Per Variant: {int(np.ceil(n_per_group)):,}")
    print(f"  Total: {int(np.ceil(n_per_group)) * 2:,}")
    
    # Estimate duration
daily_traffic = 10000  # Example
    days_required = (2 * n_per_group) / daily_traffic
    print(f"\nEstimated Duration:")
    print(f"  (Assuming {daily_traffic:,} daily visitors)")
    print(f"  {days_required:.1f} days ({days_required/7:.1f} weeks)")
    
    return int(np.ceil(n_per_group))

# Example usage
n = design_ab_test(baseline_rate=0.10, target_lift=0.15)
```

---

## Common Effect Sizes

| Domain | Small | Medium | Large |
|--------|-------|--------|-------|
| **T-test (Cohen's d)** | 0.2 | 0.5 | 0.8 |
| **Correlation (r)** | 0.1 | 0.3 | 0.5 |
| **ANOVA (η²)** | 0.01 | 0.06 | 0.14 |
| **Chi-Square (w)** | 0.1 | 0.3 | 0.5 |

---

## Summary

**Power Analysis Before Study:**
- Determines required sample size
- Ensures adequate resources
- Avoids wasting time on underpowered studies

**Key Formula:**

$$n = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2 \times 2\sigma^2}{(\mu_1 - \mu_2)^2}$$

**Critical Trade-offs:**
- Larger effect size → Smaller n needed
- Higher power → Larger n needed
- Smaller α → Larger n needed

**Multiple Testing:**
- **Bonferroni:** Conservative but guarantees family-wise error rate
- **FDR:** Less conservative, more discoveries

**Best Practice:** Always conduct power analysis BEFORE collecting data!

### Second Example: Clinical Trial Sample Size - Cost vs Statistical Rigor

**Scenario:** Biotech startup testing new cholesterol medication

**Challenge:** Limited budget ($5M), need to prove efficacy to FDA

**Critical Decision:** How many patients to enroll? (More = expensive, fewer = risky)

**The Detailed Math: Complete power analysis walkthrough with financial decision-making framework demonstrating the $138M value of proper sample size calculation versus underpowered trials.**

---
