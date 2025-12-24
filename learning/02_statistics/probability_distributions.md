# Probability Distributions

## The Power Behind the Math: Why This Changes Everything

**Imagine:** You're a data scientist at a healthcare company. A hospital calls: "Our patients are dying at unusual rates. Is this random chance or something wrong?" You open your laptop. Within minutes, using probability distributions, you calculate that there's only a 0.001% chance this is random. You've just identified a life-threatening issue that saves hundreds of lives.

**Or consider:** You're at Netflix, and executives ask, "How many new shows should we produce?" Using the Central Limit Theorem, you model user behavior across millions of subscribers. Your statistical analysis predicts optimal content investment, directly impacting a $200M budget decision.

**This is the power of probability distributions.** They transform raw data into actionable intelligence. They're not abstract formulas—they're the mathematical foundation that powers:
- **Drug approval decisions** (Is this medicine safe? Effective?)
- **Financial trading algorithms** (What's the risk? Expected return?)
- **Quality control** (Is our manufacturing process broken?)
- **Machine learning** (How confident are we in this prediction?)

Every statistical test, every confidence interval, every p-value—they all rest on understanding distributions. Master this, and you unlock the ability to **quantify uncertainty**, the single most valuable skill in data science.

**Here's the truth:** Companies don't pay data scientists for code. They pay for **certainty in uncertain situations**. Probability distributions give you that power.

Let's dive in.

---

### Quick Decision-Making Example: The Math That Saved a Hospital

**Situation:** Hospital ICU shows 15 deaths in January (normally 8-10/month)

**The calculation:**
```python
# Using Poisson distribution (rare events)
average_monthly_deaths = 9
observed_deaths = 15

# Probability of 15+ deaths if normal rate is 9
from scipy.stats import poisson
p_value = 1 - poisson.cdf(14, mu=9)
# Result: p = 0.0089 (0.89%)
```

**The decision:**
- p < 0.01 means less than 1% chance this is random
- **Action:** Immediate investigation
- **Discovery:** Ventilator contamination
- **Outcome:** Issue fixed, lives saved

**Without this math:** "Seems high, but maybe just bad luck this month" → More deaths

**With this math:** "99% confident something is wrong" → Immediate action

---

### Second Example: Quality Control at a Semiconductor Plant

**Scenario:** Manufacturing plant produces 10,000 chips/day. Acceptable defect rate: 2%

**Observation:** Today's batch has 250 defective chips (2.5%)

**Question:** Is this normal variation or a process failure?

**The Detailed Math:**

```python
# Step 1: Define the problem using Binomial Distribution
# Why Binomial? We have n independent trials (chips), each can be defect/non-defect

n = 10000              # Number of chips tested (total trials)
p_acceptable = 0.02    # Acceptable defect rate (probability of defect per chip)
observed_defects = 250 # What we actually saw today

# Step 2: Calculate expected defects under normal conditions
expected_defects = n * p_acceptable
# Substitution: expected = 10,000 × 0.02 = 200 defects
# This is what we SHOULD see if process is normal

print(f"Expected defects (normal process): {expected_defects}")
print(f"Observed defects (today): {observed_defects}")
print(f"Difference: {observed_defects - expected_defects} extra defects")

# Step 3: Calculate probability of seeing 250+ defects by chance
from scipy.stats import binom

# P(X ≥ 250) when true rate is 2%
# We use survival function: P(X ≥ k) = 1 - P(X ≤ k-1)
p_value = 1 - binom.cdf(249, n, p_acceptable)
#                        ^^^  ^  ^^^^^^^^^^^^^^
#                         |   |        |
#                         |   |        +-- True defect probability (2%)
#                         |   +-- Number of trials (10,000 chips)
#                         +-- We want P(X ≥ 250), so CDF up to 249

print(f"\nProbability of 250+ defects by chance: {p_value:.6f}")
print(f"As percentage: {p_value * 100:.4f}%")

# Step 4: Calculate standard deviation to understand variation
std_dev = np.sqrt(n * p_acceptable * (1 - p_acceptable))
#         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#         For binomial: σ = √(n × p × (1-p))
#         Substitution: σ = √(10,000 × 0.02 × 0.98)
#                         = √(196) = 14 defects

print(f"\nStandard deviation: {std_dev:.2f} defects")

# Step 5: Calculate z-score (how many standard deviations away)
z_score = (observed_defects - expected_defects) / std_dev
#          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#          z = (observed - expected) / standard_deviation
#          z = (250 - 200) / 14 = 3.57

print(f"Z-score: {z_score:.2f}")
print(f"Interpretation: {z_score:.2f} standard deviations above normal")
```

**Output:**
```
Expected defects (normal process): 200.0
Observed defects (today): 250
Difference: 50 extra defects

Probability of 250+ defects by chance: 0.000179
As percentage: 0.0179%

Standard deviation: 14.00 defects
Z-score: 3.57
Interpretation: 3.57 standard deviations above normal
```

**The Decision Analysis:**

**Interpreting the Results:**

1. **P-value = 0.000179 (0.0179%)**
   - Less than 0.02% chance this is random
   - 99.98% confident something is wrong

2. **Z-score = 3.57**
   - More than 3 standard deviations away
   - Extremely rare (beyond 99.7% rule)

3. **Expected vs Observed:**
   - Should see: 200 defects
   - Actually saw: 250 defects  
   - Difference: 50 defects (25% increase)

**Decision Tree:**

| P-value Range | Decision | Action |
|---------------|----------|--------|
| p > 0.05 | Normal variation | Continue production |
| 0.01 < p ≤ 0.05 | Borderline | Monitor closely |
| 0.001 < p ≤ 0.01 | Likely problem | Investigate |
| **p ≤ 0.001** | **Definite problem** | **STOP production immediately** |

**Our case: p = 0.000179 → STOP PRODUCTION**

**Financial Impact:**

**If we ignore the signal:**
- Continue producing defective chips
- 50 extra defects/day × $500/chip = $25,000 daily loss
- Customer returns, reputation damage
- Estimated total cost: $500,000

**If we act on the math:**
- Stop production: $50,000 (one day downtime)
- Find root cause: Contaminated raw material
- Fix issue: $10,000
- **Total cost: $60,000**
- **Savings: $440,000**

**Key Learning:**
- **Binomial distribution** models defect counting
- **P-value** quantifies "how unusual" the observation is
- **Z-score** provides intuitive scale (standard deviations)
- **Small p-value (< 0.001)** = take immediate action

**The Power of Probability Distributions:**
Turned "seems like more defects than usual" into "99.98% certain there's a problem" → Confident, defensible action → $440K saved

---

## Why This Matters

Probability distributions are the foundation of statistical inference. Understanding them allows you to:
- Model real-world phenomena mathematically
- Make predictions with quantified uncertainty
- Choose appropriate statistical tests
- Simulate data for experiment design
- Understand sampling variability

---

## 1. Normal (Gaussian) Distribution

### The Most Important Distribution

**Why:** Many natural phenomena follow this pattern, and the Central Limit Theorem guarantees normality of sample means.

### Characteristics

- **Shape:** Bell-shaped, symmetric around mean (μ)
- **Parameters:** Mean (μ) and Standard Deviation (σ)
- **Range:** -∞ to +∞
- **Total Probability:** Area under curve = 1

### Probability Density Function (PDF)

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}$$

### The 68-95-99.7 Rule (Empirical Rule)

- **68%** of data within μ ± 1σ
- **95%** of data within μ ± 2σ
- **99.7%** of data within μ ± 3σ

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Parameters
mu, sigma = 100, 15  # IQ scores

# Generate data
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
pdf = stats.norm.pdf(x, mu, sigma)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(x, pdf, 'b-', linewidth=2, label='Normal Distribution')

# Shade regions
plt.fill_between(x, pdf, where=(x >= mu-sigma) & (x <= mu+sigma), 
                 alpha=0.3, label='68% (μ±σ)')
plt.fill_between(x, pdf, where=(x >= mu-2*sigma) & (x <= mu+2*sigma), 
                 alpha=0.2, label='95% (μ±2σ)')
plt.fill_between(x, pdf, where=(x >= mu-3*sigma) & (x <= mu+3*sigma), 
                 alpha=0.1, label='99.7% (μ±3σ)')

plt.axvline(mu, color='red', linestyle='--', linewidth=2, label=f'Mean={mu}')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Normal Distribution with Empirical Rule')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Calculate probabilities
prob_above_130 = 1 - stats.norm.cdf(130, mu, sigma)
print(f"P(X > 130) = {prob_above_130:.4f} ({prob_above_130*100:.2f}%)")
print(f"This means {prob_above_130*100:.1f}% have IQ above 130")
```

### Testing for Normality

```python
from scipy.stats import shapiro, normaltest, kstest

# Generate sample data
np.random.seed(42)
data = np.random.normal(100, 15, 1000)

# 1. Shapiro-Wilk Test (best for n < 5000)
stat, p_value = shapiro(data)
print(f"Shapiro-Wilk Test: p-value = {p_value:.4f}")
if p_value > 0.05:
    print("  → Data appears normally distributed")
else:
    print("  → Data does NOT appear normal")

# 2. D'Agostino's K² Test
stat, p_value = normaltest(data)
print(f"\nD'Agostino K² Test: p-value = {p_value:.4f}")

# 3. Visual: Q-Q Plot
import scipy.stats as stats
import pylab

plt.figure(figsize=(10, 6))
stats.probplot(data, dist="norm", plot=pylab)
pylab.title("Q-Q Plot (Points should follow diagonal)")
pylab.grid(alpha=0.3)
pylab.show()
```

### Real-World Applications

- **Heights, weights** of populations
- **Measurement errors** in instruments
- **Test scores** (SAT, IQ tests)
- **Financial returns** (approximately, with fat tails)
- **Process quality control** (Six Sigma)

---

## 2. Binomial Distribution

### Definition

Models the **number of successes** in a **fixed number of independent trials**.

### Why This Matters

Essential for:
- A/B testing (conversion rates)
- Quality control (defect rates)
- Medical trials (success rates)
- Survey sampling

### Parameters

- **n:** Number of trials
- **p:** Probability of success on each trial

### Probability Mass Function (PMF)

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

Where $\binom{n}{k} = \frac{n!}{k!(n-k)!}$

### Properties

- **Mean:** μ = n × p
- **Variance:** σ² = n × p × (1 - p)
- **Standard Deviation:** σ = √(n × p × (1 - p))

### Python Implementation

```python
from scipy.stats import binom

# Example: 20 coin flips
n, p = 20, 0.5

# Probability of exactly 12 heads
prob_12 = binom.pmf(12, n, p)
print(f"P(X = 12) = {prob_12:.4f}")

# Probability of 12 or more heads
prob_12_plus = 1 - binom.cdf(11, n, p)
print(f"P(X >= 12) = {prob_12_plus:.4f}")

# Visualize
x = np.arange(0, n+1)
pmf = binom.pmf(x, n, p)

plt.figure(figsize=(12, 6))
plt.bar(x, pmf, alpha=0.7, edgecolor='black')
plt.axvline(n*p, color='red', linestyle='--', linewidth=2, 
            label=f'Expected Value = {n*p}')
plt.xlabel('Number of Successes')
plt.ylabel('Probability')
plt.title(f'Binomial Distribution (n={n}, p={p})')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.show()
```

### Real-World Example: A/B Test

```python
# Scenario: Testing new website design
# Control: 1000 visitors, 100 conversions (10%)
# Treatment: 1000 visitors, 120 conversions (12%)

n_control = 1000
conversions_control = 100
p_control = conversions_control / n_control

n_treatment = 1000
conversions_treatment = 120
p_treatment = conversions_treatment / n_treatment

# Is 120 conversions significantly different from expected 100?
# Under null hypothesis: treatment has same 10% conversion
prob_120_or_more = 1 - binom.cdf(119, n_treatment, p_control)
print(f"P(X >= 120 | p=0.10) = {prob_120_or_more:.4f}")
if prob_120_or_more < 0.05:
    print("  → Significant improvement!")
else:
    print("  → Not significantly different")
```

---

## 3. Poisson Distribution

### Definition

Models the **number of events** occurring in a **fixed interval** of time or space.

### Why This Matters

Perfect for:
- Website traffic (visits per hour)
- Customer arrivals (customers per minute)
- Defects in manufacturing (defects per batch)
- Call center volume (calls per hour)
- Rare events (accidents per year)

### Parameter

- **λ (lambda):** Average rate of occurrence

### Probability Mass Function

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

### Properties

- **Mean:** μ = λ
- **Variance:** σ² = λ
- **Standard Deviation:** σ = √λ

### Python Implementation

```python
from scipy.stats import poisson

# Example: Average 5 customers per hour
lambda_rate = 5

# Probability of exactly 3 customers
prob_3 = poisson.pmf(3, lambda_rate)
print(f"P(X = 3) = {prob_3:.4f}")

# Probability of 8 or more customers (staffing decision)
prob_8_plus = 1 - poisson.cdf(7, lambda_rate)
print(f"P(X >= 8) = {prob_8_plus:.4f}")
print(f"  → Plan for at least 8 customers {prob_8_plus*100:.1f}% of the time")

# Visualize
x = np.arange(0, 15)
pmf = poisson.pmf(x, lambda_rate)

plt.figure(figsize=(12, 6))
plt.bar(x, pmf, alpha=0.7, edgecolor='black')
plt.axvline(lambda_rate, color='red', linestyle='--', linewidth=2,
            label=f'λ = {lambda_rate}')
plt.xlabel('Number of Events')
plt.ylabel('Probability')
plt.title(f'Poisson Distribution (λ={lambda_rate})')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.show()
```

### Real-World Example: Call Center Staffing

```python
# Average 8 calls per hour
lambda_calls = 8

# Need to handle at least 95% of scenarios
# Find minimum staff capacity
for k in range(0, 20):
    prob = poisson.cdf(k, lambda_calls)
    if prob >= 0.95:
        print(f"Staff for {k} simultaneous calls to handle 95% of scenarios")
        break

# Visualize capacity planning
x = np.arange(0, 20)
cdf = poisson.cdf(x, lambda_calls)

plt.figure(figsize=(10, 6))
plt.plot(x, cdf, marker='o', linewidth=2)
plt.axhline(0.95, color='red', linestyle='--', label='95% Service Level')
plt.xlabel('Capacity (Number of Lines)')
plt.ylabel('Probability of Handling All Calls')
plt.title('Call Center Capacity Planning')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

---

## 4. Exponential Distribution

### Definition

Models the **time between events** in a Poisson process.

### Why This Matters

Essential for:
- Customer service (time until next customer)
- System reliability (time until failure)
- Queue management (wait times)
- Survival analysis

### Parameter

- **λ (lambda):** Rate parameter (events per unit time)

### Probability Density Function

$$f(x) = \lambda e^{-\lambda x}$$

### Properties

- **Mean:** μ = 1/λ
- **Variance:** σ² = 1/λ²
- **Memoryless property:** P(X > s+t | X > s) = P(X > t)

### Python Implementation

```python
from scipy.stats import expon

# Example: Average time between customers = 10 minutes
mean_time = 10
lambda_rate = 1 / mean_time

# Probability wait time < 5 minutes
prob_less_5 = expon.cdf(5, scale=mean_time)
print(f"P(Wait < 5 min) = {prob_less_5:.4f}")

# Probability wait time > 20 minutes
prob_more_20 = 1 - expon.cdf(20, scale=mean_time)
print(f"P(Wait > 20 min) = {prob_more_20:.4f}")

# Visualize
x = np.linspace(0, 50, 1000)
pdf = expon.pdf(x, scale=mean_time)

plt.figure(figsize=(12, 6))
plt.plot(x, pdf, linewidth=2, label=f'Mean={mean_time} min')
plt.fill_between(x, pdf, where=(x <= 5), alpha=0.3, label='Wait < 5 min')
plt.axvline(mean_time, color='red', linestyle='--', label='Mean Wait Time')
plt.xlabel('Time (minutes)')
plt.ylabel('Probability Density')
plt.title('Exponential Distribution: Time Between Customers')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

---

## 5. Central Limit Theorem (CLT)

### The Most Important Theorem in Statistics

**Statement:** The sampling distribution of the sample mean approaches a normal distribution as sample size increases, **regardless of the population's distribution**.

### Why This Is Game-Changing

- Use normal-based statistical tests even if data isn't normal
- Foundation of confidence intervals
- Enables hypothesis testing
- Justifies many ML assumptions

### Mathematical Form

If X₁, X₂, ..., Xₙ are random samples from any distribution with mean μ and standard deviation σ:

$$\bar{X} \sim N\left(\mu, \frac{\sigma^2}{n}\right)$$

### Key Insights

1. Mean of sample means = Population mean
2. **Standard Error** = σ/√n (decreases as n increases)
3. Works for **any** original distribution (n ≥ 30 is rule of thumb)

### Demonstration

```python
# Start with HIGHLY NON-NORMAL population
np.random.seed(42)
population = np.random.exponential(scale=2, size=100000)  # Skewed!

plt.figure(figsize=(15, 10))

# Plot 1: Population (NOT normal)
plt.subplot(2, 3, 1)
plt.hist(population, bins=100, edgecolor='black', alpha=0.7)
plt.title('Population: Exponential\n(Highly Skewed)', fontweight='bold')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Demonstrate CLT with different sample sizes
sample_sizes = [5, 10, 30, 100, 500]

for idx, n in enumerate(sample_sizes, 2):
    # Take 1000 samples of size n, compute mean of each
    sample_means = []
    for _ in range(1000):
        sample = np.random.choice(population, size=n, replace=False)
        sample_means.append(sample.mean())
    
    sample_means = np.array(sample_means)
    
    plt.subplot(2, 3, idx)
    plt.hist(sample_means, bins=30, density=True, alpha=0.7, edgecolor='black')
    
    # Overlay theoretical normal distribution
    mu = population.mean()
    sigma = population.std() / np.sqrt(n)  # Standard Error
    x = np.linspace(sample_means.min(), sample_means.max(), 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2,
            label='Theoretical Normal')
    
    plt.title(f'Sample Size n={n}\n(Distribution of Sample Means)', fontweight='bold')
    plt.xlabel('Sample Mean')
    plt.ylabel('Density')
    plt.legend()
    plt.axvline(mu, color='green', linestyle='--', label='Population Mean')

plt.tight_layout()
plt.show()

print(f"Population Mean: {population.mean():.2f}")
print(f"Population Std: {population.std():.2f}")
print("\nNotice: As n increases, sample means become more normally distributed!")
```

### Practical Application: Confidence Intervals

```python
# Even with skewed data, we can use normal-based CI for the mean

sample = np.random.exponential(scale=50, size=100)  # Skewed spending data
sample_mean = sample.mean()
sample_std = sample.std()
n = len(sample)

# 95% Confidence Interval using CLT
confidence = 0.95
z_critical = stats.norm.ppf((1 + confidence) / 2)
margin_of_error = z_critical * (sample_std / np.sqrt(n))

ci_lower = sample_mean - margin_of_error
ci_upper = sample_mean + margin_of_error

print(f"Sample Mean: ${sample_mean:.2f}")
print(f"95% Confidence Interval: [${ci_lower:.2f}, ${ci_upper:.2f}]")
print("Thanks to CLT, this is valid even though the data is skewed!")
```

---

## Summary

| Distribution | Use Case | Parameters | Example |
|--------------|----------|------------|---------|
| **Normal** | Continuous, symmetric data | μ, σ | Heights, test scores |
| **Binomial** | Count of successes in n trials | n, p | Conversion rates, defect counts |
| **Poisson** | Count of events in interval | λ | Website visits, customer arrivals |
| **Exponential** | Time between events | λ | Wait times, time to failure |

**Central Limit Theorem:** Enables using normal distribution methods even when data isn't normal, as long as you're analyzing sample means with sufficient sample size (n ≥ 30).
