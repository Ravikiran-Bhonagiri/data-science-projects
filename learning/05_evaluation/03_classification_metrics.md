# Classification Metrics: Why Accuracy is Useless

**The Fraud Detection Trap:**

Your bank has 1M transactions. 100 are fraudulent (0.01% fraud rate).

**Naive Model:**
```python
# Predict "Not Fraud" for everything
predictions = np.zeros(len(y_test))
accuracy = accuracy_score(y_test, predictions)
# Accuracy: 99.99%
```

**You show this to your boss:** "99.99% accuracy! Ship it!"

**What actually happens:**
- Model catches: 0 fraud cases
- Fraud losses: $1.2M/day
- **You get fired.**

**The problem:** Accuracy treats all errors equally. On imbalanced data, it's meaningless.

---

## The Confusion Matrix: A Map of Failure Types

**Real Case: Spam Filter**

|  | Predicted: Ham | Predicted: Spam |
|---|---|---|
| **Actual: Ham** | 9,500 ✓<br>*Correct emails* | 50 ✗<br>*Job offer sent to spam* |
| **Actual: Spam** | 200 ✗<br>*Viagra ads in inbox* | 250 ✓<br>*Blocked spam* |

**Cost Analysis:**
- **False Positive (FP):** Missed job offer = $150k salary opportunity lost
- **False Negative (FN):** See one viagra ad = Mild annoyance

**Which is worse?** Obviously FP. But accuracy treats them equally.

**Better metric:** Precision = 250/(250+50) = 83.3%
"When I say it's spam, I'm right 83% of the time"

---

## Multi-Class: The Image Classification Problem

**Dataset:** Medical X-rays (3 classes)
- Normal: 10,000 images
- Pneumonia: 500 images  
- COVID-19: 50 images

**Model Performance:**
- Normal: 98% accuracy
- Pneumonia: 65% accuracy
- COVID-19: 12% accuracy

**Macro Average:** (98 + 65 + 12) / 3 = 58.3%
**Micro Average:** (9,800 + 325 + 6) / 10,550 = 96.1%

**Which one to report?**
- **Micro:** If you care about overall patient volume
- **Macro:** If **missing COVID is as bad as missing Normal** (it usually is)

---

## Production Metrics

## 2. Multi-Class Metrics (Micro vs Macro)

What if you are classifying Images (Dog, Cat, Bird)?
You check accuracy for each class:
*   Dog: 90%
*   Cat: 80%
*   Bird: 20% (It sucks at birds)

How do you average these?

### A. Macro Average (Treat classes equally)
$$Macro = \frac{90 + 80 + 20}{3} = 63.3\%$$
*   **Use when:** Minority classes matter. You want to know if it fails on Birds, even if Birds are rare.

### B. Micro Average (Treat samples equally)
*   If there are 100 Dogs, 100 Cats, and 1 Bird.
*   The model gets 181 right out of 201.
*   **Micro Score:** ~90%.
*   **Use when:** You care about overall volume, and don't mind ignoring the rare bird.

---

## 3. Cohen's Kappa

Accuracy includes "Luck."
If you flip a coin, you get 50% accuracy.
**Cohen's Kappa** measures how much better you did than **Random Chance**.

*   **Kappa = 0:** You are random.
*   **Kappa = 1:** You are god.
*   **Kappa < 0:** You are actively worse than random (trying to fail).

```python
from sklearn.metrics import cohen_kappa_score
# Score: 0.8
print(f"Kappa: {cohen_kappa_score(y_test, y_pred)}")
```

---

## 4. The Log-Loss (Cross-Entropy)

Accuracy checks the **Label**. Log-Loss checks the **Confidence**.

*   Model A says "Dog" (51% confident). Right.
*   Model B says "Dog" (99% confident). Right.

**Accuracy** gives both 1 point.
**Log-Loss** gives Model B a much better score.
It punishes you heavily for being **Confidently Wrong**.

$$LogLoss = - \frac{1}{N} \sum (y \log(p) + (1-y) \log(1-p))$$

*   **Use when:** You need calibrated probabilities (e.g., betting, weather).
