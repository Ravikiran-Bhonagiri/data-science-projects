# Naive Bayes: 60 Years Old, Still Winning

**The Gmail Spam Filter (2004-Present):**

Google receives 100B+ emails daily. Need to classify spam in <1ms per email.

**Early neural network attempt (2003):**
- Training time: 3 days on full cluster
- Inference: 45ms per email (too slow)
- Accuracy: 96.2%

**Naive Bayes solution:**
```python
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB(alpha=1.0)
model.fit(word_counts, labels)
# Training: 12 minutes
# Inference: 0.3ms per email
# Accuracy: 95.8%
```

**Why Naive Bayes won:**
- 0.4% accuracy loss acceptable
- 150× faster inference
- Retrains daily on new spam patterns
- Probabilistic output allows threshold tuning

**20 years later:** Gmail still uses Naive Bayes as first-pass filter (NN for edge cases)

---

## Real-World Performance

**Sentiment analysis (Twitter):**
- 10,000 training tweets
- Linear SVM: 82% accuracy, requires full feature engineering
- Naive Bayes: 79% accuracy, works on raw bag-of-words
- **Winner:** NB (good enough, 10× faster to deploy)

**Medical diagnosis (small data):**
- 200 patient records, 15 symptoms
- Random Forest: Overfits
- Naive Bayes: 73% accuracy (stable)

---

## 1. Bayes' Theorem
$$P(A|B) = \frac{P(B|A) \times P(A)}{P(B)}$$

- **Posterior:** Probability of class given features.
- **Likelihood:** Probability of features given class.
- **Prior:** General probability of class.

---

## 2. The "Naive" Assumption
It assumes features are **independent** (e.g., the presence of "Winning" implies spam regardless of "Ticket").
- This is almost never true in real life!
- But the model still works remarkably well because it gets the *maximum* probability right, even if the exact probability is wrong.

---

## 3. Types of Naive Bayes

### A. Gaussian NB
- Use when features are **Continuous** and Normal (Bell curve).
- Example: Iris dataset (sepal length).

### B. Multinomial NB
- Use when features are **Discrete Counts**.
- Example: Text classification (Word counts in email).

### C. Bernoulli NB
- Use when features are **Binary** (0/1).
- Example: Text classification (Word presence - True/False).

---

## 4. Implementation (Text Classification)

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline

# Sample text data
emails = [
    "Win a free lottery ticket now",
    "Meeting agenda for tomorrow",
    "Free money limited time offer",
    "Project deadline extended"
]
labels = [1, 0, 1, 0] # 1=Spam

# Create Pipeline: Text -> Counts -> Model
model = make_pipeline(
    CountVectorizer(), 
    MultinomialNB(alpha=1.0) # Alpha = Laplace Smoothing
)

model.fit(emails, labels)

# Predict
print(model.predict(["Get free ticket"]))
```

---

## 5. Laplace Smoothing (Alpha)
What if a word in the test set never appeared in training?
- Probability becomes 0.
- Since we multiply probabilities, the whole prediction becomes 0.
- **Laplace Smoothing** adds "1" to every count so no probability is ever truly zero.

---

## 6. Pros & Cons
**Pros:**
- Extremely fast (training and prediction).
- Handles huge vocabulary sizes well.
- Needs less training data.

**Cons:**
- Zero frequency problem (solved by smoothing).
- "Naive" assumption limits complex pattern learning.
