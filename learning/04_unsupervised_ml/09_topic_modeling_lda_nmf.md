Imagine you walk into a library with **1 million books**, but the librarian is gone.
There are no labels. No "Fiction" section. No "History" section. Just a mountain of paper.
You can't read 1 million books.

**# Topic Modeling: Organizing 1M Support Tickets

**Zendesk's Auto-Routing Problem:**

Customer support receives 1M unstructured tickets/month.

**Manual categorization:**
- Requires 50 human agents to read and tag each ticket
- Average time: 2 minutes/ticket
- Cost: $400k/month in labor
- Backlogs: 48-hour response times

**LDA Topic Modeling solution:**
```python
from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_components=20, random_state=42)
topics = lda.fit_transform(ticket_tfidf_matrix)
```

**Discovered 20 topics automatically:**
1. **Topic 3:** "password reset login account access" → Routing: Tier 1 support
2. **Topic 7:** "billing invoice charge refund payment" → Routing: Finance team
3. **Topic 12:** "bug error crash feature not working" → Routing: Engineering escalation
4. **Topic 18:** "cancel subscription downgrade plan" → Routing: Retention specialist

**Results:**
- Auto-routes 87% of tickets correctly
- Human agents only handle 13% edge cases
- Labor cost: $400k → $52k/month
- Response time: 48 hours → 4 hours
- **Annual savings:** $4.2M

**Why it works:** No labeled data needed, discovers hidden structure in text

---

## Production Use Cases

**News aggregation (Google News):**
- 100k articles/day from 50k sources
- LDA finds 50 topics: Politics, Sports, Tech, etc.
- Automatically clusters related stories

**Academic research (PubMed):**
- 30M biomedical papers
- Topic modeling reveals emerging research trends
- Helps scientists find related work
    *   **Action:** Ignore Topic 1. **Fix Topic 2 immediately.**

**2. The Legal Discovery Specialist:**
A company is being sued. You have emails from 2005 to 2015.
*   **The Challenge:** Find the "smoking gun" evidence of fraud without reading 5 million emails.
*   **The Strategy:** Filter for topics containing "money, offshore, delete, secret".

---

## 1. Latent Dirichlet Allocation (LDA)

The "Bayesian" approach.
**Assumption:** Every document is a mixture of topics. Every topic is a mixture of words.

1.  **Input:** A "Bag of Words" matrix (CountVectorizer).
2.  **Output:**
    *   Document-Topic Matrix ("Doc A is 50% Topic 1").
    *   Topic-Word Matrix ("Topic 1 is 10% 'football'").

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Sample Data
documents = [
    "The quarterback threw a touchdown pass",
    "The stock market crashed today",
    "The government passed a new law",
    "The player scored a goal in the final minute",
    "Invest in index funds for retirement"
]

# 1. Vectorize (Turn text to numbers)
# stop_words='english' removes "the", "a", "is"
tf_vectorizer = CountVectorizer(stop_words='english')
dtm = tf_vectorizer.fit_transform(documents)

# 2. Run LDA
# n_components = Number of Topics (You must guess this, like K in K-Means)
lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda.fit(dtm)

# 3. View Topics
feature_names = tf_vectorizer.get_feature_names_out()

for topic_idx, topic in enumerate(lda.components_):
    print(f"Topic {topic_idx}:")
    # Get top 3 words for this topic
    top_words_idx = topic.argsort()[:-4:-1]
    print([feature_names[i] for i in top_words_idx])

# Output might look like:
# Topic 0: ['pass', 'touchdown', 'scored'] -> (Sports)
# Topic 1: ['market', 'stock', 'invest'] -> (Finance)
```

---

## 2. Non-Negative Matrix Factorization (NMF)

The "Linear Algebra" approach.
Often **faster** and produces **more distinctive** topics than LDA for short text (like Tweets).

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# NMF works better with TF-IDF (weighted counts)
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(documents)

nmf = NMF(n_components=2, random_state=42, init='nndsvd')
nmf.fit(tfidf)

# (Code to print topics is same as LDA)
```

---

## 3. Visualization: pyLDAvis

The absolute best way to explain Topic Models to non-technical stakeholders.
It creates an interactive bubble chart.

```python
import pyLDAvis
import pyLDAvis.sklearn

# Prepare the dashboard
pyLDAvis.enable_notebook()
panel = pyLDAvis.sklearn.prepare(lda, dtm, tf_vectorizer, mds='tsne')

# Save to HTML
pyLDAvis.save_html(panel, 'topic_modeling.html')
```

### 🚧 Limitations
1.  **Bag of Words:** It ignores word order. "Shark eats Man" and "Man eats Shark" are identical to LDA.
2.  **Choosing K:** Just like K-Means, finding the "right" number of topics is an art (use Coherence Score to tune).
3.  **Naming:** The model gives you lists of words. **YOU** must come up with the name "Sports" or "Finance".
