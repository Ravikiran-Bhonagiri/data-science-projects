<div align="center">

# 📝 Text EDA - Advanced NLP Pipeline

### *From Raw Text to Actionable Insights*

![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![Type](https://img.shields.io/badge/Type-NLP-purple?style=flat-square)
![Notebooks](https://img.shields.io/badge/Notebooks-3-blue?style=flat-square)
![Level](https://img.shields.io/badge/Level-Advanced-red?style=flat-square)

**Comprehensive natural language processing on 20 Newsgroups dataset**

[📊 Dataset](#-dataset) • [📚 Notebooks](#-notebooks) • [🎯 Techniques](#-techniques-covered) • [🚀 Run It](#-quick-start)

</div>

---

## 📊 Dataset

**20 Newsgroups** - Classic NLP benchmark dataset

| Attribute | Value |
|-----------|-------|
| **Documents** | ~18,000 posts |
| **Categories** | 20 topics |
| **Source** | Usenet newsgroups |
| **Type** | Multiclass text classification |

**Topics include:** Politics, sports, religion, computers, science

---

## 📚 Notebooks

<table>
<tr>
<td width="33%">

### 📄 Notebook 1
**Text Cleaning & Frequency**

**Techniques:**
- ✅ Tokenization
- ✅ Stopword removal
- ✅ Lemmatization
- ✅ Word frequency analysis
- ✅ Zipf's Law validation
- ✅ N-gram extraction
- ✅ WordCloud visualization

**Output:** Clean, analyzed text corpus

</td>
<td width="33%">

### 📊 Notebook 2
**Sentiment & Topic Modeling**

**Techniques:**
- ✅ VADER sentiment analysis
- ✅ TF-IDF vectorization
- ✅ Latent Dirichlet Allocation (LDA)
- ✅ Topic visualization
- ✅ Document clustering
- ✅ t-SNE dimensionality reduction

**Output:** Topics discovered, sentiment scored

</td>
<td width="33%">

### ⭐ Notebook 3
**Advanced NLP**

**Techniques:**
- ✅ Named Entity Recognition (spaCy)
- ✅ Part-of-Speech tagging
- ✅ Text classification (2 models)
- ✅ LDA vs NMF comparison
- ✅ VADER vs TextBlob sentiment
- ✅ Topic coherence metrics

**Output:** Production-ready NLP pipeline

</td>
</tr>
</table>

---

## 🎯 Techniques Covered

<details>
<summary><strong>🔤 Text Preprocessing</strong></summary>

- **Tokenization:** Split text into words
- **Stopword Removal:** Filter common words (the, a, is)
- **Lemmatization:** Convert to base form (running → run)
- **Cleaning:** Remove special characters, numbers
- **Normalization:** Lowercase, whitespace handling

</details>

<details>
<summary><strong>📊 Statistical Analysis</strong></summary>

- **Word Frequency:** Most common terms
- **Zipf's Law:** Verify power-law distribution
- **N-grams:** Bigrams, trigrams for phrases
- **Document Length:** Distribution analysis
- **Vocabulary Size:** Unique word count

</details>

<details>
<summary><strong>🔢 Vectorization</strong></summary>

- **Bag-of-Words:** Simple word counting
- **TF-IDF:** Term frequency × inverse document frequency
- **Word Embeddings:** Semantic representations
- **Sparse Matrices:** Efficient storage

</details>

<details>
<summary><strong>🎨 Topic Modeling</strong></summary>

- **LDA (Latent Dirichlet Allocation):** Probabilistic topics
- **NMF (Non-negative Matrix Factorization):** Linear algebra approach
- **Topic Coherence:** Quality metrics (perplexity, C_v score)
- **Topic Visualization:** Word clouds per topic

</details>

<details>
<summary><strong>😊 Sentiment Analysis</strong></summary>

- **VADER:** Rule-based, social media optimized
- **TextBlob:** Pattern-based sentiment
- **Polarity Scores:** Positive/negative/neutral
- **Comparative Analysis:** Method comparison

</details>

<details>
<summary><strong>🏷️ Named Entity Recognition</strong></summary>

- **Entities Detected:** PERSON, ORG, GPE, DATE
- **spaCy Pipeline:** Industrial-strength NLP
- **Visualization:** Entity highlighting
- **Extraction:** Structured data from text

</details>

<details>
<summary><strong>🎯 Text Classification</strong></summary>

- **Logistic Regression:** Linear classifier
- **Naive Bayes:** Probabilistic classifier
- **Evaluation:** Precision, recall, F1-score
- **Confusion Matrix:** Error analysis

</details>

---

## 🚀 Quick Start

### Installation

```bash
# Navigate to project
cd projects/project_text_eda

# Install dependencies
pip install -r requirements.txt

# Download NLP models
python -m spacy download en_core_web_sm
python -m textblob.download_corpora
```

### Run Notebooks

```bash
# Launch Jupyter
jupyter notebook notebooks/

# Execute in order:
# 1. 01_text_cleaning_and_frequency.ipynb
# 2. 02_sentiment_and_topic_modeling.ipynb
# 3. 03_advanced_nlp_techniques.ipynb
```

---

## 💡 Key Learnings

**What You'll Master:**

<table>
<tr>
<td width="50%">

### 📊 Core NLP Skills
- ✅ Text preprocessing pipeline
- ✅ TF-IDF vectorization
- ✅ Topic modeling (LDA, NMF)
- ✅ Sentiment analysis (2 methods)
- ✅ N-gram analysis
- ✅ Zipf's Law validation

</td>
<td width="50%">

### ⭐ Advanced Techniques
- ✅ Named Entity Recognition
- ✅ Part-of-Speech tagging
- ✅ Multi-model comparison
- ✅ Topic coherence metrics
- ✅ Text classification
- ✅ Production pipelines

</td>
</tr>
</table>

---

## 🛠️ Libraries Used

| Library | Purpose |
|---------|---------|
| **NLTK** | Tokenization, stopwords, lemmatization |
| **spaCy** | NER, POS tagging, pipeline |
| **TextBlob** | Sentiment analysis |
| **scikit-learn** | TF-IDF, classification, LDA |
| **Gensim** | Topic modeling |
| **WordCloud** | Text visualization |

---

## 📈 Sample Outputs

**Topics Discovered (LDA):**
```
Topic 1: [politics, government, president, election, policy]
Topic 2: [computer, software, program, system, windows]
Topic 3: [space, nasa, moon, earth, science]
Topic 4: [religion, god, church, christian, bible]
Topic 5: [baseball, game, team, player, season]
```

**Sentiment Distribution:**
- Positive: 42%
- Neutral: 38%
- Negative: 20%

---

<div align="center">

**Master Text Analysis with NLP** 📝

*3 notebooks • 7 advanced techniques • Production-ready pipeline*

[⬅️ Telco Churn](../project_telco_churn/) • [🏠 Home](../../README.md) • [➡️ Image EDA](../project_image_eda/)

</div>
