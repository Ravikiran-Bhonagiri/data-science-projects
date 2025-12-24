# Text/NLP EDA

Text data requires completely different analysis than numbers. This guide covers techniques for exploring unstructured text before building NLP models.

---

## When to Use This
- Sentiment analysis (reviews, tweets)
- Chatbot development
- Document classification
- Topic modeling
- Spam detection

---

## 1. Basic Text Statistics

```python
import pandas as pd
import numpy as np

# Sample text data
df = pd.read_csv('customer_reviews.csv')

# Word count
df['word_count'] = df['review_text'].str.split().str.len()

# Character count
df['char_count'] = df['review_text'].str.len()

# Average word length
df['avg_word_len'] = df['char_count'] / df['word_count']

# Sentence count
df['sentence_count'] = df['review_text'].str.count(r'[.!?]+')

# Summary statistics
print(df[['word_count', 'char_count', 'avg_word_len']].describe())
```

---

## 2. Word Frequency Analysis

```python
from collections import Counter
import matplotlib.pyplot as plt

# Combine all text
all_text = ' '.join(df['review_text'].values).lower()
words = all_text.split()

# Count words
word_freq = Counter(words)
top_20 = word_freq.most_common(20)

# Plot
words, counts = zip(*top_20)
plt.figure(figsize=(12, 6))
plt.barh(words, counts)
plt.xlabel('Frequency')
plt.title('Top 20 Most Common Words')
plt.gca().invert_yaxis()
plt.show()
```

**Insight:** "the", "a", "and" dominate → Need to remove stopwords!

---

## 3. Stopword Removal & Cleaning

```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Download stopwords (run once)
import nltk
nltk.download('stopwords')
nltk.download('punkt')

def clean_text(text):
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    
    return tokens

# Apply
df['cleaned_tokens'] = df['review_text'].apply(clean_text)

# Recalculate frequency on cleaned text
all_tokens = [token for tokens in df['cleaned_tokens'] for token in tokens]
word_freq_clean = Counter(all_tokens)
print(word_freq_clean.most_common(20))
```

---

## 4. Word Clouds

```python
from wordcloud import WordCloud

# Generate word cloud from cleaned tokens
text_for_cloud = ' '.join(all_tokens)

wordcloud = WordCloud(width=800, height=400, 
                     background_color='white',
                     colormap='viridis').generate(text_for_cloud)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Customer Reviews')
plt.show()
```

---

## 5. N-Gram Analysis

**N-grams:** Sequences of N words (Bigrams = 2 words, Trigrams = 3 words)

```python
from nltk import bigrams, trigrams

# Bigrams (2-word phrases)
all_bigrams = []
for tokens in df['cleaned_tokens']:
    all_bigrams.extend(list(bigrams(tokens)))

bigram_freq = Counter(all_bigrams)
print("Top 10 Bigrams:")
for bg, count in bigram_freq.most_common(10):
    print(f"{' '.join(bg)}: {count}")

# Trigrams (3-word phrases)
all_trigrams = []
for tokens in df['cleaned_tokens']:
    all_trigrams.extend(list(trigrams(tokens)))

trigram_freq = Counter(all_trigrams)
print("\nTop 10 Trigrams:")
for tg, count in trigram_freq.most_common(10):
    print(f"{' '.join(tg)}: {count}")
```

**Insight:** Bigrams reveal phrases like "customer service", "bad quality"

---

## 6. Vocabulary Size & Diversity

```python
# Unique words
unique_words = set(all_tokens)
vocab_size = len(unique_words)
total_words = len(all_tokens)

print(f"Vocabulary Size: {vocab_size}")
print(f"Total Words: {total_words}")

# Lexical Diversity (Type-Token Ratio)
diversity = vocab_size / total_words
print(f"Lexical Diversity: {diversity:.4f}")
```

**What it means:**
- **Low diversity (< 0.3):** Repetitive text (tweets, short reviews)
- **High diversity (> 0.7):** Rich vocabulary (academic papers, novels)

---

## 4. Stemming vs Lemmatization

**Stemming:** Removing suffixes (fast but crude). "Studies" → "Studi"
**Lemmatization:** Reducing word to dictionary root (slow but accurate). "Studies" → "Study"

```python
from nltk.stem import WordNetLemmatizer, PorterStemmer
nltk.download('wordnet')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

word = "better"
print(f"Stem: {stemmer.stem(word)}")     # "better"
print(f"Lemma: {lemmatizer.lemmatize(word, pos='a')}") # "good"
```

---

## 5. POS Tagging & Dependency Parsing (SpaCy)

POS (Part of Speech) tagging helps filter for only **Adjectives** (opinion) or **Nouns** (subject).

```python
import spacy
nlp = spacy.load("en_core_web_sm")

doc = nlp("The customer service was absolutely fantastic but the battery life is poor.")

for token in doc:
    print(f"{token.text:12} {token.pos_:10} {token.dep_}")

# Filter for only Adjectives
adjectives = [token.text for token in doc if token.pos_ == "ADJ"]
print(f"Opinions found: {adjectives}")
```

---

## 6. Advanced Topic Modeling Visualization (pyLDAvis)

Standard LDA prints words; **pyLDAvis** shows how topics relate to each other in a 2D space.

```python
import pyLDAvis
import pyLDAvis.sklearn

# Fit LDA model first (as shown in section 9 of original doc)
# lda = LatentDirichletAllocation(...)
# lda.fit(doc_term_matrix)

# Prepare visualization
panel = pyLDAvis.sklearn.prepare(lda, doc_term_matrix, vectorizer, mds='tsne')
pyLDAvis.save_html(panel, 'lda_visualization.html')
```

---

## 7. Word Embeddings (Word2Vec)
Count-based methods (TF-IDF) treat "King" and "Queen" as separate. **Embeddings** understand they are related.

```python
from gensim.models import Word2Vec

# Train model on our text
model = Word2Vec(sentences=df['cleaned_tokens'], vector_size=100, window=5, min_count=1)

# Find similar words
similar_to_service = model.wv.most_similar('service')
print(f"Words related to 'service': {similar_to_service}")

# Vector Math: King - Man + Woman = Queen
# result = model.wv.most_similar(positive=['woman', 'king'], negative=['man'])
```

---

## 8. Identifying Text Bias & Offensive Language

```python
# Simple offensive word check
from better_profanity import profanity

df['is_profane'] = df['review_text'].apply(lambda x: profanity.contains_profanity(x))
print(f"Percentage of toxic content: {df['is_profane'].mean():.1%}")
```

**Common entities:**
- PERSON, ORG (Organization), GPE (Geopolitical Entity), MONEY, DATE

---

## Checklist for Text/NLP EDA

- [ ] Calculate basic stats (word count, char count, sentence count)
- [ ] Clean text (lowercase, remove punctuation, stopwords)
- [ ] Analyze word frequency (after cleaning)
- [ ] Generate word cloud
- [ ] Identify common N-grams (2-3 word phrases)
- [ ] Check vocabulary size and diversity
- [ ] Detect sentiment distribution
- [ ] Identify language(s)
- [ ] Preview topics (LDA)
- [ ] Extract named entities (if relevant)
