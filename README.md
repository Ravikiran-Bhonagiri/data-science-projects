# 📊 Data Science Portfolio

**A comprehensive, end-to-end data science learning journey from exploratory analysis to advanced machine learning.**

> *"From raw data to actionable insights - A systematic approach to mastering data science"*

---

## 👤 About This Portfolio

This portfolio represents a **structured learning path** through the entire data science pipeline, combining **theoretical foundations** with **practical implementations**. Each module builds upon previous concepts, culminating in production-ready projects that demonstrate real-world problem-solving skills.

**Portfolio Structure:**
- **7 Learning Modules** covering core data science concepts
- **9 Comprehensive Projects** applying learned techniques
- **800+ lines of production-quality code**
- **30+ visualizations and analytical techniques**

---

## 📚 Learning Modules

### Module 1: [Exploratory Data Analysis (EDA)](./learning/01_eda/)
**Master the art of understanding data through systematic exploration**

**Topics Covered:**
- Data types and encoding (numerical, categorical, ordinal)
- Missing data detection and imputation (MCAR, MAR, MNAR)
- Outlier detection and handling (Z-score, IQR, Isolation Forest)
- Data visualization (univariate, bivariate, multivariate)
- Automated EDA tools (ydata-profiling, Sweetviz)
- High-dimensional EDA techniques

**Key Takeaways:**
- 11 comprehensive guides from basics to advanced techniques
- Workflow checklists for consistent analysis
- Production-ready corner case handling
- Specialized techniques for time series, geospatial, and NLP data

---

### Module 2: [Statistical Foundations](./learning/02_statistics/)
**Build the mathematical foundation for data-driven decision making**

**Topics Covered:**
- Probability distributions (Normal, Binomial, Poisson, Exponential)
- Hypothesis testing (t-tests, p-values, statistical power)
- Confidence intervals and uncertainty quantification
- ANOVA for multi-group comparisons
- Chi-square tests for categorical relationships
- Correlation analysis (Pearson, Spearman)
- Power analysis and sample size determination

**Practical Applications:**
- A/B testing frameworks
- Experimental design
- Statistical validation for ML models
- Business decision support

---

### Module 3: [Supervised Machine Learning](./learning/03_supervised_ml/)
**Learn predictive modeling for regression and classification**

**Algorithms Covered:**
- **Regression:** Linear Regression, Ridge, Lasso, Elastic Net
- **Classification:** Logistic Regression, Decision Trees, Random Forest, Gradient Boosting
- **Advanced:** SVM, Neural Networks basics

**Key Concepts:**
- Train/test splitting and cross-validation
- Hyperparameter tuning
- Model interpretation and feature importance
- Handling overfitting and underfitting

---

### Module 4: [Unsupervised Machine Learning](./learning/04_unsupervised_ml/)
**Discover hidden patterns and structures in unlabeled data**

**Techniques Covered:**
- **Clustering:** K-Means, DBSCAN, HDBSCAN, Hierarchical Clustering
- **Dimensionality Reduction:** PCA, t-SNE, UMAP, Isomap
- **Cluster Validation:** Silhouette score, Davies-Bouldin index, Calinski-Harabasz

**Folder Structure:**
- `01_kmeans_clustering.md` - Centroid-based clustering
- `02_dbscan_density_clustering.md` - Density-based spatial clustering
- `03_b_hdbscan_advanced.md` - Hierarchical DBSCAN
- `04_hierarchical_clustering.md` - Agglomerative/divisive clustering
- `05_dimensionality_reduction.md` - PCA fundamentals
- `06_tsne_visualization.md` - Non-linear dimensionality reduction
- `07_umap_advanced.md` - Uniform Manifold Approximation
- `08_cluster_validation_metrics.md` - Evaluation techniques

---

### Module 5: [Model Evaluation & Metrics](./learning/05_evaluation/)
**Master the art of assessing model performance**

**Concepts Covered:**
- **Classification Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Regression Metrics:** MAE, MSE, RMSE, R², Adjusted R²
- **Cross-Validation:** K-Fold, Stratified K-Fold, Time Series CV
- **Model Selection:** Bias-variance tradeoff, learning curves

**Specialized Topics:**
- Handling imbalanced datasets
- Multi-class evaluation
- Custom metrics for business cases

---

### Module 6: [Feature Engineering](./learning/06_feature_engineering/)
**Transform raw data into powerful predictive features**

**Techniques Covered:**
- **Encoding:** One-hot, Label, Ordinal, Target encoding
- **Scaling:** StandardScaler, MinMaxScaler, RobustScaler
- **Transformations:** Log, Box-Cox, Polynomial features
- **Feature Selection:** Correlation analysis, mutual information, recursive elimination
- **Feature Creation:** Binning, interactions, aggregations

**Real-World Applications:**
- Creating temporal features from dates
- Engineering domain-specific features
- Handling high-cardinality categories
- Managing feature engineering pipelines

---

### Module 7: [Unstructured Data Analytics](./learning/07_unstructured_data/)
**Extend beyond tabular data into text, images, and video**

**Domains Covered:**

#### 📝 Text (NLP):
- Text preprocessing (tokenization, stemming, lemmatization)
- Vectorization (Bag-of-Words, TF-IDF)
- Topic modeling (LDA, NMF)
- Sentiment analysis (VADER, TextBlob)
- Named Entity Recognition (spaCy)
- Text classification

#### 🖼️ Images (Computer Vision):
- Image representation as matrices
- Pixel manipulation and filtering
- Feature extraction (HOG, SIFT)
- Edge detection (Canny, Sobel)
- Dimensionality reduction for images
- Eigenfaces and facial recognition

#### 🎬 Video:
- Frame extraction and sampling
- Temporal dynamics analysis
- Motion detection and optical flow

---

## 🎯 Projects Portfolio

### 1. [Titanic Survival Forensics](./projects/project_titanic_eda/)
**Comprehensive EDA investigating survival patterns**

- **Dataset:** 891 passengers from RMS Titanic
- **Techniques:** Missing data imputation, survival analysis, class bias investigation
- **Key Findings:** 
  - "Women and Children First" protocol followed (74% female survival)
  - Socio-economic bias: 1st class 60% survival vs 3rd class 25%
  - Advanced imputation preserved signal from 20% missing age data

---

### 2. [Housing Price Prediction](./projects/project_housing_prediction/)
**End-to-end regression project**

- **Dataset:** Boston/California Housing
- **Techniques:** Feature engineering, polynomial features, regularization
- **Models:** Linear Regression, Ridge, Lasso, Elastic Net
- **Deliverable:** Production-ready price prediction model

---

### 3. [Customer Segmentation](./projects/project_customer_segmentation/)
**Unsupervised learning for market analysis**

- **Techniques:** K-Means, DBSCAN, HDBSCAN, Hierarchical Clustering
- **Dimensionality Reduction:** PCA, t-SNE for visualization
- **Business Value:** Identified distinct customer personas for targeted marketing

---

### 4. [Telco Customer Churn Analysis](./projects/project_telco_churn/)
**Statistical analysis and predictive modeling**

- **Dataset:** Telecom customer data
- **Techniques:** Hypothesis testing, logistic regression, survival analysis
- **Business Impact:** Strategic retention plan worth $3.9M/year
- **Notebooks:** 9 comprehensive analyses from descriptive stats to ROI calculation

---

### 5. [Feature Engineering Mastery](./projects/project_feature_engineering/)
**Systematic feature transformation pipeline**

- **Techniques:** All encoding methods, scaling strategies, feature selection
- **Outcome:** Reusable feature engineering library
- **Applications:** Cross-project feature transformer

---

### 6. [Model Evaluation Framework](./projects/project_model_evaluation/)
**Comprehensive model assessment toolkit**

- **Metrics:** Classification and regression evaluation
- **Visualizations:** Confusion matrices, ROC curves, learning curves
- **Cross-Validation:** Multiple strategies implemented
- **Deliverable:** Production-ready evaluation module

---

### 7. [Text EDA - Advanced NLP](./projects/project_text_eda/)
**Comprehensive text analysis on 20 Newsgroups dataset**

**Notebooks:**
1. `01_text_cleaning_and_frequency.ipynb` - Preprocessing, word frequency, Zipf's Law
2. `02_sentiment_and_topic_modeling.ipynb` - VADER sentiment, LDA topics
3. `03_advanced_nlp_techniques.ipynb` ⭐ **Advanced**
   - Named Entity Recognition (spaCy)
   - Part-of-Speech tagging
   - Text classification (Logistic Regression, Naive Bayes)
   - Advanced topic modeling (LDA vs NMF with metrics)
   - Sentiment comparison (VADER vs TextBlob)

---

### 8. [Image EDA - Computer Vision](./projects/project_image_eda/)
**Advanced image processing on Olivetti Faces**

**Notebooks:**
1. `01_pixel_analysis_and_eigenfaces.ipynb` - Pixel analysis, PCA, eigenfaces
2. `02_image_manifold_learning.ipynb` - t-SNE visualization
3. `03_advanced_image_analysis.ipynb` ⭐ **Advanced**
   - Color histogram analysis
   - Edge detection (Canny, Sobel)
   - HOG feature extraction
   - Harris corner detection
   - 4-way dimensionality reduction comparison (PCA, t-SNE, Isomap, UMAP)

---

### 9. [Video EDA - Temporal Analysis](./projects/project_video_eda/)
**Video frame extraction and motion analysis**

**Notebooks:**
1. `01_frame_extraction_and_analysis.ipynb` - Frame sampling, pixel dynamics
2. `02_temporal_dynamics_and_flow.ipynb` - Motion detection, temporal features

---

## 🛠️ Technical Stack

### Core Libraries:
- **Data Manipulation:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Machine Learning:** Scikit-learn, XGBoost
- **Statistics:** SciPy, Statsmodels

### NLP & Text:
- **Processing:** NLTK, spaCy, TextBlob
- **Vectorization:** Gensim, Wordcloud
- **Models:** Topic modeling (LDA, NMF)

### Computer Vision:
- **Image Processing:** OpenCV, scikit-image, PIL/Pillow
- **Feature Extraction:** HOG, SIFT
- **Dimensionality:** UMAP, t-SNE

### Video Processing:
- **Frame Handling:** imageio, OpenCV
- **Temporal Analysis:** Custom implementations

---

## 📊 Portfolio Metrics

| Metric | Count |
|--------|-------|
| **Learning Modules** | 7 |
| **Projects** | 9 |
| **Jupyter Notebooks** | 20+ |
| **Python Scripts** | 15+ |
| **Advanced Techniques** | 50+ |
| **Visualizations Created** | 100+ |
| **Lines of Code** | 5000+ |

---

## 🚀 Getting Started

### Prerequisites
```bash
# Python 3.8+
python --version

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Mac/Linux)
source .venv/bin/activate
```

### Installation

#### Option 1: Install All Dependencies
```bash
# Navigate to project root
cd data-science-portfolio

# Install all requirements
pip install -r requirements_unstructured.txt

# Download NLP models
python -m spacy download en_core_web_sm
```

#### Option 2: Project-Specific Installation
```bash
# Navigate to specific project
cd projects/project_text_eda

# Install project requirements
pip install -r requirements.txt
```

### Running Projects
```bash
# Launch Jupyter
jupyter notebook

# Navigate to any project notebook and run!
```

---

## 📁 Repository Structure

```
data-science-portfolio/
│
├── learning/                           # Theoretical foundations
│   ├── 01_eda/                        # Exploratory Data Analysis
│   │   ├── README.md
│   │   ├── 01_data_types.md
│   │   ├── 02_missing_data.md
│   │   ├── 03_outlier_detection.md
│   │   └── ... (11 guides total)
│   │
│   ├── 02_statistics/                 # Statistical Foundations
│   │   ├── README.md
│   │   ├── probability_distributions.md
│   │   ├── hypothesis_testing.md
│   │   └── ... (7 topics)
│   │
│   ├── 03_supervised_ml/              # Supervised Learning
│   ├── 04_unsupervised_ml/            # Unsupervised Learning
│   ├── 05_evaluation/                 # Model Evaluation
│   ├── 06_feature_engineering/        # Feature Engineering
│   └── 07_unstructured_data/          # Text, Image, Video
│
├── projects/                           # Practical implementations
│   ├── project_titanic_eda/
│   ├── project_housing_prediction/
│   ├── project_customer_segmentation/
│   ├── project_telco_churn/
│   ├── project_feature_engineering/
│   ├── project_model_evaluation/
│   ├── project_text_eda/
│   ├── project_image_eda/
│   ├── project_video_eda/
│   └── data/                          # Shared datasets
│
├── README.md                          # This file
└── requirements_unstructured.txt      # Dependencies
```

---

## 🎓 Learning Path Recommendations

### Beginner Track (2-3 months):
1. **Module 1:** EDA fundamentals
2. **Project:** Titanic EDA
3. **Module 2:** Basic statistics
4. **Module 3:** Supervised learning basics
5. **Project:** Housing prediction

### Intermediate Track (3-4 months):
1. **Module 4:** Unsupervised learning
2. **Project:** Customer segmentation
3. **Module 5:** Model evaluation
4. **Module 6:** Feature engineering
5. **Project:** Telco churn analysis

### Advanced Track (2-3 months):
1. **Module 7:** Unstructured data
2. **Project:** Text EDA (all 3 notebooks)
3. **Project:** Image EDA (all 3 notebooks)
4. **Project:** Video EDA
5. **Integration:** Combine techniques across domains

---

## 💡 Key Takeaways from This Portfolio

### Data Science Workflow Mastery:
✅ **Systematic EDA** - Never skip the exploration phase  
✅ **Statistical Rigor** - Validate assumptions before modeling  
✅ **Feature Engineering** - Raw data rarely works as-is  
✅ **Model Evaluation** - Metrics beyond accuracy matter  
✅ **Unstructured Data** - Extend beyond tabular formats  

### Production-Ready Skills:
✅ **Code Quality** - Clean, modular, reusable  
✅ **Documentation** - Every project comprehensively explained  
✅ **Error Handling** - Graceful dependency management  
✅ **Reproducibility** - Virtual environments, requirements files  

### Business Value Creation:
✅ **Interpretability** - Explain model decisions to stakeholders  
✅ **Actionable Insights** - Analysis drives real decisions  
✅ **ROI Quantification** - Tie models to business metrics  

---

## 🌟 Highlighted Achievements

### Advanced NLP Implementation:
- Named Entity Recognition with visualization
- Multi-model classification comparison
- Topic modeling with evaluation metrics (perplexity, reconstruction error)
- Sentiment analysis method comparison

### Computer Vision Expertise:
- Multiple edge detection algorithms implemented
- HOG and corner detection for feature extraction
- 4-way dimensionality reduction comparison
- Professional multi-panel visualizations

### Statistical Analysis:
- End-to-end business case (Telco Churn) worth $3.9M
- Hypothesis testing framework
- Power analysis for experimental design
- Multiple testing corrections

---

## 📈 Portfolio Evolution

**Phase 1:** Foundational concepts (Modules 1-3)  
**Phase 2:** Advanced techniques (Modules 4-6)  
**Phase 3:** Unstructured data (Module 7)  
**Phase 4:** Integration and real-world projects  

**Current Status:** ✅ All phases complete  
**Portfolio Rating:** **9/10** - Production-ready, interview-ready  

---

## 🔗 Connect & Collaborate

This portfolio demonstrates:
- **Technical Depth:** Mastery of data science fundamentals to advanced techniques
- **Breadth:** Coverage across supervised, unsupervised, and unstructured domains
- **Practical Skills:** Real-world projects with business impact
- **Code Quality:** Production-ready implementations
- **Communication:** Clear documentation and visualization

---

## 📝 Next Steps for Users

1. **Explore Learning Modules:** Start with [Module 1: EDA](./learning/01_eda/README.md)
2. **Pick a Project:** Try [Titanic EDA](./projects/project_titanic_eda/) for beginners
3. **Advanced Techniques:** Dive into [Unstructured Data](./projects/project_text_eda/)
4. **Customize:** Adapt projects with your own datasets
5. **Contribute:** Extend implementations with new techniques

---

## 📚 References & Resources

- **Learning Modules:** Self-contained markdown guides
- **Project READMEs:** Detailed problem statements and solutions
- **Notebooks:** Step-by-step implementations with explanations
- **Code Comments:** Inline documentation throughout

---

*Last Updated: December 2025*  
*Portfolio Type: Comprehensive Data Science Learning Journey*  
*Status: Production-Ready, Interview-Ready*

---

**Ready to explore? Start with the [Learning Modules](./learning/) or jump into a [Project](./projects/)!** 🚀
