# Module 7: Unstructured Data Analytics

**Status:** 🚧 In Progress

This module expands the data science toolkit beyond structured tabular data (rows and columns) into the messy, rich world of unstructured data: Text, Images, and Video.

## Learning Objectives
1.  **Text (NLP):** moving from "Bag of Words" to Topic Modeling. Understanding cleaning pipelines (Stopwords, Lemmatization) and vectorization (TF-IDF).
2.  **Image (Computer Vision):** treating images as numerical matrices. Understanding pixel distributions, filtering, and dimensionality reduction (Eigenfaces).
3.  **Video:** interacting with time-series of images. Feature extraction from video frames and analyzing temporal dynamics (Motion).

## Projects
| Project | Type | Dataset | Goal |
| :--- | :--- | :--- | :--- |
| **01. Text EDA** | NLP | 20 Newsgroups | Topic Modeling & Sentiment Analysis |
| **02. Image EDA** | CV | LFW Faces | Eigenfaces & Pixel Manifolds |
| **03. Video EDA** | Video | UCF101 (Sample) | Motion Analysis & Frame Statistics |

## Key Libraries
- **NLP:** `nltk`, `sklearn.feature_extraction.text`, `wordcloud`
- **CV:** `opencv-python`, `scikit-image`, `matplotlib`
- **Video:** `moviepy`, `cv2`
