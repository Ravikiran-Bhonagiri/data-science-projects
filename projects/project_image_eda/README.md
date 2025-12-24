# Project: Image Exploratory Data Analysis

**Dataset:** [Olivetti Faces](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html) (via Scikit-Learn)
**Type:** Computer Vision (CV)

## Overview
This project deals with unstructured image data. Images are typically represented as high-dimensional numerical matrices. We will explore the "Olivetti Faces" dataset (40 distinct subjects, 10 images each) to demonstrate facial feature extraction.

## Goal
To implement a Computer Vision EDA pipeline:
1.  **Pixel Statistics:** Analyzing brightness, contrast, and channel distributions.
2.  **Average Face:** Computing the "mean face" to see common features.
3.  **Eigenfaces (PCA):** Using Principal Component Analysis to find the fundamental building blocks of human faces.
4.  **Manifold Learning:** Visualization of the high-dimensional face space in 2D using t-SNE.

## Notebooks
1.  `01_pixel_analysis_and_eigenfaces.ipynb`: Basic statistics and dimensionality reduction (Eigenfaces).
2.  `02_image_manifold_learning.ipynb`: Advanced visualization of class separability.
