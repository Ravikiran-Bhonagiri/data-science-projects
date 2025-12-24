<div align="center">

# 🖼️ Image EDA - Computer Vision Analysis

### *From Pixels to Patterns*

![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![Type](https://img.shields.io/badge/Type-Computer_Vision-teal?style=flat-square)
![Notebooks](https://img.shields.io/badge/Notebooks-3-blue?style=flat-square)
![Level](https://img.shields.io/badge/Level-Advanced-red?style=flat-square)

**Advanced image processing and analysis on Olivetti Faces dataset**

[📊 Dataset](#-dataset) • [📚 Notebooks](#-notebooks) • [🎯 Techniques](#-techniques-covered) • [🚀 Run It](#-quick-start)

</div>

---

## 📊 Dataset

**Olivetti Faces** - Classic computer vision benchmark

| Attribute | Value |
|-----------|-------|
| **Images** | 400 face images |
| **Subjects** | 40 people (10 images each) |
| **Resolution** | 64×64 pixels (grayscale) |
| **Type** | Facial recognition |

**Perfect for:** PCA, eigenfaces, manifold learning

---

## 📚 Notebooks

<table>
<tr>
<td width="33%">

### 🎨 Notebook 1
**Pixel Analysis & Eigenfaces**

**Techniques:**
- ✅ Images as numerical matrices
- ✅ Pixel intensity distributions
- ✅ Average face computation
- ✅ PCA dimensionality reduction
- ✅ Eigenfaces extraction
- ✅ Face reconstruction

**Output:** Top eigenfaces, reconstructed images

</td>
<td width="33%">

### 📊 Notebook 2
**Image Manifold Learning**

**Techniques:**
- ✅ PCA for 2D projection
- ✅ t-SNE visualization
- ✅ Person clustering
- ✅ Pattern discovery
- ✅ Similarity analysis

**Output:** 2D embeddings showing face similarity

</td>
<td width="33%">

### ⭐ Notebook 3
**Advanced CV**

**Techniques:**
- ✅ Color histogram analysis
- ✅ Edge detection (Canny, Sobel)
- ✅ HOG feature extraction
- ✅ Harris corner detection
- ✅ 4-way dimensionality comparison
  (PCA, t-SNE, Isomap, UMAP)

**Output:** Advanced feature pipeline

</td>
</tr>
</table>

---

## 🎯 Techniques Covered

<details>
<summary><strong>🎨 Image Fundamentals</strong></summary>

- **Matrix Representation:** Understand images as 2D arrays
- **Pixel Values:** Intensity (0-255) for grayscale
- **Shape:** (height, width) dimensions
- **Normalization:** Scale to [0, 1] range
- **Flattening:** Convert 2D to 1D for ML

</details>

<details>
<summary><strong>⚡ Eigenfaces (PCA on Images)</strong></summary>

- **Average Face:** Mean of all images
- **Covariance Matrix:** Pixel-wise variance
- **Principal Components:** Top eigenvectors
- **Eigenfaces:** "Ghost faces" capturing variance
- **Reconstruction:** Build faces from components
- **Dimensionality:** 4096 pixels → 50 components

</details>

<details>
<summary><strong>🔍 Edge Detection</strong></summary>

- **Canny Edge Detector:** Multi-stage algorithm
- **Sobel Operator:** Gradient-based edges
- **Comparison:** Different edge strengths
- **Applications:** Object boundaries, feature extraction

</details>

<details>
<summary><strong>📐 Feature Extraction</strong></summary>

- **HOG (Histogram of Oriented Gradients):** Shape descriptors
- **Harris Corners:** Interest point detection
- **Color Histograms:** Distribution analysis
- **SIFT-like Features:** Scale-invariant descriptors

</details>

<details>
<summary><strong>🗜️ Dimensionality Reduction</strong></summary>

**4-Way Comparison:**
- **PCA:** Linear, preserves variance (fastest)
- **t-SNE:** Non-linear, preserves local structure (best visualization)
- **Isomap:** Geodesic distances on manifold
- **UMAP:** Uniform manifold approximation (balanced speed/quality)

</details>

---

## 🚀 Quick Start

### Installation

```bash
# Navigate to project
cd projects/project_image_eda

# Install dependencies
pip install -r requirements.txt
# Includes: opencv-python, scikit-image, umap-learn
```

### Run Notebooks

```bash
# Launch Jupyter
jupyter notebook notebooks/

# Execute in order:
# 1. 01_pixel_analysis_and_eigenfaces.ipynb
# 2. 02_image_manifold_learning.ipynb
# 3. 03_advanced_image_analysis.ipynb
```

---

## 💡 Key Learnings

**What You'll Master:**

<table>
<tr>
<td width="50%">

### 🎨 Core CV Skills
- ✅ Image as matrix manipulation
- ✅ Pixel intensity analysis
- ✅ PCA eigenfaces
- ✅ Face reconstruction
- ✅ Similarity visualization
- ✅ Manifold learning

</td>
<td width="50%">

### ⭐ Advanced Techniques
- ✅ Edge detection (2 methods)
- ✅ HOG feature extraction
- ✅ Corner detection
- ✅ 4-way dimension reduction
- ✅ Color histogram analysis
- ✅ Multi-panel visualizations

</td>
</tr>
</table>

---

## 🛠️ Libraries Used

| Library | Purpose |
|---------|---------|
| **NumPy** | Array operations, matrix math |
| **Matplotlib** | Visualization, image display |
| **scikit-learn** | PCA, datasets, metrics |
| **OpenCV** | Edge detection, features |
| **scikit-image** | HOG, color analysis |
| **UMAP** | Advanced dimensionality reduction |

---

## 📈 Sample Visualizations

**Eigenfaces (Top 5):**
```
Eigen 1: General face shape (explains 12% variance)
Eigen 2: Lighting variation (explains 8% variance)
Eigen 3: Face orientation (explains 6% variance)
Eigen 4: Facial expression (explains 5% variance)
Eigen 5: Hair patterns (explains 4% variance)
```

**Dimensionality Reduction Comparison:**
- PCA: Clear global structure, linear separations
- t-SNE: Best local clustering, person groups visible
- Isomap: Manifold structure preserved
- UMAP: Balanced, faster than t-SNE

---

## 🎯 Real-World Applications

**Computer Vision techniques demonstrated:**

| Application | Technique Used |
|-------------|----------------|
| **Face Recognition** | Eigenfaces, PCA |
| **Object Detection** | Edge detection, corners |
| **Image Search** | Feature extraction, similarity |
| **Biometrics** | Facial feature analysis |
| **Medical Imaging** | Pattern recognition |

---

<div align="center">

**Master Computer Vision Fundamentals** 🖼️

*3 notebooks • 10+ CV techniques • From basics to advanced*

[⬅️ Text EDA](../project_text_eda/) • [🏠 Home](../../README.md) • [➡️ Video EDA](../project_video_eda/)

</div>
