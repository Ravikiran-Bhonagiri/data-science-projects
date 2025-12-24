<div align="center">

# 🎥 Video EDA - Temporal Analysis

### *Analyzing Motion Through Time*

![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![Type](https://img.shields.io/badge/Type-Video_Analysis-orange?style=flat-square)
![Notebooks](https://img.shields.io/badge/Notebooks-2-blue?style=flat-square)
![Level](https://img.shields.io/badge/Level-Advanced-red?style=flat-square)

**Frame extraction, temporal dynamics, and motion analysis on UCF101 sample**

[📊 Dataset](#-dataset) • [📚 Notebooks](#-notebooks) • [🎯 Techniques](#-techniques-covered) • [🚀 Run It](#-quick-start)

</div>

---

## 📊 Dataset

**UCF101 Sample** - Action recognition video clips

| Attribute | Value |
|-----------|-------|
| **Source** | UCF101 dataset (sample) |
| **Videos** | Short action clips |
| **Type** | Human activity recognition |
| **Format** | MP4 video files |

**Activities:** Sports, daily actions, human motion

---

## 📚 Notebooks

<table>
<tr>
<td width="50%">

### 📹 Notebook 1
**Frame Extraction & Analysis**

**Techniques:**
- ✅ Video loading (imageio)
- ✅ Frame extraction
- ✅ Sampling strategies (uniform, keyframe)
- ✅ Frame-level statistics
- ✅ Pixel intensity over time
- ✅ Basic motion detection
- ✅ Multi-frame visualization

**Output:** Extracted frames, temporal plots

</td>
<td width="50%">

### ⚡ Notebook 2
**Temporal Dynamics & Flow**

**Techniques:**
- ✅ Motion detection
- ✅ Frame differencing
- ✅ Temporal features
- ✅ Optical flow concepts
- ✅ Activity pattern analysis
- ✅ Scene change detection
- ✅ Time-series visualization

**Output:** Motion patterns, flow analysis

</td>
</tr>
</table>

---

## 🎯 Techniques Covered

<details>
<summary><strong>🎬 Video Fundamentals</strong></summary>

- **Video as Frame Sequence:** Images over time
- **Frame Rate (FPS):** Temporal resolution
- **Resolution:** Spatial dimensions (height × width)
- **Channels:** RGB or grayscale
- **Duration:** Total frames / FPS

</details>

<details>
<summary><strong>📹 Frame Extraction</strong></summary>

- **Uniform Sampling:** Every Nth frame
- **Keyframe Detection:** Significant frames only
- **Adaptive Sampling:** Based on motion
- **Storage:** Save as image sequences
- **Metadata:** Timestamp, frame number

</details>

<details>
<summary><strong>⚡ Motion Detection</strong></summary>

- **Frame Differencing:** Subtract consecutive frames
- **Threshold Analysis:** Significant changes only
- **Motion Magnitude:** Pixel change intensity
- **Motion Regions:** Where movement occurs
- **Activity Detection:** Movement vs stillness

</details>

<details>
<summary><strong>📊 Temporal Features</strong></summary>

- **Pixel Statistics Over Time:** Mean, variance
- **Color Distribution:** Temporal color changes
- **Scene Changes:** Detect cuts/transitions
- **Action Patterns:** Repetitive motions
- **Activity Duration:** How long actions last

</details>

<details>
<summary><strong>🌊 Optical Flow (Concepts)</strong></summary>

- **Motion Vectors:** Direction and speed
- **Dense vs Sparse:** All pixels vs key points
- **Applications:** Tracking, activity recognition
- **Challenges:** Lighting changes, occlusion

</details>

---

## 🚀 Quick Start

### Installation

```bash
# Navigate to project
cd projects/project_video_eda

# Install dependencies
pip install -r requirements.txt
# Includes: imageio, opencv-python, numpy, matplotlib
```

### Run Notebooks

```bash
# Launch Jupyter
jupyter notebook notebooks/

# Execute in order:
# 1. 01_frame_extraction_and_analysis.ipynb
# 2. 02_temporal_dynamics_and_flow.ipynb
```

**Note:** Sample videos included. For full UCF101, download from [official source](https://www.crcv.ucf.edu/data/UCF101.php).

---

## 💡 Key Learnings

**What You'll Master:**

<table>
<tr>
<td width="50%">

### 🎬 Core Video Skills
- ✅ Video I/O operations
- ✅ Frame extraction methods
- ✅ Temporal sampling strategies
- ✅ Frame-level statistics
- ✅ Motion quantification
- ✅ Time-series visualization

</td>
<td width="50%">

### ⚡ Advanced Analysis
- ✅ Frame differencing
- ✅ Motion magnitude calculation
- ✅ Scene change detection
- ✅ Activity pattern recognition
- ✅ Optical flow concepts
- ✅ Temporal feature engineering

</td>
</tr>
</table>

---

## 🛠️ Libraries Used

| Library | Purpose |
|---------|---------|
| **imageio** | Video reading/writing (v3 API) |
| **OpenCV** | Advanced video processing |
| **NumPy** | Array operations, math |
| **Matplotlib** | Visualization, plotting |
| **Seaborn** | Statistical plots |

---

## 📈 Sample Analysis

**Temporal Statistics:**
```
Total Frames: 240
Duration: 8 seconds
FPS: 30
Resolution: 320×240
```

**Motion Analysis:**
```
High motion frames: 45% (rapid action)
Medium motion: 35% (normal movement)
Low motion: 20% (nearly static)
```

**Scene Changes:**
- Detected: 3 major scene transitions
- Method: Frame differencing threshold
- Application: Video segmentation

---

## 🎯 Real-World Applications

**Video analysis techniques demonstrated:**

| Application | Technique Used |
|-------------|----------------|
| **Surveillance** | Motion detection, activity recognition |
| **Sports Analytics** | Action classification, pattern analysis |
| **Medical Imaging** | Temporal dynamics (heart rate, motion) |
| **Autonomous Vehicles** | Optical flow, object tracking |
| **Content Analysis** | Scene detection, video summarization |

---

## 🔗 Building on This

**Next steps in video analysis:**

- **Deep Learning:** Use CNNs for feature extraction
- **Action Recognition:** Classify activities (run, jump, etc.)
- **Object Tracking:** Follow objects across frames
- **3D Reconstruction:** Build 3D from 2D video
- **Real-time Processing:** Stream analysis

---

<div align="center">

**Master Video Analysis Fundamentals** 🎥

*2 notebooks • Temporal analysis • Motion detection*

[⬅️ Image EDA](../project_image_eda/) • [🏠 Home](../../README.md) • [📚 Unstructured Data Guide](../README_UNSTRUCTURED.md)

</div>
