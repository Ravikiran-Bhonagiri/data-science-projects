# Project: Video Exploratory Data Analysis

**Dataset:** [ImageIO Sample Video](https://imageio.readthedocs.io/en/stable/standardimages.html) (Cockatoo/Stent)
**Type:** Video Analysis (Computer Vision + Time)

## Overview
Video is just a sequence of images (frames) played rapidly. However, the *temporal* dimension adds complexity: motion, coherence, and flow. This project creates a pipeline to analyze video data using `imageio` and `OpenCV`.

## Goal
To implement a Video Analysis pipeline:
1.  **Frame Extraction:** Breaking video into 3D tensors (Time x Height x Width x Channels).
2.  **Temporal Statistics:** How does brightness/color change over time?
3.  **Optical Flow:** Visualizing the pattern of apparent motion of objects between frames.

## Notebooks
1.  `01_frame_extraction_and_analysis.ipynb`: Loading video, extracting metadata, and analyzing frame statistics.
2.  `02_temporal_dynamics_and_flow.ipynb`: Motion detection and dense optical flow visualization.

## Key Concepts
- **FPS (Frames Per Second)**
- **Tensor Representation** (T, H, W, C)
- **Frame Differencing**
- **Dense Optical Flow** (Farneback Algorithm)
