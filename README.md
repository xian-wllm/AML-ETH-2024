Advanced Machine Learning – ETH Zurich (Autumn 2024)
====================================================

This repository contains multiple group projects completed as part of the **Advanced Machine Learning (AML)** course at **ETH Zurich**.The projects cover **regression**, **time-series classification**, and **deep learning–based medical image segmentation**, with a strong emphasis on robust preprocessing, principled model design, and rigorous evaluation.

Project 1 – Age Prediction from Brain MRI Features
--------------------------------------------------

### Overview

The objective of this project is to **predict a subject’s age** from **brain MRI–derived features** using regression techniques.The dataset is intentionally challenging and includes **outliers**, **irrelevant features**, and **missing values**, requiring careful preprocessing before model training.

### Pipeline

1.  **Outlier Detection**Identification and handling of anomalous samples and corrupted features.
    
2.  **Feature Selection**Removal of irrelevant and redundant features to reduce dimensionality and improve generalization.
    
3.  **Missing Value Imputation**Application of imputation strategies to ensure compatibility with machine learning models.
    
4.  **Regression Modeling**Training and evaluation of regression models to predict age from the cleaned feature space.
    

### Evaluation

*   Metric: Coefficient of Determination (R²)
    
*   Baseline requirement: R² ≥ 0.5
    

### Submission Format

Predictions must be submitted as a CSV file with two columns:

id,y

where y corresponds to the predicted age.

### Timeline

*   Project period: October 21 – November 11
    
*   Evaluation: Public and private leaderboards
    

Project 2 – Time-Series Classification: ECG Signal Analysis
-----------------------------------------------------------

### Overview

This project focuses on **classifying ECG signals** into predefined categories.The main challenges include **variable-length signals**, **NaN values**, and extracting meaningful features from raw biomedical time series.

### Pipeline

1.  **Preprocessing**
    
    *   Splitting raw ECG signals into individual heartbeats
        
    *   Recommended libraries: biosppy, neurokit, hrv
        
2.  **Feature Extraction**
    
    *   Computation of statistical descriptors per heartbeat (e.g., mean, variance).
        
3.  **Model Training**
    
    *   Supervised classification models trained on extracted features.
        

### Evaluation

*   Metric: F1-score (micro-averaged)
    
*   Baseline requirement: F1 ≥ 0.7 on the public leaderboard
    

### Submission Requirements

*   Predictions submitted via Kaggle
    
*   Individual written report submitted on Moodle
    
*   Maximum of 10 submissions per day per team
    

### Timeline

*   Project period: November 11, 3 PM – December 2, 2 PM
    
*   Private leaderboard reveal: December 2, 3 PM
    

Project 3 – Echocardiography Mitral Valve Segmentation
------------------------------------------------------

### Overview

This project implements a **deep learning pipeline** for **mitral valve segmentation** in echocardiography videos.The solution is based on a **U-Net architecture** and a **two-stage training strategy** leveraging both amateur-annotated and expert-annotated datasets.

Methodology
-----------

### 1\. Data Preprocessing (6\_PREPROCESSING2.ipynb)

Robust preprocessing is critical for handling variability in echocardiography data.

*   Image normalization to the \[0, 1\] range
    
*   Denoising using a pre-trained DRUNet model
    
*   Contrast enhancement via CLAHE
    
*   Padding to square shape and resizing to 224 × 224
    
*   Bounding box mask processed and used as an additional input channel
    
*   Ground truth masks refined using filtering, thresholding, dilation, and blurring
    

### 2\. Bounding Box Prediction (7\_TRAINING2\_BOX.ipynb)

Since bounding boxes are not provided in the test set:

*   A U-Net model is trained to predict bounding box masks
    
*   Training uses both amateur and expert datasets
    
*   Predicted boxes (guessed\_box) are generated for the test set
    
*   Some predicted boxes are manually refined before final segmentation
    

### 3\. Segmentation Model (model.py)

*   Architecture: U-Net with encoder–decoder structure and skip connections
    
*   Input channels:
    
    *   Preprocessed grayscale frame
        
    *   Corresponding bounding box mask
        
*   Enhancement: Squeeze-and-Excitation (SE) blocks for channel-wise feature recalibration
    

### 4\. Two-Stage Training (8\_TRAINING2.ipynb)

**Stage 1 – Amateur Data Training**

*   Training on a large amateur-annotated dataset
    
*   Strong data augmentation (flips, rotations, affine transforms)
    

**Stage 2 – Expert Data Fine-Tuning**

*   Fine-tuning on a smaller expert-annotated dataset
    
*   Lower learning rate for precise adaptation
    
*   Loss function: Power Jaccard Loss
    
*   Metrics: Dice coefficient and IoU
    

### 5\. Inference and Submission (9\_SUBMITTER2.ipynb)

1.  Load the best fine-tuned segmentation model
    
2.  Generate segmentation masks using video frames and refined bounding boxes
    
3.  Rescale predictions to original video resolution
    
4.  Apply run-length encoding and generate submission.csv
    

Workflow
--------

Recommended execution order:

1.  1\_images\_analysis.ipynb (optional exploration)
    
2.  6\_PREPROCESSING2.ipynb (final preprocessing)
    
3.  7\_TRAINING2\_BOX.ipynb (bounding box prediction)
    
4.  8\_TRAINING2.ipynb (two-stage segmentation training)
    
5.  9\_SUBMITTER2.ipynb (inference and submission)
    

Repository Structure
--------------------

Lab3/├── 1\_images\_analysis.ipynb├── 3\_PREPROCESSING.ipynb├── 6\_PREPROCESSING2.ipynb├── 7\_TRAINING2\_BOX.ipynb├── 8\_TRAINING2.ipynb├── 9\_SUBMITTER2.ipynb├── model.py├── dataset.py├── augment.py├── utils.py├── drunet/├── out/└── figures/
