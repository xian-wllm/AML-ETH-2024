# Projects - Advanced Machine Learning ETHZ Automn 2024

## AML Project 1 - Age Prediction from Brain MRI Features 

This repository is part of a group project for the Advanced Machine Learning course at **ETH Zurich**. The project focuses on predicting a person's age from brain MRI data using regression techniques. The primary goal is to preprocess the provided MRI-derived features and build a model that can predict age with high accuracy.

### Project Overview 

The main task is to predict the age of a person based on brain MRI features. The data has been perturbed with outliers, irrelevant features, and missing values to make the task more challenging. The workflow includes the following preprocessing steps:
 
1. **Outlier Detection:**  Identifying and handling outliers in the dataset.
 
2. **Feature Selection:**  Selecting relevant features to reduce dimensionality and improve model performance.
 
3. **Imputation of Missing Values:**  Filling in missing data to ensure compatibility with machine learning algorithms.

After preprocessing, we will train regression models to predict the target variable (age) and evaluate performance using the Coefficient of Determination (R²) score.

### Evaluation 

The R² score will be used to measure the accuracy of the age predictions, with a baseline score of 0.5 required to pass the project.

### Submission Requirements 
The submission includes a file with two columns: `id` and `y`, where `y` is the predicted age. The project duration is from October 21 to November 11, with public and private leaderboards for evaluation.


## AML Project 2 - Timeseries Classification: ECG Signal Analysis

This repository is part of a group project for the **Advanced Machine Learning** course at **ETH Zurich**. The goal is to classify ECG signals into predefined categories using machine learning models.

### Project Overview

The task is to preprocess and classify ECG signals, handling challenges such as variable signal lengths and NaN values. The workflow includes:

1. **Preprocessing:**  
   Splitting raw ECG signals into individual heartbeats using libraries like `biosppy`, `neurokit`, or `hrv`.

2. **Feature Extraction:**  
   Extracting statistical features (e.g., mean, variance) to represent heartbeats.

3. **Model Training:**  
   Training classification models and evaluating performance using the **F1-score** (`micro` averaging).

### Evaluation

The **F1-score** is the primary metric:

```python
from sklearn.metrics import f1_score
F1 = f1_score(y_true, y_pred, average='micro')
```
The baseline F1-score to pass the project is 0.7 on the public leaderboard.

### Submission Requirements

- Submit predictions to the Kaggle competition.  
- Each student must submit an individual description of their approach in Moodle.  
- Teams are limited to **10 submissions per day**.

### Tools and Resources

- Recommended libraries: `biosppy`, `neurokit`, `hrv`.  
- A sample Jupyter Notebook is provided for preprocessing.

### Deadlines

- **Project period:** November 11, 3 PM – December 2, 2 PM.  
- **Leaderboard access:**  
  - **Public leaderboard:** Available during the competition.  
  - **Private leaderboard:** Revealed after December 2, 3 PM.  

Refer to the competition rules on Kaggle for further details.