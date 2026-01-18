# 🩺 Fetal State Classification: Cardiotocography (CTG) Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat&logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange?style=flat&logo=scikit-learn)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Data-Pandas-150458?style=flat&logo=pandas)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

## 📖 Executive Summary

This repository implements a comparative machine learning pipeline to classify fetal health states based on Cardiotocography (CTG) exam data. Leveraging the **UCI Machine Learning Repository's CTG dataset**, the system analyzes 21 physiological features—including baseline heart rate, accelerations, and uterine contractions—to predict the fetal state (NSP) as **Normal**, **Suspect**, or **Pathologic**.

The project encompasses the full data science lifecycle: Data Ingestion $\rightarrow$ Preprocessing $\rightarrow$ Exploratory Data Analysis (EDA) $\rightarrow$ Model Training $\rightarrow$ Comparative Evaluation.

## 📊 Dataset

The model utilizes the **Cardiotocography Data Set**, containing 2,126 fetal cardiotocograms (CTGs).

* **Source:** UCI Machine Learning Repository.
* **Input Features ($X$):** 36 attributes including FHR (Fetal Heart Rate), accelerations, fetal movement, and uterine contractions.
* **Target Variable ($y$):** `NSP` Class.
    1.  **Normal (N)**
    2.  **Suspect (S)**
    3.  **Pathologic (P)**

## 🏗️ Technical Architecture

The pipeline is constructed using **Python** and **Scikit-Learn**, following a modular design:

### 1. Preprocessing Pipeline
* **Data Cleaning:** Removal of administrative metadata (Filename, Date) and null value handling.
* **Feature Scaling:** Implementation of `StandardScaler` (z-score normalization) to enforce unit variance and zero mean across all features. *Note: Transformation is fitted strictly on the training set to prevent data leakage.*

### 2. Classification Models
The system evaluates four distinct algorithms to determine optimal performance:
* **Logistic Regression:** Baseline linear classifier (Multi-class).
* **K-Nearest Neighbors (KNN):** Non-parametric instance-based learning ($k=5$).
* **Decision Tree:** Non-linear recursive partitioning.
* **Support Vector Machine (SVM):** Linear kernel implementation with probability estimates.

### 3. Evaluation Metrics
* **Accuracy Score:** Global performance metric.
* **Confusion Matrix:** Visual heatmap to identify misclassification types (False Positives/Negatives) per class.

## 🚀 Installation & Usage

### Prerequisites
Ensure you have Python 3.8+ installed.

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/yourusername/fetal-health-classification.git](https://github.com/yourusername/fetal-health-classification.git)
    cd fetal-health-classification
    ```

2.  **Install Dependencies**
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn xlrd
    ```

3.  **Data Placement**
    Ensure the dataset file `CTG.xls` is located in the root directory.

4.  **Run the Analysis**
    ```bash
    python finalcardio.py
    ```

## 📉 Results & Visualization

Upon execution, the script generates:
1.  **Class Distribution Plot:** A bar chart visualizing the imbalance between Normal, Suspect, and Pathologic cases.
2.  **Accuracy Metrics:** Terminal output detailing the exact accuracy percentage for all four models.
3.  **Confusion Matrices:** Individual heatmaps for each model to visualize true vs. predicted labels.

| Model | Accuracy (Approx) | Characteristics |
| :--- | :--- | :--- |
| **Logistic Regression** | ~88% | Fast, interpretable baseline. |
| **Decision Tree** | ~92% | High variance, captures non-linear patterns. |
| **SVM (Linear)** | ~89% | Robust in high-dimensional space. |
| **KNN** | ~90% | Effective for local clustering. |

## 📂 Project Structure

```text
├── finalcardio.py       # Main execution script (ETL & ML Pipeline)
├── CTG.xls              # Raw dataset (Excel format)
├── README.md            # Project documentation
└── requirements.txt     # Python dependencies
```
##🤝 Contributing

Contributions are welcome. Please open an issue to discuss proposed changes or submit a Pull Request. Areas for improvement include:

    Hyperparameter tuning (GridSearchCV).

    Implementation of Ensemble methods (Random Forest, XGBoost).

    Cross-validation integration.



