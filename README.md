# Dry Bean Classification Using Machine Learning

## 1. Project Overview

This project implements and compares five supervised Machine Learning classification algorithms using the Dry Bean Dataset.

The objective is to classify dry beans into seven different categories based on their physical characteristics using machine learning techniques.

The project also includes an interactive Streamlit web application that allows users to upload test data, select a classification model, and evaluate its performance.

## 2. Dataset Description

**Dataset:** Dry Bean Dataset  
**Source:** UCI Machine Learning Repository  
**Dataset ID:** 602  
**Total Instances:** 13,611  
**Number of Features:** 16  
**Number of Classes:** 7  

### Bean Categories

- BARBUNYA
- BOMBAY
- CALI
- DERMASON
- HOROZ
- SEKER
- SIRA

### Features

The dataset contains 16 numerical features describing the physical characteristics of dry beans, including area, perimeter, major axis length, minor axis length, eccentricity, compactness and shape factors.

## 3. Machine Learning Algorithms Implemented

The following five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## 4. Data Preprocessing

The following preprocessing steps were performed:

- Checked missing values and duplicate records.
- Performed exploratory data analysis (EDA).
- Analyzed feature distributions and correlations.
- Encoded target class labels using Label Encoding.
- Split the dataset into 80% training data and 20% testing data.
- Applied Standard Scaling for Logistic Regression, KNN and SVM.
- Used stratified sampling to preserve class distribution.

## 5. Model Evaluation

All five models were evaluated using the following metrics:

- Accuracy
- Area Under the ROC Curve (AUC)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Performance Comparison

| Machine Learning Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9207 | 0.9934 | 0.9215 | 0.9207 | 0.9209 | 0.9042 |
| Decision Tree | 0.8917 | 0.9330 | 0.8912 | 0.8917 | 0.8913 | 0.8691 |
| KNN | 0.9166 | 0.9812 | 0.9174 | 0.9166 | 0.9168 | 0.8992 |
| Gaussian Naive Bayes | 0.7639 | 0.9644 | 0.7654 | 0.7639 | 0.7615 | 0.7154 |
| Support Vector Machine | **0.9221** | **0.9937** | **0.9224** | **0.9221** | **0.9222** | **0.9059** |

### Results and Observations

- Support Vector Machine achieved the highest overall performance on the test dataset.
- Logistic Regression demonstrated performance close to SVM.
- KNN also achieved strong classification results.
- Decision Tree performed reasonably well.
- Gaussian Naive Bayes achieved the lowest accuracy among the five models.
- SVM achieved 92.21% accuracy and an AUC of 0.9937.

Based on the evaluated test results, Support Vector Machine was the best-performing model.

## 6. Streamlit Web Application

An interactive web application was developed using Streamlit.

### Application Features

- Upload a CSV test dataset.
- Select one of the five trained classification models.
- Generate class predictions.
- Display Accuracy, AUC, Precision, Recall, F1 Score and MCC.
- Visualize the confusion matrix.
- Display the classification report.
- Download prediction results as a CSV file.

## 7. Project Structure

```text
ML_Assignment_2/
│
├── Assignment_2_ACI_Code.ipynb
├── app.py
├── test_data.csv
├── requirements.txt
├── README.md
│
└── models/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── gaussian_naive_bayes.pkl
    ├── svm.pkl
    ├── scaler.pkl
    └── label_encoder.pkl
```

## 8. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib
- Jupyter Notebook
- Git and GitHub

## 9. Installation and Execution

### Step 1: Clone the repository

```bash
git clone https://github.com/idevmou/MTech-DSE-ML-Assignment-2.git
```

### Step 2: Navigate to the project directory

```bash
cd MTech-DSE-ML-Assignment-2
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit application

```bash
streamlit run app.py
```

### Step 5: Upload the test dataset

Upload `test_data.csv` through the Streamlit application and select a classification model to evaluate its performance.

## 10. Project Links

**GitHub Repository:**  
https://github.com/idevmou/MTech-DSE-ML-Assignment-2

**Live Streamlit Application:**  
https://mtech-dse-ml-assignment-2.streamlit.app/

## 11. Conclusion

This project demonstrates the implementation and comparison of five supervised machine learning classification algorithms using the Dry Bean Dataset.

Based on the experimental results, Support Vector Machine achieved the highest accuracy of 92.21%, followed closely by Logistic Regression with 92.07%.

The Streamlit application provides an interactive interface for model selection, prediction and performance evaluation.

---

**Project:** M.Tech Data Science and Engineering – Machine Learning Assignment 2  
**Institution:** BITS Pilani (WILP)
