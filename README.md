# Lung-cancer-prediction-using-KNN
This project uses the K-Nearest Neighbors (KNN) algorithm for the early detection of lung cancer based on patient health data. Key features include age, smoking status, air pollution exposure, and chronic lung disease. The model was trained on a dataset of 1000 records with 26 features, achieving 99% accuracy. Implemented in Python using scikit-learn, the system is compared against other models like SVM, Random Forest, and Logistic Regression, with KNN showing the best performance for real-time cancer prediction.

🎯 Objectives

- Build a classification model to detect lung cancer based on health-related features.
- Compare multiple machine learning models to determine the most accurate one.
- Provide insights that support early medical intervention and decision-making.

🧠 Techniques & Workflow
1. Data Preprocessing

Label encoding of categorical variables

Handling missing values

Feature scaling

2. Classification Models Used

Logistic Regression

Decision Tree

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN) ← Best performer

3. Evaluation Metrics

Accuracy Score

Confusion Matrix

Precision, Recall, F1-score

 🧠 Techniques Used

- **Data Preprocessing**:
  - Label Encoding of categorical features
  - Handling missing values
  - Feature scaling using StandardScaler

- **Machine Learning Models Evaluated**:
  - Logistic Regression
  - Decision Tree
  - Support Vector Machine (SVM)
  - **K-Nearest Neighbors (KNN)** ✅

- **Evaluation Metrics**:
  - Accuracy Score
  - Confusion Matrix
  - Precision, Recall, F1-Score



## 🧰 Tools & Libraries

| Tool/Library         | Purpose                                      |
|----------------------|----------------------------------------------|
| Python               | Core programming language                    |
| Google Colab         | Cloud-based Jupyter notebook environment     |
| Pandas, NumPy        | Data manipulation and numerical operations   |
| Seaborn, Matplotlib  | Data visualization                           |
| Scikit-learn         | ML models and evaluation metrics             |


## 📁 Dataset Description

The dataset contains medical records of patients including:
- Age
- Smoking habits
- Presence of symptoms like fatigue, shortness of breath, wheezing
- Alcohol consumption
- Chronic diseases
- Other related risk factors

🧪 Results Summary

The proposed KNN-based lung cancer prediction system was tested on a dataset with 1000 records and 26 features related to health conditions (e.g., Age, Smoking, Air Pollution).

Data was split using an 80:20 ratio for training and testing.

The KNN model consistently achieved high accuracy (99%), outperforming other models like:

SVM (100% but may overfit)

Random Forest (98%)

Logistic Regression (96%)

Naive Bayes (89%)


Evaluation was done using metrics: Accuracy, Precision, Recall, and F1-score — all close to 0.99 for KNN.

Tools used: Python 3.8.8, scikit-learn 0.24.2.





