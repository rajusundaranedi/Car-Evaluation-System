# 🚗 Car Evaluation System Using Machine Learning

## 📌 Project Overview

The **Car Evaluation System** is a Machine Learning project that predicts the quality of a car based on various attributes such as buying price, maintenance cost, number of doors, passenger capacity, luggage boot size, and safety rating.

The project uses multiple Machine Learning algorithms and compares their performance to identify the best model for car evaluation classification. The system includes data preprocessing, exploratory data analysis (EDA), model training, hyperparameter tuning, visualization, feature importance analysis, and prediction capabilities. 

---

# 🎯 Objectives

* Analyze the Car Evaluation Dataset.
* Perform data preprocessing and feature encoding.
* Train and compare multiple machine learning models.
* Optimize the SVM model using GridSearchCV.
* Visualize dataset patterns and model performance.
* Generate predictions for new car configurations.
* Save trained models for future deployment.

---

# 📂 Dataset Information

### Dataset Name

**Car Evaluation Dataset**

### Source

University of California, Irvine

### Features

| Feature  | Description       |
| -------- | ----------------- |
| buying   | Buying price      |
| maint    | Maintenance cost  |
| doors    | Number of doors   |
| persons  | Seating capacity  |
| lug_boot | Luggage boot size |
| safety   | Safety rating     |

### Target Classes

| Class | Meaning      |
| ----- | ------------ |
| unacc | Unacceptable |
| acc   | Acceptable   |
| good  | Good         |
| vgood | Very Good    |

---

# 🛠 Technologies Used

```text
Python
NumPy
Pandas
Matplotlib
Seaborn
Scikit-Learn
Joblib
```

---

# 📁 Project Structure

```text
Car-Evaluation-System/
│
├── car_evaluation.py
│
├── outputs/
│   ├── eda_plots.png
│   ├── confusion_matrix.png
│   ├── kernel_comparison.png
│   ├── model_comparison.png
│   ├── pca_visualization.png
│   ├── feature_importance.png
│   └── learning_curve.png
│
├── models/
│   ├── best_car_model.pkl
│   └── car_scaler.pkl
│
└── README.md
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/car-evaluation-system.git

cd car-evaluation-system
```

## Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

---

# 🔍 Machine Learning Workflow

## 1. Data Loading

* Load dataset from UCI Repository.
* Create DataFrame.
* Handle fallback dataset if internet is unavailable.

---

## 2. Exploratory Data Analysis (EDA)

The project generates:

### Class Distribution Plot

Shows distribution of:

* Acceptable Cars
* Good Cars
* Very Good Cars
* Unacceptable Cars

### Safety Rating Distribution

Shows frequency of:

* Low Safety
* Medium Safety
* High Safety

Output:

```text
eda_plots.png
```

---

## 3. Data Preprocessing

### Label Encoding

Converts categorical features into numerical values.

Example:

```text
low   → 0
med   → 1
high  → 2
vhigh → 3
```

### Standardization

Uses:

```python
StandardScaler()
```

to normalize feature values before SVM training.

---

## 4. Baseline SVM Model

A default Support Vector Machine is trained using:

```python
SVC(kernel="rbf")
```

Performance is evaluated using:

* Accuracy Score
* Classification Report

---

## 5. Hyperparameter Tuning

GridSearchCV is used to optimize:

```python
C
Gamma
Kernel
```

Example:

```python
param_grid = {
    "C":[0.1,1,10,100],
    "gamma":["scale","auto",0.1,0.01],
    "kernel":["linear","poly","rbf"]
}
```

---

## 6. Model Evaluation

### Metrics Used

* Accuracy
* Precision
* Recall
* F1 Score
* Cross Validation Accuracy

### Outputs

```text
classification_report
confusion_matrix
```

Generated File:

```text
confusion_matrix.png
```

---

# 🤖 Models Compared

The project compares:

| Model               | Description               |
| ------------------- | ------------------------- |
| Logistic Regression | Linear classifier         |
| Random Forest       | Ensemble tree-based model |
| Gradient Boosting   | Boosting algorithm        |
| KNN                 | Distance-based classifier |
| SVM                 | Support Vector Machine    |

---

# 📊 Visualizations

## Kernel Comparison

Compares:

* Linear Kernel
* Polynomial Kernel
* RBF Kernel
* Sigmoid Kernel

Output:

```text
kernel_comparison.png
```

---

## PCA Visualization

Principal Component Analysis reduces data to 2 dimensions for visualization.

Output:

```text
pca_visualization.png
```

---

## Feature Importance

Random Forest identifies the most influential features.

Output:

```text
feature_importance.png
```

---

## Learning Curve

Shows:

* Training Accuracy
* Validation Accuracy

Output:

```text
learning_curve.png
```

---

## Model Comparison

Compares all machine learning models.

Output:

```text
model_comparison.png
```

---

# 🚗 Prediction Example

```python
predict_car(
    buying="low",
    maint="low",
    doors="4",
    persons="more",
    lug_boot="big",
    safety="high"
)
```

### Example Output

```text
Predicted Class: VGOOD
```

---

# 💾 Saving Models

The best-performing model is saved as:

```text
best_car_model.pkl
```

Scaler is saved as:

```text
car_scaler.pkl
```

Load later using:

```python
import joblib

model = joblib.load("best_car_model.pkl")
scaler = joblib.load("car_scaler.pkl")
```

---

# 📈 Generated Files

After execution, the following files are created:

```text
eda_plots.png
confusion_matrix.png
kernel_comparison.png
model_comparison.png
pca_visualization.png
feature_importance.png
learning_curve.png

best_car_model.pkl
car_scaler.pkl
```

---

# 🔮 Future Enhancements

* XGBoost Integration
* LightGBM Integration
* CatBoost Integration
* Streamlit Web Application
* Flask REST API
* Docker Deployment
* Cloud Deployment (AWS/Azure/GCP)
* Real-Time Car Recommendation System

---

# 👨‍💻 Author

**Car Evaluation System Using Machine Learning**

A complete Machine Learning project demonstrating data preprocessing, model selection, hyperparameter tuning, visualization, evaluation, and deployment-ready model persistence.
