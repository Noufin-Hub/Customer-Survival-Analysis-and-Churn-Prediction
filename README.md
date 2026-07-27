# 📊 Customer Survival Analysis and Telecom Customer Churn Prediction

An end-to-end Machine Learning and Survival Analysis project that predicts customer churn, explains the prediction using Explainable AI (SHAP), estimates Customer Lifetime Value (CLTV), and visualizes customer survival probability through an interactive Flask web application.

> Updated for **Python 3.14**, **Flask 3.x**, **scikit-learn 1.9**, **SHAP 0.52**, and **lifelines 0.30**.

---

## 🚀 Live Demo

Coming Soon

---

# 📌 Overview

Customer churn is one of the most important business metrics for telecom companies. Acquiring a new customer is significantly more expensive than retaining an existing one.

This project helps telecom companies identify customers who are likely to leave, understand *why* they are likely to churn, estimate how long they may remain customers, and calculate their expected Customer Lifetime Value (CLTV).

Unlike traditional churn prediction projects, this application combines:

- Machine Learning
- Survival Analysis
- Explainable AI
- Interactive Web Dashboard

into one complete solution.

---

# ✨ Features

- 🔹 Telecom Customer Churn Prediction
- 🔹 Customer Survival Analysis
- 🔹 Hazard Curve Visualization
- 🔹 Survival Probability Curve
- 🔹 Customer Lifetime Value (CLTV)
- 🔹 SHAP Explainability
- 🔹 Feature Importance
- 🔹 Partial Dependence Analysis
- 🔹 Gauge Meter Risk Visualization
- 🔹 Modern Flask Web Application
- 🔹 Python 3.14 Compatible

---

# 🖥️ Application Preview

> Replace this image after redesigning the UI.

```
app-pic.png
```

---

# 🏗️ Project Structure

```text
Customer-Survival-Analysis-and-Churn-Prediction/

│
├── Images/
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── templates/
│   └── index.html
│
├── Exploratory Data Analysis.ipynb
├── Customer Survival Analysis.ipynb
├── Churn Prediction Model.ipynb
│
├── app.py
│
├── model.pkl
├── survivemodel.pkl
├── explainer.bz2
│
├── requirements.txt
├── Procfile
├── README.md
└── LICENSE
```

---

# 📊 Dataset

**IBM Telco Customer Churn Dataset**

The dataset contains customer information including:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Internet Service
- Contract Type
- Payment Method
- Monthly Charges
- Total Charges
- Churn Status

---

# 🔬 Project Workflow

```
Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Feature Engineering
      │
      ▼
Random Forest Model
      │
      ▼
Hyperparameter Tuning
      │
      ▼
Customer Churn Prediction
      │
      ├────────────► SHAP Explainability
      │
      └────────────► Survival Analysis
                          │
                          ▼
                    Customer Lifetime Value
                          │
                          ▼
                    Flask Web Application
```

---

# 📈 Survival Analysis

The project performs Survival Analysis using the **Cox Proportional Hazards Model**.

It answers questions such as:

- How does churn probability change over time?
- Which customer characteristics increase churn risk?
- What is the expected customer lifetime?
- What is the cumulative hazard over time?

The application dynamically generates:

- Survival Curve
- Hazard Curve
- Customer Lifetime Value (CLTV)

---

# 🤖 Machine Learning

The churn prediction model uses a **Random Forest Classifier**.

### Training Pipeline

- Data Cleaning
- Label Encoding
- Train/Test Split
- Class Imbalance Handling
- Hyperparameter Tuning
- Cross Validation
- Model Evaluation

---

# ⚖️ Handling Class Imbalance

The telecom dataset contains an imbalanced target distribution.

The project uses:

- Class Weighting
- GridSearchCV
- Cross Validation

to improve recall and F1-score while minimizing false negatives.

---

# 📊 Model Performance

| Metric | Score |
|---------|--------|
| ROC-AUC | 0.85 |
| F1 Score | 0.62 |

---

# 🔍 Explainable AI

Understanding **why** a customer is predicted to churn is as important as the prediction itself.

This project includes:

### SHAP (SHapley Additive Explanations)

Explains:

- Which features increased churn probability
- Which features reduced churn probability

### Permutation Importance

Ranks the importance of features by measuring the decrease in model performance after shuffling feature values.

### Partial Dependence Plots

Implemented using **scikit-learn's PartialDependenceDisplay** to visualize the effect of individual features on churn probability.

---

# 📉 Customer Lifetime Value (CLTV)

Customer Lifetime Value is estimated using the survival function.

```
CLTV = Expected Customer Lifetime × Monthly Charges
```

The expected lifetime is calculated until the survival probability falls below **10%**.

---

# 🌐 Flask Web Application

The web application allows users to:

- Enter customer information
- Predict churn probability
- View customer risk level
- Understand prediction using SHAP
- View Hazard Curve
- View Survival Curve
- Estimate Customer Lifetime Value

---

# 🛠️ Tech Stack

### Programming

- Python 3.14

### Machine Learning

- scikit-learn
- SHAP
- lifelines
- NumPy
- Pandas
- SciPy

### Visualization

- Matplotlib
- Seaborn

### Backend

- Flask

### Frontend

- HTML
- CSS
- JavaScript
- Bootstrap

---

# 📦 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Customer-Survival-Analysis-and-Churn-Prediction.git
```

Move into the project

```bash
cd Customer-Survival-Analysis-and-Churn-Prediction
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

Windows

```bash
.venv\Scripts\activate
```

Linux/macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

Open

```
http://127.0.0.1:5000
```

---

# 🔄 Modernization

This project has been upgraded from the original implementation to support the latest Python ecosystem.

### Updates

- ✅ Python 3.14 Compatibility
- ✅ Flask 3.x Support
- ✅ scikit-learn 1.9 Compatibility
- ✅ SHAP 0.52 Compatibility
- ✅ lifelines 0.30 Compatibility
- ✅ Updated Random Forest Parameters
- ✅ Removed Deprecated APIs
- ✅ Replaced Deprecated PDPBox with scikit-learn Partial Dependence
- ✅ Updated Dependency Management
- ✅ Modern Flask Application Structure

---

# 🚀 Future Improvements

- Docker Support
- XGBoost Model
- LightGBM Model
- Deep Learning Model
- User Authentication
- Cloud Deployment
- REST API
- Interactive SHAP Dashboard
- Real-time Prediction
- Database Integration

---

# 📄 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Noufin P**

- MCA Graduate
- Aspiring Data Scientist
- AI & Applied Data Science Enthusiast

LinkedIn: (https://www.linkedin.com/in/noufinp)

GitHub: (https://github.com/Noufin-Hub)

---