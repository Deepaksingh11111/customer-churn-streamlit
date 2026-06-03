# 📊 Customer Churn Prediction Web App

A sleek, data-driven web application built with **Streamlit** that utilizes a Machine Learning model to predict whether a customer is likely to churn (leave a service) or stay. This tool empowers businesses to proactively identify at-risk customers and implement targeted retention strategies.

🚀 **Live Demo:** [View the Live App](https://customer-churn-app-z3akt7drkmjkocexcfz4hf.streamlit.app/)

---

## 🧠 Model Training & Development

The machine learning pipeline behind this application was trained in Google Colab using a dataset of 7,043 customers. Since the dataset suffered from class imbalance, **SMOTE** (Synthetic Minority Over-sampling Technique) was implemented during pipeline development to balance the dataset and improve model reliability.

### 📊 Model Performance Evaluation
After evaluating multiple classifiers (Logistic Regression, SVM, XGBoost, and Random Forest), **Random Forest** yielded the highest accuracy score and was selected for parameter tuning:

* **Initial Random Forest Accuracy:** 78.11%
* **Cross-Validation Average Accuracy:** ~85.42%[cite: 2]
* **Final Fine-Tuned Model Accuracy:** 78.18%[cite: 2]

#### Classification Report:
```text
              precision    recall  f1-score   support

           0       0.86      0.84      0.85      1033
           1       0.59      0.61      0.60       374

    accuracy                           0.78      1407
   macro avg       0.72      0.73      0.72      1407
weighted avg       0.79      0.78      0.78      1407
💻 Complete Training Pipeline Code
Below is the clean code copied directly from the development notebook (project work.ipynb)[cite: 2]:

Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.syntax import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import joblib

warnings.filterwarnings('ignore')

# ==========================================
#  Load Dataset
# ==========================================
url = "[https://drive.google.com/uc?id=1rHeRG1swiTm2mIdw45Hg2fz4XBkU9jhk&export=download](https://drive.google.com/uc?id=1rHeRG1swiTm2mIdw45Hg2fz4XBkU9jhk&export=download)"
df = pd.read_csv(url)

# ==========================================
#  Data Cleaning
# ==========================================
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

# Remove Unnecessary Column
df.drop("customerID", axis=1, inplace=True)

# ==========================================
#  Encoding
# ==========================================
df = pd.get_dummies(df, drop_first=True)

# ==========================================
#  Define Features & Target
# ==========================================
X = df.drop("Churn", axis=1)
y = df["Churn"]

# ==========================================
#  Train Test Split
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
#  Feature Scaling
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
#  Handle Imbalance
# ==========================================
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train_scaled, y_train
)

# ==========================================
#  Train Best Models
# ==========================================
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

# Select Best Model
best_model_name = max(results, key=results.get)

# ==========================================
#  Hyperparameter Tuning
# ==========================================
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train_balanced, y_train_balanced)
best_model = grid_search.best_estimator_

# ==========================================
#  Prediction & Evaluation
# ==========================================
y_pred = best_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

# ==========================================
#  Save Model using Joblib
# ==========================================
joblib.dump(best_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model Saved Successfully")
✨ Web App Features
Real-time Predictions: Input customer demographics, account details, and activity metrics to get instant churn forecasts.

Interactive Dashboard: Clean, functional markdown data tables with optimized form validation.

Pre-trained Pipeline Workflow: Seamless backend parsing via standardized scaler.pkl transformations alongside a Random Forest classification architecture[cite: 2].

🛠️ Tech Stack
Frontend UI Framework: Streamlit

Languages & Environments: Python 3, Google Colab Notebooks[cite: 2]

Machine Learning Tools: Scikit-Learn, Imbalanced-Learn, XGBoost, Joblib[cite: 2]

📁 Repository Structure
Plaintext
├── app.py              # Main Streamlit web application source code
├── model.pkl           # Optimized trained Random Forest model artifact[cite: 2]
├── scaler.pkl          # Serialized StandardScaler data artifact[cite: 2]
└── requirements.txt    # Set of workspace software dependencies
🚀 Local Setup and Deployment
1. Clone the Repository
Bash
git clone [https://github.com/Deepaksingh11111/customer-churn-streamlit.git](https://github.com/Deepaksingh11111/customer-churn-streamlit.git)
cd customer-churn-streamlit
2. Set Up a Virtual Environment
Bash
# Setup
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate
3. Install Dependencies & Run
Bash
pip install -r requirements.txt
streamlit run app.py
🤝 Contributing
Contributions, issue tracking flags, and layout refinements are welcome! Feel free to review or start discussion boards via our open GitHub issues directory.


### 📝 Final Verification Step:
If your project depends on other libraries or images, you can reference files like `image_a4a6c0.png` or `mlp.pdf` directly using standard markdown notation inside this document if needed.
