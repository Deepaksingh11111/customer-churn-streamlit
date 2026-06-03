# 📊 Customer Churn Prediction Web App

A sleek, data-driven web application built with **Streamlit** that utilizes a Machine Learning model to predict whether a customer is likely to churn (leave a service) or stay. This tool empowers businesses to proactively identify at-risk customers and implement targeted retention strategies.

🚀 **Live Demo:** [View the Live App](https://customer-churn-app-z3akt7drkmjkocexcfz4hf.streamlit.app/)

---

## ✨ Features
* **Real-time Predictions:** Input customer demographics, account details, and activity metrics to get instant churn probabilities.
* **Interactive Dashboard:** Intuitive user interface with classic styling and seamless validation.
* **Pre-trained ML Pipeline:** Leverages an optimized Machine Learning model backed by standardized feature scaling for highly accurate predictions.

---

## 🛠️ Tech Stack
* **Frontend/UI:** [Streamlit](https://streamlit.io/)
* **Language:** Python
* **Machine Learning:** Scikit-learn, Pandas, NumPy
* **Model Serialization:** Pickle (`.pkl`)

---

## 📁 Repository Structure
```text
├── app.py              # Main Streamlit application source code
├── model.py            # Machine learning model architecture/training logic (optional)
├── model.pkl           # Trained classification model file
├── scaler.pkl          # Serialized StandardScaler/MinMaxScaler instance
└── requirements.txt    # List of dependencies required to run the project

🚀 Local Setup and Installation
Follow these steps to clone this repository and run the application locally on your machine:

1. Clone the Repository
git clone [https://github.com/Deepaksingh11111/customer-churn-streamlit.git](https://github.com/Deepaksingh11111/customer-churn-streamlit.git)
cd customer-churn-streamlit
2. Set Up a Virtual Environment (Recommended)
Bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
Bash
pip install -r requirements.txt
4. Run the Streamlit App
Bash
streamlit run app.py
Your local browser should automatically open the app at http://localhost:8501.

📈 How It Works
Data Preprocessing: The application captures user input (such as tenure, monthly charges, contract type, etc.) and uses scaler.pkl to scale numerical features exactly how the model expects them.

Model Inference: The scaled features are fed into the pre-trained model.pkl file.

Output Generation: The app outputs whether the customer is at a high risk of churning along with a visual confidence score percentage.

🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check out the issues page if you want to contribute to the project's development.
📝 License
This project is open-source and available under the MIT License.


### How to add this to your repository:
1. Erase the single `j` that is currently on line 1 of your editor in the screenshot.
2. Copy the code block above.
3. Paste it directly into the GitHub text area.
4. Click the green **"Commit changes..."** button in the top right corner to save it to your `main` branch.
