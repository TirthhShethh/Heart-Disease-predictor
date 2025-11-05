# Heart-Disease-predictor
# 💙 Heart Disease Predictor

A desktop application built with Python that uses a machine learning model to predict the likelihood of heart disease based on user input.



This project is a complete demonstration of how to integrate several key Python technologies into one functional application:
* **Machine Learning** (`scikit-learn`) for prediction.
* **Desktop GUI** (`tkinter`) for a user-friendly interface.
* **Database Connectivity** (`sqlite3`) for logging prediction history.
* **Data Visualization** (`seaborn`, `matplotlib`) for data analysis.
* **File Handling** (`joblib`) for saving and loading the trained model.

## 🚀 Features

* **Predictive Model:** Uses a Logistic Regression model trained on a heart disease dataset to make predictions.
* **Intuitive GUI:** A clean `tkinter` interface to input patient data (age, sex, cholesterol, etc.).
* **Real-time Results:** Instantly receive a prediction ("Likely Heart Disease" or "Likely Healthy") with a confidence score.
* **Prediction History:** Every prediction is automatically saved to a local **SQLite database**. View all past entries in the "View History" window.
* **Data Analysis:** Click "Show Analysis" to see a **Seaborn** scatter plot visualizing the training data (Age vs. Cholesterol).

## 🛠️ Tech Stack / Requirements

This project uses the following Python libraries:

* `pandas`
* `scikit-learn`
* `seaborn`
* `matplotlib`
* `joblib`

`tkinter` and `sqlite3` are part of the Python standard library and do not need to be installed separately.

## 🏃 How to Run

1.  **Clone or download** this repository.
2.  **Install the dependencies** using pip:
    ```bash
    pip install pandas scikit-learn seaborn matplotlib joblib
    ```
3.  **Run the application:**
    ```bash
    python main.py
    ```

### How it Works:

* **On the first run**, the script will automatically train the ML model and save it as `heart_model.joblib`. It will also create the `prediction_log.db` SQLite database file.
* **On subsequent runs**, it will skip training and directly load the saved model file.