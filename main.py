import tkinter as tk
from tkinter import ttk, messagebox, Toplevel
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sqlite3
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# --- 1. MODEL TRAINING & FILE HANDLING ---
# (This part handles the ML and File Handling)

# We'll use a small, classic dataset directly in the code to avoid file-not-found errors.
# This is a tiny sample of the Heart Disease dataset.
DATA = {
    'age': [63, 67, 67, 37, 41, 56, 62, 57, 63, 53, 57, 56, 44, 52, 57, 54, 48, 54, 50, 58, 58, 57, 58, 60, 50, 44, 47, 66, 60, 41],
    'sex': [1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0],
    'cp': [1, 4, 4, 3, 2, 2, 4, 4, 4, 4, 3, 2, 2, 3, 3, 2, 3, 4, 3, 2, 2, 3, 2, 3, 4, 3, 3, 4, 4, 2],
    'trestbps': [145, 160, 120, 130, 130, 120, 140, 120, 130, 140, 120, 140, 120, 172, 150, 150, 110, 140, 120, 130, 136, 140, 130, 150, 140, 140, 110, 178, 130, 105],
    'chol': [233, 286, 229, 250, 204, 236, 268, 354, 254, 203, 303, 294, 263, 199, 168, 195, 229, 239, 244, 275, 230, 264, 216, 240, 233, 226, 211, 228, 206, 198],
    'target': [0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]
}
MODEL_FILE = 'heart_model.joblib'
SCALER_FILE = 'heart_scaler.joblib'

def get_data():
    """Returns the training data."""
    return pd.DataFrame(DATA)

def train_model():
    """Trains and saves the model if it doesn't exist."""
    if not os.path.exists(MODEL_FILE):
        print("Training new model...")
        df = get_data()
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train a simple model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_scaled, y)
        
        # Save the model and scaler (File Handling)
        joblib.dump(model, MODEL_FILE)
        joblib.dump(scaler, SCALER_FILE)
        print(f"Model and scaler saved to {MODEL_FILE} and {SCALER_FILE}")
    else:
        print("Model already exists. Loading.")

# --- 2. DATABASE CONNECTIVITY ---
# (This part handles the SQLite database)

DB_FILE = 'prediction_log.db'

def setup_database():
    """Creates the database table if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        age INTEGER,
        sex INTEGER,
        cp INTEGER,
        trestbps INTEGER,
        chol INTEGER,
        prediction TEXT,
        probability REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    conn.close()

def log_prediction(data, prediction, probability):
    """Saves a prediction to the SQLite database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO predictions (age, sex, cp, trestbps, chol, prediction, probability)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (data['age'], data['sex'], data['cp'], data['trestbps'], data['chol'], prediction, probability))
    conn.commit()
    conn.close()

def get_prediction_history():
    """Fetches all prediction records from the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, age, sex, prediction, probability, timestamp FROM predictions ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows

# --- 3. TKINTER GUI APPLICATION ---
# (This part builds the user interface)

class HeartApp(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Heart Disease Predictor")
        self.geometry("400x350")
        
        # Load model and scaler
        try:
            self.model = joblib.load(MODEL_FILE)
            self.scaler = joblib.load(SCALER_FILE)
        except FileNotFoundError:
            messagebox.showerror("Error", "Model file not found. Please run the script to train.")
            self.destroy()
            return
            
        # Configure styles
        style = ttk.Style(self)
        style.configure("TLabel", font=("Arial", 11))
        style.configure("TButton", font=("Arial", 11))
        style.configure("TEntry", font=("Arial", 11))
        style.configure("Result.TLabel", font=("Arial", 12, "bold"))

        # --- Create Widgets ---
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(expand=True, fill="both")
        
        self.entries = {}
        features = ['age', 'sex (1=M, 0=F)', 'cp (chest pain 1-4)', 'trestbps (BP)', 'chol (cholesterol)']
        self.feature_keys = ['age', 'sex', 'cp', 'trestbps', 'chol']

        for i, text in enumerate(features):
            label = ttk.Label(main_frame, text=f"{text}:")
            label.grid(row=i, column=0, sticky="w", pady=5)
            
            entry = ttk.Entry(main_frame, width=20)
            entry.grid(row=i, column=1, sticky="ew", pady=5)
            self.entries[self.feature_keys[i]] = entry
            
        # --- Buttons ---
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=len(features), column=0, columnspan=2, pady=10)
        
        self.predict_button = ttk.Button(button_frame, text="Predict", command=self.on_predict)
        self.predict_button.pack(side="left", padx=5)
        
        self.analysis_button = ttk.Button(button_frame, text="Show Analysis", command=self.show_analysis)
        self.analysis_button.pack(side="left", padx=5)
        
        self.history_button = ttk.Button(button_frame, text="View History", command=self.show_history)
        self.history_button.pack(side="left", padx=5)
        
        # --- Result Label ---
        self.result_label = ttk.Label(main_frame, text="Enter values and press Predict", style="Result.TLabel", wraplength=350, justify="center")
        self.result_label.grid(row=len(features)+1, column=0, columnspan=2, pady=(10, 0))

        main_frame.columnconfigure(1, weight=1)

    def on_predict(self):
        """Called when the 'Predict' button is clicked."""
        try:
            # 1. Get data from GUI
            input_data = {key: float(self.entries[key].get()) for key in self.feature_keys}
            
            # 2. Format for model
            data_df = pd.DataFrame([input_data])
            data_scaled = self.scaler.transform(data_df)
            
            # 3. Make prediction (ML)
            prediction_val = self.model.predict(data_scaled)[0]
            probability = self.model.predict_proba(data_scaled)[0][prediction_val]
            
            result_text = "Likely Heart Disease" if prediction_val == 1 else "Likely Healthy"
            probability_pct = f"{probability*100:.2f}%"
            
            # 4. Update GUI
            self.result_label.config(text=f"Result: {result_text}\nConfidence: {probability_pct}", 
                                     foreground="red" if prediction_val == 1 else "green")
                                     
            # 5. Log to database (Database Connectivity)
            log_prediction(input_data, result_text, probability)
            
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numbers for all fields.")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

    def show_analysis(self):
        """Called when 'Show Analysis' is clicked. (Seaborn)"""
        analysis_window = Toplevel(self)
        analysis_window.title("Data Analysis")
        analysis_window.geometry("600x450")
        
        # Create a matplotlib figure and a seaborn plot
        fig, ax = plt.subplots(figsize=(6, 4))
        df = get_data()
        sns.scatterplot(
            data=df, 
            x='age', 
            y='chol', 
            hue='target', 
            style='sex', 
            ax=ax,
            palette={0: 'green', 1: 'red'}
        )
        ax.legend(title='Target', labels=['Healthy (0)', 'Disease (1)'])
        ax.set_title("Age vs. Cholesterol (from Training Data)")
        
        # Embed the plot in the tkinter window
        canvas = FigureCanvasTkAgg(fig, master=analysis_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    def show_history(self):
        """Called when 'View History' is clicked. (Database)"""
        history_window = Toplevel(self)
        history_window.title("Prediction History")
        history_window.geometry("600x400")
        
        # Create a Treeview to show the database table
        columns = ("id", "age", "sex", "prediction", "probability", "timestamp")
        tree = ttk.Treeview(history_window, columns=columns, show="headings")
        
        for col in columns:
            tree.heading(col, text=col.capitalize())
            tree.column(col, width=100)
            
        tree.column("timestamp", width=150)
        
        # Get data from DB and populate the tree
        rows = get_prediction_history()
        for row in rows:
            tree.insert("", "end", values=row)
            
        # Add scrollbar
        scrollbar = ttk.Scrollbar(history_window, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side="right", fill="y")
        tree.pack(side="left", fill="both", expand=True)

# --- 4. RUN THE APPLICATION ---
if __name__ == "__main__":
    # First, make sure the model and database are set up
    train_model()
    setup_database()
    
    # Then, run the GUI
    app = HeartApp()
    app.mainloop()