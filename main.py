import tkinter as tk
from tkinter import ttk, messagebox, Toplevel, filedialog
import pandas as pd
import sqlite3
import joblib
import os
import sys
import matplotlib
matplotlib.use("Agg")  # prevent backend issues before embedding
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, RocCurveDisplay
from sklearn.preprocessing import StandardScaler
import numpy as np

# --- Tooltip helper for hover info ---
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        widget.bind("<Enter>", self.show_tip)
        widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        if self.tipwindow or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 10
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify="left",
                         background="#ffffe0", relief="solid", borderwidth=1,
                         font=("Arial", 9), padx=6, pady=4, wraplength=340)
        label.pack(ipadx=1)

    def hide_tip(self, event=None):
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None

# Normal criteria / meanings shown in tooltips and info popups
NORMAL_INFO = {
    "age": "Age in years (adult).",
    "sex": "0 = Female, 1 = Male.",
    "cp": "Chest pain type:\n 0 Typical angina (exertional)\n 1 Atypical angina\n 2 Non‑anginal pain\n 3 Asymptomatic\n‘Normal’ generally means no chest pain.",
    "trestbps": "Resting blood pressure (mmHg).\nNormal: <120\nElevated: 120–129\nHypertension: ≥130",
    "chol": "Serum cholesterol (mg/dl).\nDesirable: <200\nBorderline high: 200–239\nHigh: ≥240",
    "fbs": "Fasting blood sugar >120 mg/dl (dataset threshold).\nClinical: Normal <100, Prediabetes 100–125, Diabetes ≥126.",
    "restecg": "Resting ECG: 0 Normal, 1 ST‑T abnormality, 2 LVH by Estes criteria.",
    "thalach": "Max heart rate achieved.\nAge‑predicted maximum ≈ 220 − age.",
    "exang": "Exercise‑induced angina: 1 Yes, 0 No.\n‘Normal’ is No (0).",
    "oldpeak": "ST depression induced by exercise relative to rest.\n‘Normal’ is near 0.",
    "slope": "Slope of peak exercise ST segment: 0 Upsloping, 1 Flat, 2 Downsloping.\nUpsloping is often normal.",
    "ca": "Number of major vessels (0–3/4) colored by fluoroscopy.\nLower is better; 0 is often normal.",
    "thal": "Thalassemia test: 0 Normal, 1 Fixed defect (scar), 2 Reversible defect (ischemia)."
}

# -------------------------------
# CONFIG
# -------------------------------
DEFAULT_CSV = "/mnt/data/heart.csv"  # user's uploaded dataset
MODEL_FILE = "heart_model.joblib"
SCALER_FILE = "heart_scaler.joblib"
DB_FILE = "prediction_log.db"

FEATURES = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal"
]
TARGET = "target"

# Human-friendly labels / option maps
LABELS = {
    "age": "Age (years)",
    "sex": "Sex",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure",
    "chol": "Cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar >120 mg/dl",
    "restecg": "Resting ECG",
    "thalach": "Max heart rate achieved",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression (oldpeak)",
    "slope": "Slope of peak exercise ST",
    "ca": "Number of major vessels (0-3/4)",
    "thal": "Thalassemia (thal)"
}

OPTION_MAPS = {
    "sex": [("Male",1), ("Female",0)],
    "cp": [("Typical angina",0),("Atypical angina",1),("Non-anginal pain",2),("Asymptomatic",3)],
    "fbs": [("True (>120)",1),("False (<=120)",0)],
    "restecg": [("Normal",0),("ST-T abnormality",1),("LVH",2)],
    "exang": [("Yes",1),("No",0)],
    "slope": [("Upsloping",0),("Flat",1),("Downsloping",2)],
    "ca": [("0",0),("1",1),("2",2),("3",3),("4",4)],
    "thal": [("Normal",0),("Fixed defect",1),("Reversible defect",2)]
}

NUMERIC_FIELDS = ["age","trestbps","chol","thalach","oldpeak"]

# -------------------------------
# DATA LOADING / TRAINING
# -------------------------------
def choose_or_default_csv():
    if os.path.exists(DEFAULT_CSV):
        return DEFAULT_CSV
    path = filedialog.askopenfilename(
        title="Select heart.csv (must contain 'target')",
        filetypes=[("CSV files","*.csv")]
    )
    if not path:
        raise FileNotFoundError("No dataset provided.")
    return path

def load_dataset():
    path = choose_or_default_csv()
    df = pd.read_csv(path)
    # normalize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]
    # standardize possible target naming
    if "num" in df.columns and TARGET not in df.columns:
        df.rename(columns={"num": TARGET}, inplace=True)
    # binarize target if needed
    if df[TARGET].max() > 1:
        df[TARGET] = (df[TARGET] > 0).astype(int)
    # type coercions (some datasets have strings)
    if "thal" in df.columns and df["thal"].dtype == object:
        map_thal = {"normal":0,"fixed":1,"fixed defect":1,"reversible":2,"reversable defect":2}
        df["thal"] = df["thal"].str.lower().map(map_thal).fillna(df["thal"]).astype(float)
    if "slope" in df.columns and df["slope"].dtype == object:
        map_slope = {"upsloping":0,"flat":1,"downsloping":2}
        df["slope"] = df["slope"].str.lower().map(map_slope).fillna(df["slope"]).astype(float)
    missing = [f for f in FEATURES+[TARGET] if f not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    df = df.dropna(subset=FEATURES+[TARGET]).copy()
    return df

def train_and_save(df):
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X_train_s, y_train)
    # eval
    y_pred = model.predict(X_val_s)
    y_proba = model.predict_proba(X_val_s)[:,1]
    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "roc_auc": float(roc_auc_score(y_val, y_proba)),
        "report": classification_report(y_val, y_pred, digits=3)
    }
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    return metrics, (X_val, y_val, y_pred, y_proba)


def ensure_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    # Create table if not exists (latest schema)
    cur.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        age REAL, sex REAL, cp REAL, trestbps REAL, chol REAL, fbs REAL, restecg REAL,
        thalach REAL, exang REAL, oldpeak REAL, slope REAL, ca REAL, thal REAL,
        prediction TEXT, probability REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()

    # Migrate: add any missing columns from older versions
    cur.execute("PRAGMA table_info(predictions)")
    cols = {row[1] for row in cur.fetchall()}  # column names set

    expected_types = {
        "age":"REAL","sex":"REAL","cp":"REAL","trestbps":"REAL","chol":"REAL","fbs":"REAL","restecg":"REAL",
        "thalach":"REAL","exang":"REAL","oldpeak":"REAL","slope":"REAL","ca":"REAL","thal":"REAL",
        "prediction":"TEXT","probability":"REAL","timestamp":"DATETIME"
    }

    for col, typ in expected_types.items():
        if col not in cols:
            if col == "timestamp":
                cur.execute(f"ALTER TABLE predictions ADD COLUMN {col} {typ} DEFAULT CURRENT_TIMESTAMP")
            else:
                cur.execute(f"ALTER TABLE predictions ADD COLUMN {col} {typ}")
    conn.commit()
    conn.close()

def log_prediction(rowdict, pred_text, prob):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO predictions (age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,prediction,probability)
        VALUES (:age,:sex,:cp,:trestbps,:chol,:fbs,:restecg,:thalach,:exang,:oldpeak,:slope,:ca,:thal,:prediction,:probability)
    ''', {**rowdict, "prediction": pred_text, "probability": float(prob)})
    conn.commit()
    conn.close()

def fetch_history():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT id, age, sex, prediction, probability, timestamp FROM predictions ORDER BY timestamp DESC")
    rows = cur.fetchall()
    conn.close()
    return rows

# -------------------------------
# GUI
# -------------------------------
class HeartGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Heart Disease Predictor — Kaggle Dataset")
        self.geometry("780x760")
        self.minsize(740, 720)

        ensure_db()
        # Load / train
        try:
            self.df = load_dataset()
        except Exception as e:
            messagebox.showerror("Dataset Error", str(e))
            self.destroy()
            return

        if not (os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE)):
            self._train_model_and_notify()
        else:
            try:
                self.model = joblib.load(MODEL_FILE)
                self.scaler = joblib.load(SCALER_FILE)
            except Exception as e:
                messagebox.showwarning("Model Load", f"Retraining due to error: {e}")
                self._train_model_and_notify()

        # theme hint
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        self._build_ui()

    def _train_model_and_notify(self):
        try:
            metrics, eval_pack = train_and_save(self.df)
            self.model = joblib.load(MODEL_FILE)
            self.scaler = joblib.load(SCALER_FILE)
            self.eval_pack = eval_pack
            messagebox.showinfo("Training Complete",
                                f"Accuracy: {metrics['accuracy']:.3f}\nROC-AUC: {metrics['roc_auc']:.3f}")
        except Exception as e:
            messagebox.showerror("Training Error", str(e))
            self.destroy()

    def _build_ui(self):
        container = ttk.Frame(self, padding=12)
        container.pack(expand=True, fill="both")

        # --- Input Card ---
        input_labelframe = ttk.LabelFrame(container, text="Input (patient features)")
        input_labelframe.pack(fill="x", padx=4, pady=6)

        self.widgets = {}
        grid_cols = 4
        r = 0; c = 0
        medians = self.df[FEATURES].median(numeric_only=True)

        
        for feat in FEATURES:
            label = ttk.Label(input_labelframe, text=LABELS.get(feat, feat))
            label.grid(row=r, column=c, sticky="w", padx=4, pady=6)

            # Info button with tooltip
            info_btn = ttk.Button(input_labelframe, text="ℹ", width=2, command=lambda f=feat: messagebox.showinfo(LABELS.get(f, f), NORMAL_INFO.get(f, "Information not available.")))
            info_btn.grid(row=r, column=c, sticky="e", padx=(0,28), pady=6)
            ToolTip(info_btn, NORMAL_INFO.get(feat, ""))

            if feat in OPTION_MAPS:
                values = [name for name,val in OPTION_MAPS[feat]]
                cb = ttk.Combobox(input_labelframe, values=values, state="readonly", width=22)
                cb.current(0)
                self.widgets[feat] = cb
            else:
                e = ttk.Entry(input_labelframe, width=24)
                e.insert(0, str(round(float(medians.get(feat, 0)), 2)))
                self.widgets[feat] = e

            self.widgets[feat].grid(row=r, column=c+1, sticky="ew", padx=4, pady=6)

            c += 2
            if c >= grid_cols:
                c = 0
                r += 1

        for i in range(grid_cols):
            input_labelframe.columnconfigure(i, weight=1)

        # Buttons row
        btns = ttk.Frame(container)
        btns.pack(fill="x", pady=6)
        ttk.Button(btns, text="Predict", command=self.on_predict).pack(side="left", padx=4)
        ttk.Button(btns, text="Autofill from dataset", command=self.autofill_from_dataset).pack(side="left", padx=4)
        ttk.Button(btns, text="Clear", command=self.clear_inputs).pack(side="left", padx=4)
        ttk.Button(btns, text="Show Evaluation", command=self.show_evaluation).pack(side="left", padx=4)
        ttk.Button(btns, text="History", command=self.show_history).pack(side="left", padx=4)
        ttk.Button(btns, text="Retrain", command=self.retrain).pack(side="left", padx=4)

        # Result label
        self.result_label = ttk.Label(container, text="Enter values and click Predict.",
                                      font=("Arial", 12, "bold"), anchor="center")
        self.result_label.pack(fill="x", pady=(6,2))

        # Status bar
        self.status = ttk.Label(self, text="Ready", anchor="w")
        self.status.pack(fill="x", side="bottom")

        # Helpful hint
        hint = ttk.Label(container, foreground="#555",
                         text="Tip: Use 'Autofill from dataset' to pull a real record and tweak values.")
        hint.pack(fill="x", pady=(2,8))

        # Collapsible 'Normal ranges & meanings'
        all_info = ttk.LabelFrame(container, text="📘 Normal ranges & meanings")
        all_info.pack(fill="x", padx=4, pady=(0,8))
        info_text = tk.Text(all_info, height=10, wrap="word")
        info_text.insert("1.0", "\n".join([f"{LABELS.get(k,k)}: {NORMAL_INFO[k]}" for k in FEATURES]))
        info_text.configure(state="disabled")
        info_text.pack(fill="x", padx=6, pady=6)

    def _value_for(self, feat):
        w = self.widgets[feat]
        if feat in OPTION_MAPS:
            name = w.get()
            mapping = dict(OPTION_MAPS[feat])
            return float(mapping[name])
        else:
            txt = w.get().strip()
            if txt == "":
                raise ValueError(f"Missing value for {feat}")
            try:
                return float(txt)
            except:
                raise ValueError(f"Invalid number in {feat}: {txt}")

    def _collect_inputs(self):
        vals = {}
        for f in FEATURES:
            vals[f] = self._value_for(f)
        return vals

    def on_predict(self):
        try:
            row = self._collect_inputs()
            X = pd.DataFrame([row])[FEATURES]
            Xs = self.scaler.transform(X)
            pred = int(self.model.predict(Xs)[0])
            p = float(self.model.predict_proba(Xs)[0][1])
            text = "Likely Heart Disease" if pred == 1 else "Likely Healthy"
            self.result_label.config(text=f"{text}  |  P(disease) = {p*100:.2f}%",
                                     foreground=("red" if pred==1 else "green"))
            log_prediction(row, text, p)
            self.status.config(text="Saved prediction to history.")
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

    def autofill_from_dataset(self):
        row = self.df.sample(1, random_state=np.random.randint(0, 1_000_000)).iloc[0]
        for f in FEATURES:
            val = row[f]
            if f in OPTION_MAPS:
                inv = {v:k for k,v in OPTION_MAPS[f]}
                if val in inv:
                    self.widgets[f].set(inv[val])
                else:
                    self.widgets[f].set(list(dict(OPTION_MAPS[f]).keys())[0])
            else:
                self.widgets[f].delete(0, tk.END)
                self.widgets[f].insert(0, str(val))
        self.status.config(text="Autofilled from dataset.")

    def clear_inputs(self):
        for f,w in self.widgets.items():
            if f in OPTION_MAPS:
                w.current(0)
            else:
                w.delete(0, tk.END)
        self.result_label.config(text="Cleared. Enter values and click Predict.", foreground="black")
        self.status.config(text="Inputs cleared.")

    def show_evaluation(self):
        try:
            X = self.df[FEATURES]
            y = self.df[TARGET]
            Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
            Xtr_s = self.scaler.transform(Xtr)
            Xva_s = self.scaler.transform(Xva)

            y_pred = self.model.predict(Xva_s)
            y_proba = self.model.predict_proba(Xva_s)[:,1]

            acc = accuracy_score(yva, y_pred)
            auc = roc_auc_score(yva, y_proba)
            cm = confusion_matrix(yva, y_pred)

            win = Toplevel(self)
            win.title("Model Evaluation")
            win.geometry("950x560")

            # Confusion matrix plot
            fig1, ax1 = plt.subplots(figsize=(4.5,4))
            im = ax1.imshow(cm, interpolation="nearest")
            ax1.set_title("Confusion Matrix")
            ax1.set_xticks([0,1]); ax1.set_yticks([0,1])
            ax1.set_xlabel("Predicted"); ax1.set_ylabel("Actual")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax1.text(j, i, cm[i, j], ha="center", va="center")
            fig1.tight_layout()

            # ROC plot
            fig2, ax2 = plt.subplots(figsize=(4.5,4))
            RocCurveDisplay.from_predictions(yva, y_proba, ax=ax2)
            ax2.set_title("ROC Curve")
            fig2.tight_layout()

            # embed both
            frm = ttk.Frame(win, padding=8)
            frm.pack(expand=True, fill="both")

            lbl = ttk.Label(frm, text=f"Accuracy: {acc:.3f}    ROC-AUC: {auc:.3f}", font=("Arial", 11, "bold"))
            lbl.pack(anchor="w", pady=(0,8))

            canv1 = FigureCanvasTkAgg(fig1, master=frm)
            canv1.draw()
            canv1.get_tk_widget().pack(side="left", expand=True, fill="both", padx=8)

            canv2 = FigureCanvasTkAgg(fig2, master=frm)
            canv2.draw()
            canv2.get_tk_widget().pack(side="left", expand=True, fill="both", padx=8)

        except Exception as e:
            messagebox.showerror("Evaluation Error", str(e))

    def show_history(self):
        rows = fetch_history()
        win = Toplevel(self)
        win.title("Prediction History")
        win.geometry("760x420")
        cols = ("id","age","sex","prediction","probability","timestamp")
        tree = ttk.Treeview(win, columns=cols, show="headings")
        for c in cols:
            tree.heading(c, text=c.capitalize())
            tree.column(c, width=120)
        tree.column("timestamp", width=160)

        sb = ttk.Scrollbar(win, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        tree.pack(side="left", expand=True, fill="both")

        for r in rows:
            tree.insert("", "end", values=r)

    def retrain(self):
        try:
            metrics, eval_pack = train_and_save(self.df)
            self.model = joblib.load(MODEL_FILE)
            self.scaler = joblib.load(SCALER_FILE)
            messagebox.showinfo("Retrained",
                                f"New Accuracy: {metrics['accuracy']:.3f}\nROC-AUC: {metrics['roc_auc']:.3f}")
            self.status.config(text="Model retrained.")
        except Exception as e:
            messagebox.showerror("Retrain Error", str(e))

if __name__ == "__main__":
    app = HeartGUI()
    app.mainloop()
