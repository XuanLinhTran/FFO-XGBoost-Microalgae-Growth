import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import numpy as np
from numpy import genfromtxt
import xgboost

def ZScoreNorm(X):
    MeanX = np.mean(X, axis=0)
    StdX = np.std(X, axis=0)
    return (X - MeanX) / StdX, MeanX, StdX

def ZScoreNormY(X):
    MeanX = np.mean(X)
    StdX = np.std(X)
    return (X - MeanX) / StdX, MeanX, StdX

def ComputeSum():
    DataLoc = 'dataset.csv'
    dataset = genfromtxt(DataLoc, delimiter=',')
    Dim = dataset.shape[1]
    X0 = dataset[:, 0:Dim-1]
    Y0 = dataset[:, -1]
    
    X_train, meanX, stdX = ZScoreNorm(X0)
    Y_train, meanY, stdY = ZScoreNormY(Y0)
    
    X_queried = np.array([
        float(ent_value_NaNO3.get()), float(ent_value_KH2PO4.get()),
        float(ent_value_NaHCO3.get()), float(ent_value_Glucose.get()),
        float(ent_value_Age.get())
    ])
    
    X_queried_normalized = ((X_queried - meanX) / stdX).reshape((1, len(X_queried)))
    PredictionModel = xgboost.XGBRegressor()
    PredictionModel.load_model('Trained_FFO_XGBoost_Model.json')
    Yp_normalized = PredictionModel.predict(X_queried_normalized)
    Yp = Yp_normalized * stdY + meanY
    
    ent_value_MG.delete(0, tk.END)
    ent_value_MG.insert(0, str(Yp[0]))

window = tk.Tk()
window.title("Microalgae Growth Estimator")
window.resizable(False, False)
window.iconbitmap("test.ico")
frm_entry = tk.Frame(master=window)

# Load and display the microalgae image
img = Image.open("microalgae.png")  # Ensure the file exists
img = img.resize((400, 400), Image.LANCZOS)
photo = ImageTk.PhotoImage(img)
window.photo = photo  # Store a reference to avoid garbage collection
img_label = Label(master=frm_entry, image=photo)
img_label.grid(row=0, column=0, columnspan=3, pady=10)

# Title Label
label_title = tk.Label(master=frm_entry, text="Predictor Variables", font=('Helvetica', 14, 'bold'))
label_title.grid(row=1, column=0, columnspan=3, pady=10)

# Input fields and labels
labels = ["NaNO₃", "KH₂PO₄", "NaHCO₃", "Glucose", "Age"]
units = ["mg/L", "mg/L", "mg/L", "mg/L", "day"]
entries = []
X_input = [425, 100, 840, 0, 8]

for i, (label, unit) in enumerate(zip(labels, units)):
    tk.Label(master=frm_entry, text=label, font=('Helvetica', 12)).grid(row=i+2, column=0, sticky="w", padx=10, pady=5)
    entry = tk.Entry(master=frm_entry, width=12, font=('Helvetica', 12))
    entry.grid(row=i+2, column=1, sticky="e", padx=10, pady=5)
    entry.insert(0, X_input[i])
    entries.append(entry)
    tk.Label(master=frm_entry, text=unit, font=('Helvetica', 12)).grid(row=i+2, column=2, sticky="e", padx=10, pady=5)

ent_value_NaNO3, ent_value_KH2PO4, ent_value_NaHCO3, ent_value_Glucose, ent_value_Age = entries

# Compute Button with added spacing
tk.Label(master=frm_entry, text="", font=('Helvetica', 12)).grid(row=8, column=0, columnspan=3, pady=10)
tk.Button(master=frm_entry, text='Compute', font=('Helvetica', 12, 'bold'), command=ComputeSum, padx=10, pady=5).grid(row=9, column=1, pady=10)
tk.Label(master=frm_entry, text="", font=('Helvetica', 12)).grid(row=10, column=0, columnspan=3, pady=10)

# Output for Microalgae Growth
tk.Label(master=frm_entry, text="Microalgae Growth", font=('Helvetica', 12, 'bold')).grid(row=11, column=0, sticky="w", padx=10, pady=5)
ent_value_MG = tk.Entry(master=frm_entry, width=12, font=('Helvetica', 12))
ent_value_MG.grid(row=11, column=1, sticky="e", padx=10, pady=5)

frm_entry.grid(row=0, column=0, padx=20, pady=20)
window.mainloop()
