# ===========================
# pert7_revised.py — ANN untuk klasifikasi Lulus
# ===========================

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

# ===========================
# Langkah 1 — Siapkan Data
file_path = "data/processed_kelulusan.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"File {file_path} tidak ditemukan. Pastikan path benar.")

df = pd.read_csv(file_path)
X = df.drop("Lulus", axis=1)
y = df["Lulus"]

sc = StandardScaler()
Xs = sc.fit_transform(X)

# Split train 70%, sisanya 30% untuk val+test
X_train, X_temp, y_train, y_temp = train_test_split(
    Xs, y, test_size=0.3, random_state=42  # hilangkan stratify jika dataset kecil
)

# Split val/test masing-masing 50% dari sisa 30%
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42  # tanpa stratify
)

print("Shapes: X_train, X_val, X_test =", X_train.shape, X_val.shape, X_test.shape)

# ===========================
# Langkah 2 — Bangun Model ANN
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", "AUC"]
)

model.summary()

# ===========================
# Langkah 3 — Training dengan Early Stopping
es = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[es],
    verbose=1
)

# ===========================
# Langkah 4 — Evaluasi di Test Set
loss, acc, auc = model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy:", acc)
print("Test AUC:", auc)

y_proba = model.predict(X_test).ravel()
y_pred = (y_proba >= 0.5).astype(int)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=3))

# F1-score macro
f1_macro = f1_score(y_test, y_pred, average="macro")
print("F1-score (macro):", f1_macro)

# ROC-AUC safe check
try:
    roc = roc_auc_score(y_test, y_proba)
except ValueError:
    roc = np.nan
print("ROC-AUC:", roc)

# ===========================
# Langkah 5 — Visualisasi Learning Curve
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Learning Curve")
plt.tight_layout()
plt.savefig("learning_curve.png", dpi=120)
plt.show()

# ===========================
# Langkah 6 — Eksperimen (optional)
# Bisa ubah neuron, optimizer, Dropout, BatchNorm, dll.
