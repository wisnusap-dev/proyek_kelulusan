import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# === 1. BACA DATA ===
df = pd.read_csv("kelulusan_mahasiswa.csv")
print(df.head())

# Pisahkan fitur dan label
X = df.drop(columns=["Lulus"])
y = df["Lulus"]

# === 2. SPLIT DATA DENGAN STRATIFY (BIAR SEIMBANG) ===
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print("Shape data:", X_train.shape, X_val.shape, X_test.shape)

# === 3. BASELINE MODEL: LOGISTIC REGRESSION ===
num_cols = X_train.select_dtypes(include="number").columns

pre = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ]), num_cols),
], remainder="drop")

logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
pipe_lr = Pipeline([("pre", pre), ("clf", logreg)])

pipe_lr.fit(X_train, y_train)
y_val_pred = pipe_lr.predict(X_val)

print("\nBaseline (LogReg) F1(val):", f1_score(y_val, y_val_pred, average="macro"))
print("\nClassification Report (Validation):")
print(classification_report(y_val, y_val_pred, digits=3))

# === 4. RANDOM FOREST + GRIDSEARCH ===
rf = RandomForestClassifier(random_state=42)
pipe_rf = Pipeline([("pre", pre), ("clf", rf)])

param_grid = {
    "clf__n_estimators": [50, 100, 200],
    "clf__max_depth": [3, 5, None],
}

grid = GridSearchCV(pipe_rf, param_grid, cv=3, scoring="f1_macro", n_jobs=-1)
grid.fit(X_train, y_train)

best_rf = grid.best_estimator_
y_val_pred_rf = best_rf.predict(X_val)

print("\nRandomForest F1(val):", f1_score(y_val, y_val_pred_rf, average="macro"))

# === 5. EVALUASI AKHIR (TEST SET) ===
final_model = best_rf  # pakai model terbaik
y_test_pred = final_model.predict(X_test)

print("\nF1(test):", f1_score(y_test, y_test_pred, average="macro"))
print("\nClassification Report (Test):")
print(classification_report(y_test, y_test_pred, digits=3))

print("\nConfusion matrix (test):")
print(confusion_matrix(y_test, y_test_pred))

# === 6. ROC-AUC (HANYA JIKA BISA) ===
if len(set(y_test)) > 1 and hasattr(final_model, "predict_proba"):
    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    print("ROC-AUC(test):", roc_auc_score(y_test, y_test_proba))

    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test Set)")
    plt.tight_layout()
    plt.savefig("roc_test.png", dpi=120)
else:
    print("ROC-AUC(test): tidak bisa dihitung (kelas test hanya satu)")

import joblib

joblib.dump(final_model, "model.pkl")
print("✅ Model tersimpan ke file model.pkl")
