import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV

# === 1️⃣ Load data ===
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv").values.ravel()
X_val = pd.read_csv("data/X_val.csv")
y_val = pd.read_csv("data/y_val.csv").values.ravel()
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").values.ravel()

# === 2️⃣ Preprocessing & Baseline Model ===
num_cols = X_train.select_dtypes(include="number").columns

pre = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ]), num_cols),
], remainder="drop")

rf = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42
)

pipe = Pipeline([("pre", pre), ("clf", rf)])
pipe.fit(X_train, y_train)

# === 3️⃣ Validasi Silang ===
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="f1_macro", n_jobs=-1)
print("CV F1-macro (train):", scores.mean(), "±", scores.std())

# === 4️⃣ Evaluasi di data validasi ===
y_val_pred = pipe.predict(X_val)
print("\nBaseline RF — F1(val):", f1_score(y_val, y_val_pred, average="macro"))
print("\nClassification Report (Validation):")
print(classification_report(y_val, y_val_pred, digits=3))

print("Jumlah data per kelas (train):")
print(pd.Series(y_train).value_counts())

# === 5️⃣ Grid Search Tuning ===
print("\n🔍 Melakukan Grid Search tuning...")

param = {
  "clf__max_depth": [None, 12, 20, 30],
  "clf__min_samples_split": [2, 5, 10]
}

gs = GridSearchCV(pipe, param_grid=param, cv=3,
                  scoring="f1_macro", n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)

print("\nBest params:", gs.best_params_)
best_model = gs.best_estimator_

y_val_best = best_model.predict(X_val)
print("\nBest RF — F1(val):", f1_score(y_val, y_val_best, average="macro"))
print("\nClassification Report (Validation - Best Model):")
print(classification_report(y_val, y_val_best, digits=3))

# === 6️⃣ Evaluasi Akhir (Test Set) ===
final_model = best_model  # gunakan model terbaik hasil GridSearch
y_test_pred = final_model.predict(X_test)

print("\n=== Evaluasi di Data Test ===")
print("F1(test):", f1_score(y_test, y_test_pred, average="macro"))
print(classification_report(y_test, y_test_pred, digits=3))
print("Confusion Matrix (test):")
print(confusion_matrix(y_test, y_test_pred))

# ROC-AUC (bila ada predict_proba)
if hasattr(final_model, "predict_proba"):
    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    try:
        print("ROC-AUC(test):", roc_auc_score(y_test, y_test_proba))
    except:
        pass

    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC (test)")
    plt.tight_layout()
    plt.savefig("roc_test.png", dpi=120)

    prec, rec, _ = precision_recall_curve(y_test, y_test_proba)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve (test)")
    plt.tight_layout()
    plt.savefig("pr_test.png", dpi=120)

# === 7️⃣ Pentingnya Fitur ===
try:
    importances = final_model.named_steps["clf"].feature_importances_
    fn = final_model.named_steps["pre"].get_feature_names_out()
    top = sorted(zip(fn, importances), key=lambda x: x[1], reverse=True)
    print("\nTop Feature Importance:")
    for name, val in top[:10]:
        print(f"{name}: {val:.4f}")
except Exception as e:
    print("Feature importance tidak tersedia:", e)

# === 8️⃣ Simpan Model ===
joblib.dump(final_model, "rf_model.pkl")
print("\n✅ Model disimpan sebagai rf_model.pkl")

# === 9️⃣ Cek Inference Lokal (contoh input fiktif) ===
mdl = joblib.load("rf_model.pkl")
sample = pd.DataFrame([{
    "IPK": 3.4,
    "Jumlah_Absensi": 4,
    "Waktu_Belajar_Jam": 7,
    "Rasio_Absensi": 4/14,
    "IPK_x_Study": 3.4*7
}])
print("\n🔮 Prediksi sampel fiktif:", int(mdl.predict(sample)[0]))
