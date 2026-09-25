import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("stress_dataset_large.csv")

print("=== DATASET ===")
print(df.head())
print("\nJumlah data:", len(df))
print("\nJumlah kelas:")
print(df["stress_level"].value_counts())

label_map = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

df["stress_level"] = df["stress_level"].map(label_map)

X = df[["bpm", "skin_temp"]]
y = df["stress_level"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\n=== HASIL EVALUASI ===")
print("Accuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:")
print(classification_report(
    y_test,
    pred,
    target_names=["Low", "Medium", "High"]
))

joblib.dump(model, "stress_predict_model.joblib")
print("\nModel tersimpan sebagai stress_predict_model.joblib")

# Contoh prediksi
contoh = [[106, 34.5]]
hasil = int(model.predict(contoh)[0])

inverse_label = {
    0: "Low",
    1: "Medium",
    2: "High"
}

print("\n=== CONTOH PREDIKSI ===")
print("BPM       :", contoh[0][0])
print("Skin Temp :", contoh[0][1])
print("Prediction:", hasil)
print("Label     :", inverse_label[hasil])
