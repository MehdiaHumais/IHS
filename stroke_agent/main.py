import kagglehub
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from imblearn.over_sampling import SMOTE

print("=" * 50)
print("      STROKE PREDICTION AGENT")
print("=" * 50)

# ==================================================
# DOWNLOAD DATASET
# ==================================================

path = kagglehub.dataset_download(
    "fedesoriano/stroke-prediction-dataset"
)

print("\nDataset downloaded successfully!")

csv_file = None

for file in os.listdir(path):
    if file.endswith(".csv"):
        csv_file = os.path.join(path, file)
        break

df = pd.read_csv(csv_file)

print("\nDataset Shape:", df.shape)

# ==================================================
# PREPROCESSING
# ==================================================

df["bmi"] = df["bmi"].fillna(df["bmi"].median())

categorical_columns = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status"
]

encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

df.drop("id", axis=1, inplace=True)

# ==================================================
# FEATURES & TARGET
# ==================================================

X = df.drop("stroke", axis=1)
y = df["stroke"]

# ==================================================
# TRAIN TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==================================================
# SMOTE BALANCING
# ==================================================

print("\nApplying SMOTE...")

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(
    X_train,
    y_train
)

print("Balanced Training Shape:", X_train.shape)

# ==================================================
# MODEL TRAINING
# ==================================================

print("\nTraining Model...")

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# ==================================================
# EVALUATION
# ==================================================

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nAccuracy:")
print(round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

# ==================================================
# SAVE MODEL
# ==================================================

joblib.dump(
    model,
    "stroke_model.pkl"
)

print("\nModel saved as stroke_model.pkl")

# ==================================================
# USER PREDICTION
# ==================================================

print("\n")
print("=" * 50)
print("ENTER PATIENT INFORMATION")
print("=" * 50)

gender = int(input("Gender (0=Female, 1=Male): "))
age = float(input("Age: "))
hypertension = int(input("Hypertension (0/1): "))
heart_disease = int(input("Heart Disease (0/1): "))
ever_married = int(input("Ever Married (0=No,1=Yes): "))
work_type = int(input("Work Type (0-4): "))
residence = int(input("Residence Type (0=Rural,1=Urban): "))
glucose = float(input("Average Glucose Level: "))
bmi = float(input("BMI: "))
smoking = int(input("Smoking Status (0-3): "))

sample = pd.DataFrame({
    "gender": [gender],
    "age": [age],
    "hypertension": [hypertension],
    "heart_disease": [heart_disease],
    "ever_married": [ever_married],
    "work_type": [work_type],
    "Residence_type": [residence],
    "avg_glucose_level": [glucose],
    "bmi": [bmi],
    "smoking_status": [smoking]
})

prediction = model.predict(sample)

print("\n" + "=" * 50)

if prediction[0] == 1:
    print("⚠ HIGH STROKE RISK DETECTED")
else:
    print("✓ LOW STROKE RISK")

print("=" * 50)