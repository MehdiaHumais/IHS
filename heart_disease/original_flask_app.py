# app.py - Heart Disease Prediction with Template-Based Clinical Summary
import os
from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# =========================================================
# LOAD DATA + TRAIN MODEL
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "cleaned_merged_heart_dataset.csv")

df = pd.read_csv(CSV_PATH)

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalachh", "exang", "oldpeak", "slope", "ca", "thal"
]

X = df[FEATURES]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

test_accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Model trained. Test accuracy: {test_accuracy:.2%}")


def generate_clinical_summary(patient_data, risk_probability):
    """
    Generate clinical summary based on patient data and risk level.
    This uses template-based logic, no LLM needed.
    """
    
    # Extract key values
    age = int(patient_data.get('age', 0))
    sex = "Male" if patient_data.get('sex') == 1 else "Female"
    bp = int(patient_data.get('trestbps', 0))
    chol = int(patient_data.get('chol', 0))
    heart_rate = int(patient_data.get('thalachh', 0))
    st_depression = float(patient_data.get('oldpeak', 0))
    exercise_angina = "present" if patient_data.get('exang') == 1 else "absent"
    major_vessels = int(patient_data.get('ca', 0))
    
    # Determine risk category
    if risk_probability > 70:
        risk_category = "HIGH RISK"
        urgency = "immediate further evaluation"
    elif risk_probability > 40:
        risk_category = "MODERATE RISK"
        urgency = "follow-up testing recommended"
    else:
        risk_category = "LOW RISK"
        urgency = "continue routine monitoring"
    
    # Build risk factors list
    risk_factors = []
    
    if bp > 140:
        risk_factors.append("elevated blood pressure")
    if chol > 240:
        risk_factors.append("high cholesterol")
    if st_depression > 1.0:
        risk_factors.append("significant ST depression")
    if major_vessels > 1:
        risk_factors.append(f"multiple vessel involvement ({major_vessels} vessels)")
    if exercise_angina == "present":
        risk_factors.append("exercise-induced angina")
    if heart_rate < 60:
        risk_factors.append("low resting heart rate")
    
    # Generate summary
    if risk_factors:
        factors_text = ", ".join(risk_factors)
        summary = f"{age}-year-old {sex} with {risk_category} profile ({risk_probability:.0f}% probability). Key findings include {factors_text}. Clinical recommendation: {urgency}. Always consult qualified physician."
    else:
        summary = f"{age}-year-old {sex} with {risk_category} profile ({risk_probability:.0f}% probability). Vital signs and ECG findings are largely reassuring. Recommendation: {urgency}. Always consult qualified physician."
    
    return summary[:300]  # Limit to 300 characters


# =========================================================
# ROUTES
# =========================================================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        row = [float(data[f]) for f in FEATURES]
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400

    input_df = pd.DataFrame([row], columns=FEATURES)

    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])
    probability_percent = probability * 100
    
    # Generate clinical summary
    patient_dict = dict(zip(FEATURES, row))
    clinical_summary = generate_clinical_summary(patient_dict, probability_percent)
    
    # Determine risk level
    if probability_percent > 70:
        risk_level = "High Risk"
    elif probability_percent > 40:
        risk_level = "Moderate Risk"
    else:
        risk_level = "Low Risk"

    return jsonify({
        "prediction": prediction,
        "label": "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected",
        "probability": round(probability_percent, 1),
        "risk_level": risk_level,
        "clinical_summary": clinical_summary
    })


if __name__ == "__main__":
    app.run(debug=True)