"""
Heart Disease Analytics module for the Agentic Diagnostic application.

This is an application/demo screening component, not a clinically validated
diagnostic system.
"""

import re
import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
except ImportError as exc:
    raise ImportError(
        "scikit-learn is required. Install it with: pip install scikit-learn"
    ) from exc


class HeartDiseaseAnalytics:
    """Random Forest screening component used by app.py."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = self._build_demo_model()

    @staticmethod
    def _build_demo_model():
        rng = np.random.RandomState(42)
        n = 1200

        age = rng.randint(18, 90, n)
        systolic = np.clip(rng.normal(125, 22, n), 80, 220)
        heart_rate = np.clip(rng.normal(78, 14, n), 45, 150)
        glucose = np.clip(rng.normal(105, 30, n), 55, 300)
        male = rng.binomial(1, 0.5, n)

        score = (
            0.035 * (age - 50)
            + 0.025 * (systolic - 120)
            + 0.018 * (heart_rate - 75)
            + 0.012 * (glucose - 100)
            + 0.35 * male
            + rng.normal(0, 0.8, n)
        )
        y = (score > 1.25).astype(int)

        X = np.column_stack([age, systolic, heart_rate, glucose, male])

        model = RandomForestClassifier(
            n_estimators=160,
            max_depth=6,
            min_samples_leaf=4,
            random_state=42,
            class_weight="balanced",
        )
        model.fit(X, y)
        return model

    @staticmethod
    def _number(value, default=0.0):
        try:
            if value is None:
                return float(default)
            if isinstance(value, (int, float, np.number)):
                return float(value)
            match = re.search(r"-?\d+(?:\.\d+)?", str(value))
            return float(match.group()) if match else float(default)
        except Exception:
            return float(default)

    @classmethod
    def _systolic_bp(cls, blood_pressure):
        text = str(blood_pressure or "")
        match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", text)
        if match:
            return float(match.group(1))
        return cls._number(text, 120.0)

    @staticmethod
    def _gender_value(gender):
        return 1.0 if str(gender or "").strip().lower() in {
            "male", "m", "man"
        } else 0.0

    @staticmethod
    def _contains(text, keywords):
        text = str(text or "").lower()
        return any(k in text for k in keywords)

    def predict(
        self,
        age=0,
        gender="Male",
        blood_pressure="",
        heart_rate=0,
        blood_glucose=0,
        clinical_text="",
        medical_history="",
    ):
        age_value = self._number(age, 0)
        systolic = self._systolic_bp(blood_pressure)
        hr = self._number(heart_rate, 75)
        glucose = self._number(blood_glucose, 100)
        gender_value = self._gender_value(gender)

        combined_text = f"{clinical_text} {medical_history}".lower()

        cardiac_keywords = (
            "chest pain", "chest pressure", "chest tightness", "angina",
            "shortness of breath", "breathlessness", "dyspnea",
            "palpitation", "palpitations", "irregular heartbeat",
            "heart pain", "cardiac", "coronary", "heart disease",
            "heart failure", "myocardial", "fainting", "syncope",
        )

        history_keywords = (
            "hypertension", "high blood pressure", "diabetes",
            "high cholesterol", "hyperlipidemia", "smoker", "smoking",
            "family history of heart disease", "coronary artery disease",
            "previous heart attack", "heart attack",
        )

        relevant_symptoms = self._contains(combined_text, cardiac_keywords)
        history_signal = self._contains(combined_text, history_keywords)

        X = np.array([[age_value, systolic, hr, glucose, gender_value]])
        probability = float(self.model.predict_proba(X)[0][1])

        if relevant_symptoms:
            probability = min(0.99, probability + 0.18)
        elif not history_signal:
            probability = min(probability, 0.49)

        prediction = "YES" if probability >= 0.50 else "NO"

        if prediction == "YES":
            risk_level = (
                "High" if probability >= 0.75
                else "Moderate" if probability >= 0.50
                else "Low"
            )
            description = (
                "The available clinical information contains findings that "
                "warrant a positive heart-disease screening result. The "
                "screening probability is an ML estimate and should be "
                "correlated with clinical examination and appropriate "
                "diagnostic investigations."
            )
        else:
            risk_level = "Low"
            if not relevant_symptoms and not history_signal:
                description = (
                    "Heart disease analytics could not identify sufficient "
                    "heart-related symptoms or history in the available "
                    "clinical information. The screening result is NO; this "
                    "does not rule out disease and should not replace "
                    "clinical assessment."
                )
            else:
                description = (
                    "The available information does not produce a positive "
                    "heart-disease screening result. The result is NO, but "
                    "clinical correlation is still required if symptoms or "
                    "risk factors persist."
                )

        return {
            "prediction": prediction,
            "probability": probability,
            "risk_level": risk_level,
            "description": description,
            "model_used": "Random Forest Classifier",
        }
