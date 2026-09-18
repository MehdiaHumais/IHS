# ==========================================================
# STROKE AI AGENT
# ML + RAG + MCP FINAL VERSION
# ==========================================================

import kagglehub
import pandas as pd
import os
import joblib


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from imblearn.over_sampling import SMOTE



print("="*70)
print("        STROKE AI AGENT (ML + RAG + MCP)")
print("="*70)



# ==========================================================
# RAG KNOWLEDGE BASE
# ==========================================================

knowledge_base = {

"symptoms":
"""
Stroke Symptoms:

- Sudden numbness or weakness in face, arm, or leg
- Trouble speaking
- Vision problems
- Difficulty walking
- Loss of balance
- Severe sudden headache
""",


"prevention":
"""
Stroke Prevention:

- Exercise regularly
- Control blood pressure
- Maintain healthy weight
- Stop smoking
- Control diabetes
- Eat healthy food
""",


"treatment":
"""
Stroke Treatment:

- Emergency medical care
- Clot-busting medicines
- Surgery if required
- Physical therapy
- Speech therapy
- Rehabilitation
"""

}



# ==========================================================
# RAG RETRIEVER
# ==========================================================

def rag_retrieve(question):

    question = question.lower()


    if any(x in question for x in
           ["symptom","sign","warning"]):

        return knowledge_base["symptoms"]


    elif any(x in question for x in
             ["prevent","avoid","reduce"]):

        return knowledge_base["prevention"]


    elif any(x in question for x in
             ["treat","medicine","therapy"]):

        return knowledge_base["treatment"]


    else:

        return """
Available topics:

- Symptoms
- Prevention
- Treatment
"""



# ==========================================================
# MCP TOOLS
# ==========================================================


def calculate_bmi(weight,height):

    if height > 3:
        height = height / 100

    return round(
        weight/(height**2),
        2
    )




def classify_risk(probability):

    if probability >= 0.60:

        return "HIGH STROKE RISK"

    elif probability >= 0.30:

        return "MODERATE STROKE RISK"

    else:

        return "LOW STROKE RISK"




def create_report(
        name,
        age,
        bmi,
        probability,
        risk):


    return f"""

================================
        STROKE REPORT
================================

Patient:
{name}

Age:
{age}

BMI:
{bmi}

Stroke Probability:
{probability*100:.2f}%

Prediction:
{risk}

================================

"""




def recommendations(risk):

    if risk == "HIGH STROKE RISK":

        return """

Recommendations:

- Seek medical consultation
- Monitor blood pressure
- Follow healthy diet
- Exercise regularly
- Avoid smoking

"""


    elif risk == "MODERATE STROKE RISK":

        return """

Recommendations:

- Improve lifestyle habits
- Exercise regularly
- Monitor glucose level
- Reduce unhealthy food

"""


    else:

        return """

Recommendations:

- Continue healthy lifestyle
- Maintain healthy weight
- Regular checkups

"""



# ==========================================================
# DATASET
# ==========================================================


print("\nLoading Dataset...")


path = kagglehub.dataset_download(
    "fedesoriano/stroke-prediction-dataset"
)



csv_file = None


for file in os.listdir(path):

    if file.endswith(".csv"):

        csv_file = os.path.join(
            path,
            file
        )

        break



df = pd.read_csv(csv_file)



print(
"Dataset:",
df.shape
)



# ==========================================================
# PREPROCESSING
# ==========================================================


df["bmi"] = df["bmi"].fillna(
    df["bmi"].median()
)



columns = [

"gender",
"ever_married",
"work_type",
"Residence_type",
"smoking_status"

]


for col in columns:

    encoder = LabelEncoder()

    df[col] = encoder.fit_transform(
        df[col]
    )



df.drop(
"id",
axis=1,
inplace=True
)



X = df.drop(
"stroke",
axis=1
)


y = df["stroke"]



# ==========================================================
# TRAINING
# ==========================================================


X_train,X_test,y_train,y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y

)



smote = SMOTE(
random_state=42
)



X_train,y_train = smote.fit_resample(
    X_train,
    y_train
)



print(
"\nSMOTE Applied"
)



model = RandomForestClassifier(

    n_estimators=400,

    class_weight="balanced",

    random_state=42

)



model.fit(
X_train,
y_train
)



# ==========================================================
# EVALUATION
# ==========================================================


prediction = model.predict(
X_test
)


probabilities = model.predict_proba(
X_test
)[:,1]



print("\nMODEL PERFORMANCE")


print(
"Accuracy:",
round(
accuracy_score(y_test,prediction)*100,
2
),
"%"
)



print(
"ROC-AUC:",
round(
roc_auc_score(y_test,probabilities),
3
)
)



print(
"\nClassification Report"
)


print(
classification_report(
y_test,
prediction,
zero_division=0
)
)



print(
"\nConfusion Matrix"
)


print(
confusion_matrix(
y_test,
prediction
)
)



joblib.dump(
model,
"stroke_model.pkl"
)



print(
"\nModel Saved"
)



# ==========================================================
# STREAMLIT MEDICAL ASSISTANT FUNCTION
# ==========================================================


def ask_medical_agent(question):

    response = rag_retrieve(question)

    final_response = f"""
{response}


DISCLAIMER:

This AI system is for educational purposes only.
It does not replace professional medical diagnosis.
"""

    return final_response



# ==========================================================
# RAG ASSISTANT FUNCTION FOR STREAMLIT
# ==========================================================


def ask_medical_agent(question):

    response = rag_retrieve(question)

    final_response = f"""
{response}


DISCLAIMER:

This AI system is for educational purposes only.
It does not replace professional medical diagnosis.
"""

    return final_response
