import kagglehub
import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==================================================
# RAG KNOWLEDGE BASE
# ==================================================

knowledge_base = {

    "symptoms": """
Stroke Symptoms:

- Sudden numbness or weakness in face, arm, or leg
- Trouble speaking or understanding speech
- Vision problems
- Loss of balance or difficulty walking
- Severe sudden headache
""",


    "prevention": """
Stroke Prevention:

- Exercise regularly
- Control blood pressure
- Maintain a healthy weight
- Stop smoking
- Control diabetes
- Eat a healthy diet
""",


    "treatment": """
Stroke Treatment:

- Emergency medical care
- Clot-busting medicines
- Surgery in some cases
- Physical therapy
- Speech therapy
- Rehabilitation
"""
}



# ==================================================
# RAG RETRIEVAL FUNCTION
# ==================================================

def rag_retrieve(question):

    question = question.lower()


    if any(word in question for word in 
           ["symptom", "sign", "warning"]):

        return knowledge_base["symptoms"]


    elif any(word in question for word in 
             ["prevent", "prevention", "avoid", "reduce"]):

        return knowledge_base["prevention"]


    elif any(word in question for word in 
             ["treat", "treatment", "medicine", "therapy"]):

        return knowledge_base["treatment"]


    else:

        return """
Sorry, information not found.

You can ask about:
- Stroke symptoms
- Stroke prevention
- Stroke treatment
"""



# ==================================================
# MACHINE LEARNING SECTION
# ==================================================

print("="*60)
print("      STROKE PREDICTION AGENT WITH RAG")
print("="*60)



# Download dataset

path = kagglehub.dataset_download(
    "fedesoriano/stroke-prediction-dataset"
)



csv_file = None


for file in os.listdir(path):

    if file.endswith(".csv"):

        csv_file = os.path.join(path,file)
        break



df = pd.read_csv(csv_file)



print("\nDataset Loaded")
print("Dataset Shape:", df.shape)



# ==================================================
# DATA PREPROCESSING
# ==================================================


df["bmi"] = df["bmi"].fillna(
    df["bmi"].median()
)



categorical_columns = [

    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status"

]



for col in categorical_columns:

    encoder = LabelEncoder()

    df[col] = encoder.fit_transform(
        df[col]
    )



df.drop(
    "id",
    axis=1,
    inplace=True
)



# ==================================================
# FEATURES AND LABEL
# ==================================================

X = df.drop(
    "stroke",
    axis=1
)


y = df["stroke"]



# ==================================================
# TRAIN TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y

)



# ==================================================
# MODEL TRAINING
# ==================================================

model = RandomForestClassifier(

    n_estimators=300,
    class_weight="balanced",
    random_state=42

)



model.fit(

    X_train,
    y_train

)



# ==================================================
# MODEL TESTING
# ==================================================

prediction = model.predict(
    X_test
)



accuracy = accuracy_score(

    y_test,
    prediction

)



print("\nModel Accuracy:")
print(
    round(accuracy*100,2),
    "%"
)



print("\nClassification Report:")

print(
    classification_report(
        y_test,
        prediction,
        zero_division=0
    )
)



# Save model

joblib.dump(

    model,
    "stroke_model.pkl"

)


print("\nModel Saved Successfully")



# ==================================================
# PATIENT PREDICTION
# ==================================================

print("\n")
print("="*60)
print("PATIENT DATA")
print("="*60)



gender = int(
input("Gender (0=Female,1=Male): ")
)


age = float(
input("Age: ")
)


hypertension = int(
input("Hypertension (0/1): ")
)


heart = int(
input("Heart Disease (0/1): ")
)


married = int(
input("Ever Married (0/1): ")
)


work = int(
input("Work Type (0-4): ")
)


residence = int(
input("Residence Type (0=Rural,1=Urban): ")
)


glucose = float(
input("Average Glucose Level: ")
)


bmi = float(
input("BMI: ")
)


smoking = int(
input("Smoking Status (0-3): ")
)



patient = pd.DataFrame({

"gender":[gender],

"age":[age],

"hypertension":[hypertension],

"heart_disease":[heart],

"ever_married":[married],

"work_type":[work],

"Residence_type":[residence],

"avg_glucose_level":[glucose],

"bmi":[bmi],

"smoking_status":[smoking]

})



result = model.predict(
    patient
)



print("\n")
print("="*60)



if result[0] == 1:

    print("HIGH STROKE RISK DETECTED")

else:

    print("LOW STROKE RISK")



print("="*60)



# ==================================================
# RAG ASSISTANT
# ==================================================

print("\n")
print("="*60)
print("RAG MEDICAL ASSISTANT")
print("="*60)



question = input(

"Ask about symptoms, prevention, or treatment: "

)



answer = rag_retrieve(
    question
)



print("\nRetrieved Knowledge:")

print(answer)



print("\nProgram Finished Successfully")