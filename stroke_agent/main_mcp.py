# ==================================================
# MCP STROKE AGENT
# ==================================================

print("====================")
print(" MCP STROKE AGENT ")
print("====================")


# ==================================================
# MCP TOOLS
# ==================================================


def calculate_bmi(weight, height):

    # If height is entered in cm, convert to meters
    if height > 3:
        height = height / 100

    bmi = weight / (height ** 2)

    return round(bmi, 2)



def patient_summary(patient):

    report = f"""

============================
PATIENT REPORT
============================

Age:
{patient['age']}

Glucose Level:
{patient['glucose']}

Weight:
{patient['weight']} kg

Height:
{patient['height']} cm

BMI:
{patient['bmi']}

============================
"""

    return report



def risk_tool(prediction):

    if prediction == 1:

        return "HIGH STROKE RISK"

    else:

        return "LOW STROKE RISK"



def generate_recommendation(risk):

    if risk == "HIGH STROKE RISK":

        return """

Recommendations:

- Consult a doctor immediately
- Monitor blood pressure
- Maintain healthy diet
- Exercise regularly
- Avoid smoking
"""

    else:

        return """

Recommendations:

- Continue healthy lifestyle
- Exercise regularly
- Maintain normal blood pressure
- Regular health checkups
"""



# ==================================================
# USER INPUT
# ==================================================


age = int(
    input("Age: ")
)


glucose = float(
    input("Glucose level: ")
)


weight = float(
    input("Weight kg: ")
)


height = float(
    input("Height (cm or meter): ")
)



# ==================================================
# MCP TOOL CALL 1
# BMI CALCULATION
# ==================================================

bmi = calculate_bmi(
    weight,
    height
)



# ==================================================
# CREATE PATIENT DATA
# ==================================================

patient = {

    "age": age,

    "glucose": glucose,

    "weight": weight,

    "height": height,

    "bmi": bmi

}



# ==================================================
# MCP TOOL CALL 2
# REPORT GENERATION
# ==================================================

report = patient_summary(
    patient
)



print(report)



# ==================================================
# STROKE PREDICTION TOOL
# (Temporary prediction)
# Replace with ML model later
# ==================================================


# Example logic

if age > 60 or glucose > 140 or bmi > 30:

    prediction = 1

else:

    prediction = 0



# ==================================================
# MCP TOOL CALL 3
# RISK ASSESSMENT
# ==================================================

risk = risk_tool(
    prediction
)



print("Stroke Prediction:")
print(risk)



# ==================================================
# MCP TOOL CALL 4
# RECOMMENDATION
# ==================================================

recommendation = generate_recommendation(
    risk
)



print(recommendation)



print("====================")
print(" MCP AGENT FINISHED ")
print("====================")