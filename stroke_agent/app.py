import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from datetime import datetime

from utils.database import (
    create_database,
    save_prediction,
    get_history
)

from utils.pdf_report import generate_pdf


# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Brain Stroke Diagnosis",
    page_icon="🧠",
    layout="wide"
)


create_database()


# ==========================
# OCEAN BLUE THEME
# ==========================

st.markdown("""
<style>

/* =========================
   STROKEGUARD AI THEME
   Sea Green + Sage + Beige Glass
========================= */


/* Main background */
.stApp {
    background: linear-gradient(
        135deg,
        #9ADBC8 0%,
        #B7E4C7 35%,
        #DDE5B6 70%,
        #F1E8C8 100%
    );
}


/* Content spacing */
.block-container {
    padding-top: 2rem;
}


/* Rounded sections */
div[data-testid="stVerticalBlock"] {
    border-radius: 25px;
}


/* Glass cards */
div[data-testid="stMetric"],
div[data-testid="stExpander"],
.stAlert {

    background: rgba(255,255,255,0.65);
    border-radius: 25px;
    border: 1px solid rgba(255,255,255,0.5);
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);

}


/* Headings */
h1, h2, h3 {
    color: #006D5B;
    font-weight: 800;
}


/* Text */
p, label, span {
    color: #334E48;
}


/* Inputs */
div[data-baseweb="input"] input,
div[data-baseweb="select"] {

    background-color: rgba(255,255,255,0.85);
    border-radius: 12px;

}


/* Buttons */
.stButton > button {

    background: linear-gradient(
        135deg,
        #52B788,
        #74C69D
    );

    color: white;
    border-radius: 15px;
    border: none;
    font-weight: 700;

}


.stButton > button:hover {

    background: linear-gradient(
        135deg,
        #40916C,
        #52B788
    );

}


/* Sidebar */
section[data-testid="stSidebar"] {

    background: #D8F3DC;

}


/* Focus glow */
input:focus,
textarea:focus,
div[data-baseweb="select"]:focus-within {

    border-color: #52B788 !important;
    box-shadow: 0 0 0 1px #52B788 !important;

}


/* Dropdown */
div[data-baseweb="popover"] {

    background-color: #F1FAEE;

}


/* Embedded Brain Stroke Diagnosis readability and layout */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: linear-gradient(
        135deg,
        #B7E4C7 0%,
        #BDE8D2 48%,
        #C9F1F5 100%
    ) !important;
}
.block-container {
    max-width: 100% !important;
    padding: 1.25rem 1.5rem 2.5rem !important;
}
.hero-container {
    background: rgba(255,255,255,0.58) !important;
    border: 1px solid rgba(255,255,255,0.55) !important;
    border-radius: 18px !important;
    padding: 28px 32px !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08) !important;
}
.hero-title { color: #006D78 !important; }
.hero-title .stroke { color: #006D78 !important; }
.hero-title .guard { color: #E0569A !important; }
.hero-subtitle { color: #334E48 !important; }
.hero-tagline { color: #087F68 !important; }
.section-title {
    color: #172033 !important;
    font-size: 1.22rem !important;
    line-height: 1.4 !important;
    font-weight: 800 !important;
    margin: 0 0 .7rem 0 !important;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,0.58) !important;
    border: 1px solid rgba(255,255,255,0.50) !important;
    border-radius: 16px !important;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.05) !important;
}
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
.stTextInput label, .stNumberInput label, .stSelectbox label,
div[data-baseweb="select"] *, div[data-baseweb="input"] input,
input, textarea {
    color: #172033 !important;
    opacity: 1 !important;
    -webkit-text-fill-color: #172033 !important;
}
div[data-baseweb="input"], div[data-baseweb="select"] > div {
    background: rgba(245,246,250,0.94) !important;
    border-color: rgba(213,221,232,0.85) !important;
}
div[data-baseweb="input"] input::placeholder {
    color: #8a96a8 !important;
    opacity: 1 !important;
    -webkit-text-fill-color: #8a96a8 !important;
}
[data-testid="stAlert"] p, [data-testid="stAlert"] span, [data-testid="stAlert"] div,
[data-testid="stMetricLabel"] *, [data-testid="stMetricValue"] * {
    color: #17365d !important;
    opacity: 1 !important;
    -webkit-text-fill-color: #17365d !important;
}
.stButton > button {
    min-height: 3rem !important;
    background: linear-gradient(135deg, #52B788, #74C69D) !important;
    color: #ffffff !important;
    border-radius: 9px !important;
    font-weight: 800 !important;
}
.stButton > button p, .stButton > button span {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)


# ==========================
# HERO HEADER
# ==========================

st.markdown("""
<style>

.hero-container{
    background: rgba(255,255,255,0.55);
    border-radius: 30px;
    padding: 35px;
    margin-bottom: 30px;
}

.hero-title{
    font-size: 3rem;
    font-weight: 800;
}

.stroke{
    color:#006D5B;
}

.guard{
    color:#E0569A;
}

.hero-subtitle{
    font-size:1.2rem;
    color:#334E48;
    margin-top:10px;
}

.hero-tagline{
    margin-top:20px;
    font-weight:600;
    color:#006D5B;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-container">

<div class="hero-title">
🧠 <span class="stroke">Brain Stroke</span><span class="guard"> Diagnosis</span>
</div>

<div class="hero-subtitle">
AI-powered Stroke Risk Prediction & Medical Assistance
</div>

<div class="hero-tagline">
Predict • Analyze • Understand • Prevent
</div>

</div>
""", unsafe_allow_html=True)
# ==========================
# LOAD MODEL
# ==========================

try:

    model = joblib.load(
        "stroke_model.pkl"
    )

except Exception as e:

    st.error(
        "❌ stroke_model.pkl not found"
    )

    st.write(e)

    st.stop()


# ==========================
# HELPER FUNCTIONS
# ==========================

def calculate_bmi(weight, height):

    if height == 0:
        return 0

    height = height / 100

    return round(
        weight / (height * height),
        2
    )


def get_risk_level(probability):

    if probability < 0.30:
        return "LOW RISK"

    elif probability < 0.60:
        return "MODERATE RISK"

    else:
        return "HIGH RISK"


def get_recommendations(probability):

    if probability >= 0.60:

        return [

            "Consult a healthcare professional",

            "Monitor blood pressure regularly",

            "Follow a heart-healthy diet",

            "Avoid smoking",

            "Exercise consistently"

        ]

    elif probability >= 0.30:

        return [

            "Improve daily lifestyle habits",

            "Reduce processed foods",

            "Maintain a healthy weight",

            "Schedule regular health checkups"

        ]

    else:

        return [

            "Continue healthy habits",

            "Stay physically active",

            "Maintain balanced nutrition"

        ]


# ==========================
# PATIENT INFORMATION
# ==========================

with st.container(border=True):
    st.markdown(
        """
        <div class="section-title">
        👤 Patient Information
        </div>
        """,
        unsafe_allow_html=True
    )

    row1 = st.columns(5)
    with row1[0]:
        patient_name = st.text_input("Patient Name", placeholder="Enter patient name")
    with row1[1]:
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
    with row1[2]:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with row1[3]:
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    with row1[4]:
        work_type = st.selectbox(
            "Work Type",
            ["Private", "Self-employed", "Government Job", "Children", "Never Worked"]
        )

    row2 = st.columns(5)
    with row2[0]:
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    with row2[1]:
        ever_married = st.selectbox("Ever Married", ["No", "Yes"])
    with row2[2]:
        residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    with row2[3]:
        smoking_status = st.selectbox(
            "Smoking Status",
            ["never smoked", "formerly smoked", "smokes"]
        )

# ==========================
# HEALTH METRICS
# ==========================

with st.container(border=True):
    st.markdown(
        """
        <div class="section-title">
        📈 Health Metrics
        </div>
        """,
        unsafe_allow_html=True
    )

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        avg_glucose = st.number_input(
            "Average Glucose Level",
            min_value=50.0,
            max_value=300.0,
            value=100.0
        )

    with m3:
        weight = st.number_input(
            "Weight (KG)",
            min_value=20.0,
            max_value=250.0,
            value=70.0
        )

    with m4:
        height = st.number_input(
            "Height (CM)",
            min_value=50.0,
            max_value=250.0,
            value=170.0
        )

    bmi = calculate_bmi(weight, height)

    with m2:
        st.metric("BMI (Calculated)", f"{bmi:.2f}")

    if bmi < 18.5:
        bmi_status = "Underweight"
    elif bmi < 25:
        bmi_status = "Normal"
    elif bmi < 30:
        bmi_status = "Overweight"
    else:
        bmi_status = "Obese"

    st.info(
        f"ℹ️ Calculated BMI: {bmi:.2f}\n\n"
        f"Your BMI is in the **{bmi_status}** range."
    )

# ==========================
# STROKE RISK ANALYSIS
# ==========================

st.markdown(
    """
    <div class="section-title">
    🧠 Stroke Risk Assessment
    </div>
    """,
    unsafe_allow_html=True
)


if st.button(
    "🚀 Analyze Stroke Risk",
    use_container_width=True
):


    # --------------------------
    # ENCODE VALUES
    # --------------------------

    gender_value = (
        1 if gender == "Male"
        else 0
    )


    hypertension_value = (
        1 if hypertension == "Yes"
        else 0
    )


    heart_value = (
        1 if heart_disease == "Yes"
        else 0
    )


    married_value = (
        1 if ever_married == "Yes"
        else 0
    )


    work_mapping = {

        "Government Job": 0,

        "Children": 1,

        "Private": 2,

        "Self-employed": 3,

        "Never Worked": 4

    }


    smoking_mapping = {

        "never smoked": 0,

        "formerly smoked": 1,

        "smokes": 2

    }


    work_value = work_mapping[work_type]

    smoking_value = smoking_mapping[smoking_status]

    residence_value = (
        1 if residence_type == "Urban"
        else 0
    )


    # --------------------------
    # CREATE MODEL INPUT
    # --------------------------

    input_data = pd.DataFrame(

        [[

            gender_value,

            age,

            hypertension_value,

            heart_value,

            married_value,

            work_value,

            residence_value,

            avg_glucose,

            bmi,

            smoking_value

        ]],

        columns=[

            "gender",

            "age",

            "hypertension",

            "heart_disease",

            "ever_married",

            "work_type",

            "Residence_type",

            "avg_glucose_level",

            "bmi",

            "smoking_status"

        ]

    )


    # --------------------------
    # MODEL PREDICTION
    # --------------------------

    prediction = model.predict(
        input_data
    )[0]


    probability = model.predict_proba(
        input_data
    )[0][1]


    risk_level = get_risk_level(
        probability
    )


    st.divider()


    # --------------------------
    # RESULT MESSAGE
    # --------------------------

    if prediction == 1:

        st.error(
            "⚠️ Higher Stroke Risk Detected"
        )

    else:

        st.success(
            "✅ Lower Stroke Risk Detected"
        )


    # --------------------------
    # METRICS
    # --------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "BMI",
        bmi
    )

    col2.metric(
        "Risk Probability",
        f"{probability*100:.2f}%"
    )

    col3.metric(
        "Risk Level",
        risk_level
    )


    # --------------------------
    # RISK METER
    # --------------------------

    st.subheader(
        "📊 Risk Meter"
    )

    gauge = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=probability * 100,

            title={

                "text":
                "Stroke Risk %"

            },

            gauge={

                "axis": {

                    "range": [0, 100]

                }

            }

        )

    )

    st.plotly_chart(
        gauge,
        use_container_width=True
    )


    # --------------------------
    # RECOMMENDATIONS
    # --------------------------

    st.subheader(
        "💡 Health Recommendations"
    )

    for item in get_recommendations(
        probability
    ):

        st.write(
            "✔️ " + item
        )


    # --------------------------
    # SAVE RESULT TEMPORARILY
    # --------------------------

    st.session_state["latest_result"] = {

        "name": patient_name,

        "age": age,

        "glucose": avg_glucose,

        "bmi": bmi,

        "probability": probability,

        "risk": risk_level

    }
    # ==========================
# SAVE PATIENT RECORD
# ==========================

if "latest_result" in st.session_state:


    st.divider()


    st.markdown(
        """
        <div class="section-title">
        💾 Save Patient Record
        </div>
        """,
        unsafe_allow_html=True
    )


    if st.button(
        "Save Record"
    ):


        data = st.session_state["latest_result"]


        save_prediction(

            data["age"],

            data["glucose"],

            data["bmi"],

            data["risk"]

        )


        st.success(
            "Patient record saved successfully."
        )


# ==========================
# MEDICAL ASSISTANT
# ==========================

st.divider()

st.markdown(
    """
    <div class="section-title">
    🤖 Medical Assistant
    </div>
    """,
    unsafe_allow_html=True
)

st.info(
    "Ask questions about stroke symptoms, prevention, recovery, or risk factors."
)


user_question = st.text_input(
    "Ask a health question"
)


if st.button("Get Answer"):

    if user_question.strip():

        try:

            from rag_mcp import ask_medical_agent

            st.info("🔄 Connecting to Medical AI...")


            with st.spinner("🧠 Generating response..."):

                response = ask_medical_agent(user_question)


            st.success("✅ Answer Generated")


            if response:

                st.write(response)

            else:

                st.warning(
                    "AI returned an empty response."
                )


        except ImportError:

            st.error(
                "❌ Cannot find ask_medical_agent in rag_mcp.py"
            )


        except Exception as e:

            st.error(
                "❌ Medical Assistant Error"
            )

            st.code(
                str(e)
            )


    else:

        st.warning(
            "Please enter a question first."
        )
# ==========================
# ANALYTICS DASHBOARD
# ==========================

st.divider()

st.markdown(
    """
    <div class="section-title">
    📊 Analytics Dashboard
    </div>
    """,
    unsafe_allow_html=True
)


records = get_history()


if len(records) > 0:


    analytics_df = pd.DataFrame(

        records,

        columns=[

            "Age",

            "Glucose",

            "BMI",

            "Risk"

        ]

    )


    col1, col2, col3 = st.columns(3)


    col1.metric(
        "Total Records",
        len(analytics_df)
    )


    high_risk = len(

        analytics_df[
            analytics_df["Risk"]
            ==
            "HIGH RISK"
        ]

    )


    col2.metric(
        "High Risk Cases",
        high_risk
    )


    col3.metric(
        "Average Age",
        round(
            analytics_df["Age"].mean(),
            1
        )
    )


    st.subheader(
        "Risk Distribution"
    )


    st.bar_chart(
        analytics_df["Risk"].value_counts()
    )


else:


    st.info(
        "Analytics will appear after saving patient records."
    )


# ==========================
# PDF REPORT
# ==========================

st.divider()

st.markdown(
    """
    <div class="section-title">
    📄 Download Medical Report
    </div>
    """,
    unsafe_allow_html=True
)


if "latest_result" in st.session_state:


    if st.button(
        "Generate PDF Report"
    ):


        report_data = st.session_state["latest_result"]


        try:


            pdf_file = generate_pdf(
                report_data
            )


            with open(
                pdf_file,
                "rb"
            ) as file:


                st.download_button(

                    label="⬇️ Download PDF",

                    data=file,

                    file_name="Brain_Stroke_Diagnosis_Report.pdf",

                    mime="application/pdf"

                )


        except Exception as e:


            st.warning(
                "PDF report generation unavailable."
            )

            st.write(e)


# ==========================
# ABOUT PROJECT
# ==========================

st.divider()

st.markdown(
    """
    <div class="section-title">
    ℹ️ About Brain Stroke Diagnosis
    </div>
    """,
    unsafe_allow_html=True
)


st.write(
"""
Brain Stroke Diagnosis is a healthcare application designed to help users understand potential stroke risk factors through predictive analysis.

### Features

✔ Stroke risk assessment

✔ Risk probability estimation

✔ Health recommendations

✔ Medical question assistant

✔ Prediction history tracking

✔ Analytics dashboard

✔ PDF report generation

### Important Notice

This tool is intended for educational and informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.
"""
)


# ==========================
# FOOTER
# ==========================

st.divider()

st.markdown(
"""
<center>

<b>Brain Stroke Diagnosis</b>

<br>

Early Detection Saves Lives

</center>
""",
unsafe_allow_html=True
)