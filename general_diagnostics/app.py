try:
    from RAG.rag_service import get_rag_service
except ImportError:
    get_rag_service = None
import streamlit as st
import os
import re
import joblib
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

# Load configuration before importing agents and settings, because those
# modules read API keys while they are imported or instantiated.
for env_file in (
    PROJECT_DIR / ".env",
    BASE_DIR / ".env",
    PROJECT_DIR / "clinical_chatbot" / ".env",
):
    if env_file.is_file():
        load_dotenv(env_file, override=False)

try:
    from RAG.rag_service import get_rag_service
except ImportError:
    get_rag_service = None
from utils.file_handler import FileHandler, PatientManager
from agents.diagnostic_agent import DiagnosticAgent
from agents.readmission_predictor import ReadmissionPredictor
from agents.safety_agent import SafetyAgent
from agents.medication_agent import MedicationAgent
from config.settings import settings
try:
    from utils.pdf_generator import PDFReportGenerator
except ImportError:
    PDFReportGenerator = None
import pandas as pd
from tempfile import NamedTemporaryFile
import sqlite3
import inspect
from datetime import datetime
from utils.data_parser import MedicalDataParser
from heart import HeartDiseaseAnalytics
from breast_cancer import BreastCancerClassifier

CLINICAL_NOTE_MODEL_PATH = BASE_DIR / "clinical_note_model.joblib"

# Breast cancer ultrasound classifier (MobileNetV2 / BUSI), trained
# separately by train_model.py.
BREAST_CANCER_MODEL_PATH = BASE_DIR / "model" / "breast_cancer_model.keras"
BREAST_CANCER_CLASS_INDEX_PATH = BASE_DIR / "model" / "class_indices.json"


@st.cache_resource
def load_clinical_note_model():
    if not os.path.exists(CLINICAL_NOTE_MODEL_PATH):
        raise FileNotFoundError(
            "clinical_note_model.joblib was not found. "
            "Copy it into the Agentic-Diagnostic folder beside app.py."
        )

    model_information = joblib.load(
        CLINICAL_NOTE_MODEL_PATH
    )

    return model_information["model"]


@st.cache_resource
def load_breast_cancer_classifier():
    return BreastCancerClassifier(
        model_path=BREAST_CANCER_MODEL_PATH,
        class_index_path=BREAST_CANCER_CLASS_INDEX_PATH,
    )


def _get_existing_file_record(file_handler, original_filename):
    """Look up a previously-saved file's DB record by original filename.

    Tries the current patient first, then falls back to any patient_id
    (in case the file was saved under a stale/incorrect patient_id by an
    earlier buggy run, before patient_id was passed correctly) so
    re-uploads of already-processed files still resolve instead of
    silently failing."""
    record = file_handler.get_file_info_by_name(
        original_filename, st.session_state.patient_id
    )
    if record:
        return record

    try:
        conn = sqlite3.connect(file_handler.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM uploaded_files WHERE original_filename = ? "
            "ORDER BY id DESC LIMIT 1",
            (original_filename,)
        )
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception:
        return None


disease_recommendations = {
    "acne": [
        "Keep the affected skin clean and avoid squeezing pimples.",
        "Use gentle, non-comedogenic skin products.",
        "Seek medical advice if the condition becomes severe.",
    ],
    "arthritis": [
        "Avoid activities that increase joint pain.",
        "Use gentle movement and stretching when comfortable.",
        "Seek medical assessment if the pain continues.",
    ],
    "bronchial asthma": [
        "Avoid smoke, dust, and other known triggers.",
        "Follow the treatment plan given by a healthcare professional.",
        "Seek emergency care if breathing becomes very difficult.",
    ],
    "cervical spondylosis": [
        "Avoid heavy lifting and sudden neck movements.",
        "Maintain a comfortable posture.",
        "Seek medical advice if pain or weakness continues.",
    ],
    "chicken pox": [
        "Avoid scratching the affected skin.",
        "Drink enough fluids and rest.",
        "Seek medical advice if symptoms become severe.",
    ],
    "common cold": [
        "Rest and drink plenty of fluids.",
        "Monitor fever and breathing symptoms.",
        "Seek medical advice if symptoms become severe or continue.",
    ],
    "dengue": [
        "Seek prompt medical assessment.",
        "Drink fluids as advised by a healthcare professional.",
        "Seek emergency care for bleeding, severe weakness, or abdominal pain.",
    ],
    "dimorphic hemorrhoids": [
        "Drink enough water.",
        "Include fibre-rich foods in the diet.",
        "Seek medical advice if there is pain or bleeding.",
    ],
    "dimorphic haemorrhoids": [
        "Drink enough water.",
        "Include fibre-rich foods in the diet.",
        "Seek medical advice if there is pain or bleeding.",
    ],
    "fungal infection": [
        "Keep the affected area clean and dry.",
        "Avoid sharing towels or personal items.",
        "Seek medical advice if the infection spreads.",
    ],
    "hypertension": [
        "Have blood pressure checked regularly.",
        "Reduce excessive salt intake.",
        "Seek urgent care for severe chest pain, weakness, or confusion.",
    ],
    "impetigo": [
        "Keep the affected skin clean.",
        "Avoid touching or scratching the sores.",
        "Seek medical advice because treatment may be required.",
    ],
    "jaundice": [
        "Seek medical assessment to identify the cause.",
        "Drink fluids unless advised otherwise.",
        "Seek urgent care if there is confusion, severe pain, or vomiting.",
    ],
    "malaria": [
        "Seek medical testing and treatment promptly.",
        "Drink fluids and monitor fever.",
        "Seek urgent care if there is confusion or difficulty breathing.",
    ],
    "migraine": [
        "Rest in a quiet and dark room.",
        "Drink enough fluids.",
        "Seek urgent care for a sudden severe headache, weakness, or confusion.",
    ],
    "pneumonia": [
        "Rest and drink plenty of fluids.",
        "Follow the treatment advised by a healthcare professional.",
        "Seek emergency care if breathing becomes difficult.",
    ],
    "psoriasis": [
        "Keep the skin moisturised.",
        "Avoid scratching the affected areas.",
        "Seek medical advice for suitable treatment.",
    ],
    "typhoid": [
        "Seek medical assessment and testing.",
        "Drink safe fluids to prevent dehydration.",
        "Seek urgent care if weakness, vomiting, or abdominal pain becomes severe.",
    ],
    "varicose veins": [
        "Avoid standing for very long periods.",
        "Elevate the legs while resting.",
        "Seek medical advice if there is swelling or severe pain.",
    ],
    "allergy": [
        "Avoid the suspected allergy trigger.",
        "Seek medical advice if symptoms continue.",
        "Seek emergency care for facial swelling or breathing difficulty.",
    ],
    "diabetes": [
        "Monitor blood sugar as advised.",
        "Follow a healthy eating and activity plan.",
        "Seek urgent care for confusion, fainting, or severe weakness.",
    ],
    "drug reaction": [
        "Contact a healthcare professional promptly.",
        "Do not take another dose without professional advice.",
        "Seek emergency care for swelling, breathing difficulty, or fainting.",
    ],
    "gastroesophageal reflux disease": [
        "Avoid lying down immediately after eating.",
        "Avoid foods that clearly worsen symptoms.",
        "Seek medical advice if symptoms continue.",
    ],
    "peptic ulcer disease": [
        "Avoid foods or medicines that worsen stomach pain.",
        "Seek medical advice for proper assessment.",
        "Seek urgent care for vomiting blood or black stools.",
    ],
    "urinary tract infection": [
        "Drink enough water unless medically restricted.",
        "Seek medical testing and treatment.",
        "Seek urgent care for high fever, vomiting, or severe back pain.",
    ],
}


default_recommendations = [
    "Monitor the symptoms carefully.",
    "Seek advice from a qualified healthcare professional.",
    "Seek urgent medical care if symptoms become severe.",
]


COMMON_SYMPTOMS = [
    "Fever",
    "Cough",
    "Headache",
    "Fatigue",
    "Nausea",
    "Vomiting",
    "Shortness of Breath",
    "Chest Pain",
    "Abdominal Pain",
    "Dizziness",
    "Sore Throat",
    "Body Ache",
    "Diarrhea",
    "Skin Rash",
    "Joint Pain",
    "Loss of Appetite",
    "Chills",
    "Runny Nose",
    "Back Pain",
    "Swelling",
]


st.set_page_config(
    page_title="Smart Clinic Diagnostic Agent",
    page_icon="🏥",
    layout="wide"
)


if "clinical_case_input" not in st.session_state:
    st.session_state.clinical_case_input = ""

if "chief_complaint" not in st.session_state:
    st.session_state.chief_complaint = ""

if "chief_complaint_input" not in st.session_state:
    st.session_state.chief_complaint_input = ""

if "symptoms" not in st.session_state:
    st.session_state.symptoms = ""

if "symptoms_input" not in st.session_state:
    st.session_state.symptoms_input = ""

if "saved_other_symptoms" not in st.session_state:
    st.session_state.saved_other_symptoms = ""

if "selected_common_symptoms" not in st.session_state:
    st.session_state.selected_common_symptoms = []

if "clinical_notes_input" not in st.session_state:
    st.session_state.clinical_notes_input = ""

if "lab_data" not in st.session_state:
    st.session_state.lab_data = ""

if "image_findings" not in st.session_state:
    st.session_state.image_findings = ""

if "medical_analysis_image_findings_input" not in st.session_state:
    st.session_state.medical_analysis_image_findings_input = ""

if "saved_image_findings" not in st.session_state:
    st.session_state.saved_image_findings = ""

if "document_findings" not in st.session_state:
    st.session_state.document_findings = ""

if "medical_analysis_document_findings_input" not in st.session_state:
    st.session_state.medical_analysis_document_findings_input = ""

if "saved_document_findings" not in st.session_state:
    st.session_state.saved_document_findings = ""

if "document_types" not in st.session_state:
    st.session_state.document_types = {}

if "uploaded_files_info" not in st.session_state:
    st.session_state.uploaded_files_info = []

if "uploader_version" not in st.session_state:
    st.session_state.uploader_version = 0

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

if "readmission_result" not in st.session_state:
    st.session_state.readmission_result = None

if "medication_result" not in st.session_state:
    st.session_state.medication_result = None

if "saved_readmission_inputs" not in st.session_state:
    st.session_state.saved_readmission_inputs = {}

if "patient_id" not in st.session_state:
    st.session_state.patient_id = ""

if "patient_name" not in st.session_state:
    st.session_state.patient_name = ""

if "patient_age" not in st.session_state:
    st.session_state.patient_age = 30

if "patient_gender" not in st.session_state:
    st.session_state.patient_gender = "Male"

if "patient_phone" not in st.session_state:
    st.session_state.patient_phone = ""

if "patient_address" not in st.session_state:
    st.session_state.patient_address = ""

if "patient_cnic" not in st.session_state:
    st.session_state.patient_cnic = ""

if "patient_email" not in st.session_state:
    st.session_state.patient_email = ""

if "emergency_contact_phone" not in st.session_state:
    st.session_state.emergency_contact_phone = ""

if "emergency_contact_relationship" not in st.session_state:
    st.session_state.emergency_contact_relationship = ""

if "member_type" not in st.session_state:
    st.session_state.member_type = "Regular"

if "patient_date_of_birth" not in st.session_state:
    st.session_state.patient_date_of_birth = None

if "payment_method" not in st.session_state:
    st.session_state.payment_method = "Cash"

if "selected_date" not in st.session_state:
    st.session_state.selected_date = None

if "last_visit_input" not in st.session_state:
    st.session_state.last_visit_input = ""

if "last_visit" not in st.session_state:
    st.session_state.last_visit = ""

if "saved_last_visit" not in st.session_state:
    st.session_state.saved_last_visit = ""

if "medical_history" not in st.session_state:
    st.session_state.medical_history = ""

if "chronic_conditions" not in st.session_state:
    st.session_state.chronic_conditions = ""

if "past_surgeries" not in st.session_state:
    st.session_state.past_surgeries = ""

if "known_allergies" not in st.session_state:
    st.session_state.known_allergies = ""

if "current_medications" not in st.session_state:
    st.session_state.current_medications = ""

if "family_history" not in st.session_state:
    st.session_state.family_history = ""

if "immunization_history" not in st.session_state:
    st.session_state.immunization_history = ""

if "show_results" not in st.session_state:
    st.session_state.show_results = False

if "clinical_classification_result" not in st.session_state:
    st.session_state.clinical_classification_result = None

if "heart_disease_result" not in st.session_state:
    st.session_state.heart_disease_result = None

# Breast cancer ultrasound classification state.
if "breast_cancer_result" not in st.session_state:
    st.session_state.breast_cancer_result = None

if "last_breast_us_signature" not in st.session_state:
    st.session_state.last_breast_us_signature = None

if "breast_cancer_image_path" not in st.session_state:
    st.session_state.breast_cancer_image_path = None

if "breast_cancer_note_added_signature" not in st.session_state:
    st.session_state.breast_cancer_note_added_signature = None

if "previous_admissions" not in st.session_state:
    st.session_state.previous_admissions = 0

if "previous_admissions_input" not in st.session_state:
    st.session_state.previous_admissions_input = 0

if "length_of_stay" not in st.session_state:
    st.session_state.length_of_stay = 0

if "length_of_stay_input" not in st.session_state:
    st.session_state.length_of_stay_input = 0

if "emergency_visits" not in st.session_state:
    st.session_state.emergency_visits = 0

if "emergency_visits_input" not in st.session_state:
    st.session_state.emergency_visits_input = 0

if "number_of_medications" not in st.session_state:
    st.session_state.number_of_medications = 0

if "number_of_medications_input" not in st.session_state:
    st.session_state.number_of_medications_input = 0

if "discharge_disposition" not in st.session_state:
    st.session_state.discharge_disposition = "Home / Self Care"

if "discharge_disposition_input" not in st.session_state:
    st.session_state.discharge_disposition_input = "Home / Self Care"


if "blood_pressure" not in st.session_state:
    st.session_state.blood_pressure = ""

if "heart_rate" not in st.session_state:
    st.session_state.heart_rate = 0

if "respiratory_rate" not in st.session_state:
    st.session_state.respiratory_rate = 0

if "patient_weight" not in st.session_state:
    st.session_state.patient_weight = 0.0

if "patient_height" not in st.session_state:
    st.session_state.patient_height = 0.0

if "blood_glucose" not in st.session_state:
    st.session_state.blood_glucose = 0.0

if "body_temperature" not in st.session_state:
    st.session_state.body_temperature = 0.0

if "temperature_scale" not in st.session_state:
    st.session_state.temperature_scale = "°F"

if "oxygen_saturation" not in st.session_state:
    st.session_state.oxygen_saturation = 0

if "waist_circumference" not in st.session_state:
    st.session_state.waist_circumference = 0.0

if "blood_group" not in st.session_state:
    st.session_state.blood_group = "A+"

if "saved_medical_history" not in st.session_state:
    st.session_state.saved_medical_history = None

if "saved_chief_complaint" not in st.session_state:
    st.session_state.saved_chief_complaint = ""

if "saved_symptoms" not in st.session_state:
    st.session_state.saved_symptoms = ""

if "saved_selected_common_symptoms" not in st.session_state:
    st.session_state.saved_selected_common_symptoms = []

if "saved_clinical_notes" not in st.session_state:
    st.session_state.saved_clinical_notes = ""

if "saved_clinical_case" not in st.session_state:
    st.session_state.saved_clinical_case = ""


WIDGET_TO_CANONICAL = {
    "chief_complaint_input": "chief_complaint",
    "symptoms_input": "symptoms",
    "medical_analysis_image_findings_input": "image_findings",
    "medical_analysis_document_findings_input": "document_findings",

    "chronic_conditions_input": "chronic_conditions",
    "past_surgeries_input": "past_surgeries",
    "known_allergies_input": "known_allergies",
    "current_medications_input": "current_medications",
    "family_history_input": "family_history",
    "immunization_history_input": "immunization_history",
    "medical_history_input": "medical_history",
    "patient_id_input": "patient_id",
    "patient_name_input": "patient_name",
    "patient_age_input": "patient_age",
    "patient_gender_input": "patient_gender",
    "patient_phone_input": "patient_phone",
    "patient_address_input": "patient_address",
    "patient_cnic_input": "patient_cnic",
    "patient_email_input": "patient_email",
    "emergency_contact_phone_input": "emergency_contact_phone",
    "emergency_contact_relationship_input": "emergency_contact_relationship",
    "member_type_input": "member_type",
    "patient_date_of_birth_input": "patient_date_of_birth",
    "payment_method_input": "payment_method",
    "selected_date_input": "selected_date",
    "last_visit_input": "last_visit",
    "blood_pressure_input": "blood_pressure",
    "heart_rate_input": "heart_rate",
    "respiratory_rate_input": "respiratory_rate",
    "patient_weight_input": "patient_weight",
    "patient_height_feet_input": "patient_height_feet",
    "patient_height_inches_input": "patient_height_inches",
    "blood_glucose_input": "blood_glucose",
    "body_temperature_input": "body_temperature",
    "temperature_scale_input": "temperature_scale",
    "oxygen_saturation_input": "oxygen_saturation",
    "waist_circumference_input": "waist_circumference",
    "blood_group_input": "blood_group",

    "previous_admissions_input": "previous_admissions",
    "length_of_stay_input": "length_of_stay",
    "emergency_visits_input": "emergency_visits",
    "number_of_medications_input": "number_of_medications",
    "discharge_disposition_input": "discharge_disposition",
}


MEDICAL_HISTORY_FIELD_KEYS = [
    ("chronic_conditions", "Health Issues"),
    ("past_surgeries", "Past Surgeries"),
    ("known_allergies", "Known Allergies"),
    ("current_medications", "Current Medications"),
    ("family_history", "Family History"),
    ("immunization_history", "Immunization History"),
    ("medical_history", "Additional Notes"),
]


def sync_clinical_widget_state():
    """Keep canonical session keys aligned with widget keys across tab switches."""
    for widget_key, canonical_key in WIDGET_TO_CANONICAL.items():
        if widget_key in st.session_state:
            st.session_state[canonical_key] = st.session_state[widget_key]


# Run on every Streamlit rerun at module level — BEFORE any tab is rendered.
# At the start of each run all widget session_state keys from the previous run
# are still present, so this captures values from unrendered tabs before
# Streamlit cleans them up at the end of the run.  This is the fail-safe that
# covers cases where on_change callbacks did not fire (e.g. user clicked the
# sidebar while the cursor was still inside a number_input).
sync_clinical_widget_state()

def _on_symptom_checkbox_change(symptom_index):
    """Write the toggle straight into the canonical list the instant it happens.

    Streamlit drops a checkbox's own `symptom_checkbox_*` session_state key
    once that checkbox isn't rendered on a run (e.g. the user switched to a
    different tab). Recomputing `selected_common_symptoms` only *after* the
    checkboxes render depends on those keys still being there, which is
    exactly what tab switches break. An on_change callback runs immediately
    when the checkbox is clicked -- before any of that cleanup can happen --
    so this is the reliable place to update the canonical, non-widget
    session_state list that actually survives tab switches.
    """
    symptom_name = COMMON_SYMPTOMS[symptom_index]
    checkbox_key = f"symptom_checkbox_{symptom_index}"
    is_checked = st.session_state.get(checkbox_key, False)

    current = list(st.session_state.get("selected_common_symptoms", []))
    if is_checked and symptom_name not in current:
        current.append(symptom_name)
    elif not is_checked and symptom_name in current:
        current.remove(symptom_name)
    st.session_state.selected_common_symptoms = current


# ---------------------------------------------------------------------------
# Readmission input on_change callbacks
# ---------------------------------------------------------------------------
# Streamlit deletes a widget's session_state key the moment that widget is
# not rendered (e.g. because the user switched to a different tab).  The
# `value=` fallback we pass to each widget only sets the *initial* value when
# the key is absent -- it is ignored once the key already exists.  This means
# the only reliable window to copy a widget's live value into a canonical,
# tab-switch-safe key is an on_change callback, which fires immediately after
# the user changes the field -- before any cleanup or re-render can occur.
# ---------------------------------------------------------------------------

def _save_previous_admissions():
    st.session_state["previous_admissions"] = st.session_state.get("previous_admissions_input", 0)


def _save_length_of_stay():
    st.session_state["length_of_stay"] = st.session_state.get("length_of_stay_input", 0)


def _save_emergency_visits():
    st.session_state["emergency_visits"] = st.session_state.get("emergency_visits_input", 0)


def _save_number_of_medications():
    st.session_state["number_of_medications"] = st.session_state.get("number_of_medications_input", 0)


def _save_discharge_disposition():
    st.session_state["discharge_disposition"] = st.session_state.get("discharge_disposition_input", "Home / Self Care")



def _save_discharge_information():
    st.session_state["discharge_information"] = st.session_state.get("discharge_information_input", "")


def persist_all_widget_state():
    """Commit all widget-bound values to session state before save or tab navigation."""
    sync_clinical_widget_state()
    if "symptoms_input" in st.session_state:
        st.session_state.symptoms = st.session_state.symptoms_input
        if st.session_state.symptoms_input:
            st.session_state.saved_other_symptoms = st.session_state.symptoms_input
    if "medical_analysis_image_findings_input" in st.session_state:
        st.session_state.image_findings = st.session_state.medical_analysis_image_findings_input
        if st.session_state.medical_analysis_image_findings_input:
            st.session_state.saved_image_findings = st.session_state.medical_analysis_image_findings_input
    if "medical_analysis_document_findings_input" in st.session_state:
        st.session_state.document_findings = st.session_state.medical_analysis_document_findings_input
        if st.session_state.medical_analysis_document_findings_input:
            st.session_state.saved_document_findings = st.session_state.medical_analysis_document_findings_input
    if "chief_complaint_input" in st.session_state:
        st.session_state.chief_complaint = st.session_state.chief_complaint_input
    if "last_visit_input" in st.session_state:
        st.session_state.last_visit = st.session_state.last_visit_input
        if st.session_state.last_visit_input:
            st.session_state.saved_last_visit = st.session_state.last_visit_input
    # Only recompute from the checkboxes when they're actually present this
    # run (i.e. the Clinical Examination tab is the one currently being
    # rendered/just rendered). Streamlit clears symptom_checkbox_* keys once
    # that tab isn't rendered, so recomputing unconditionally here would wipe
    # out previously selected symptoms whenever a save button on *any other*
    # tab calls this function.
    if any(f"symptom_checkbox_{index}" in st.session_state for index in range(len(COMMON_SYMPTOMS))):
        st.session_state.selected_common_symptoms = [
            name
            for index, name in enumerate(COMMON_SYMPTOMS)
            if st.session_state.get(f"symptom_checkbox_{index}", False)
        ]


def _build_medical_history_summary_from_values(values):
    formatted_sections = [
        f"{label}: {values.get(field_key, '').strip()}"
        for field_key, label in MEDICAL_HISTORY_FIELD_KEYS
        if values.get(field_key, "").strip()
    ]
    return "\n".join(formatted_sections)


def save_medical_history_snapshot():
    sync_clinical_widget_state()
    st.session_state.saved_medical_history = {
        field_key: st.session_state.get(field_key, "")
        for field_key, _ in MEDICAL_HISTORY_FIELD_KEYS
    }
    refresh_integrated_clinical_summary()


def build_vitals_summary_for_clinical_notes():
    """Return the currently entered vitals in a compact, readable form."""
    return (
        f"Blood Pressure: {st.session_state.get('blood_pressure', '') or 'Not provided'} mmHg; "
        f"Heart Rate: {st.session_state.get('heart_rate', 0)} bpm; "
        f"Respiratory Rate: {st.session_state.get('respiratory_rate', 0)} breaths/min; "
        f"Weight: {st.session_state.get('patient_weight', 0.0)} kg; "
        f"Height: {st.session_state.get('patient_height_feet', 0)} ft "
        f"{st.session_state.get('patient_height_inches', 0)} in; "
        f"Blood Glucose: {st.session_state.get('blood_glucose', 0.0)} mg/dL; "
        f"Temperature: {st.session_state.get('body_temperature', 0.0)} "
        f"{st.session_state.get('temperature_scale', '°F')}; "
        f"Oxygen Saturation: {st.session_state.get('oxygen_saturation', 0)}%; "
        f"Waist Circumference: {st.session_state.get('waist_circumference', 0.0)} cm; "
        f"Blood Group: {st.session_state.get('blood_group', '') or 'Not provided'}."
    )


def build_integrated_clinical_summary_and_notes():
    """Build a source-grounded case description and clinical notes from the
    clinical information already entered in the application.

    This is an additive feature only: it does not alter any of the existing
    patient, history, examination, document, readmission, diagnostic, or
    heart-disease processing. It only provides fallback text when the
    Clinical Case Description and Clinical Notes fields are empty.
    """
    persist_all_widget_state()

    complaint = get_effective_chief_complaint()
    history = build_effective_medical_history_summary()
    symptoms = build_effective_symptoms_summary()
    vitals = build_vitals_summary_for_clinical_notes()
    documents = get_effective_document_findings()
    images = get_effective_image_findings()

    source_sections = []
    if complaint:
        source_sections.append(f"Chief Complaint: {complaint}")
    if history:
        source_sections.append(f"Medical History: {history}")
    if symptoms:
        source_sections.append(f"Symptoms: {symptoms}")
    if vitals:
        source_sections.append(f"Vitals: {vitals}")
    if images:
        source_sections.append(f"Imaging Findings: {images}")
    if documents:
        source_sections.append(f"Documents and Clinical Data: {documents}")

    if not source_sections:
        return "", ""

    case_description = " ".join(
        f"{section}." for section in source_sections
    )

    note_parts = [
        "Clinical notes summary:"
    ]

    if complaint:
        note_parts.append(
            f"The patient presented with the chief complaint of {complaint}."
        )
    if history:
        note_parts.append(
            f"Relevant medical history includes: {history}."
        )
    if symptoms:
        note_parts.append(
            f"Reported symptoms include: {symptoms}."
        )
    if vitals:
        note_parts.append(
            f"Current recorded vital signs are: {vitals}"
        )
    if images:
        note_parts.append(
            f"Imaging analysis reports: {images}."
        )
    if documents:
        note_parts.append(
            f"Available clinical documents and related data report: {documents}."
        )

    note_parts.append(
        "This summary is generated only from information entered or extracted "
        "within the application and does not add unsupported clinical findings."
    )

    return case_description, " ".join(note_parts)


def refresh_integrated_clinical_summary():
    """Prepare the combined summary for the Clinical Description and Notes fields."""
    generated_case, generated_notes = build_integrated_clinical_summary_and_notes()
    st.session_state.pending_generated_clinical_case = generated_case
    st.session_state.pending_generated_clinical_notes = generated_notes


def save_chief_complaint_snapshot():
    sync_clinical_widget_state()
    st.session_state.saved_chief_complaint = st.session_state.get("chief_complaint", "")
    refresh_integrated_clinical_summary()


def save_symptoms_snapshot():
    sync_clinical_widget_state()
    st.session_state.saved_other_symptoms = (
        st.session_state.get("symptoms_input")
        or st.session_state.get("symptoms", "")
    )
    st.session_state.saved_symptoms = build_symptoms_summary()
    st.session_state.saved_selected_common_symptoms = list(
        st.session_state.get("selected_common_symptoms", [])
    )
    refresh_integrated_clinical_summary()


def save_medical_analysis_snapshot():
    sync_clinical_widget_state()
    st.session_state.saved_image_findings = (
        st.session_state.get("medical_analysis_image_findings_input")
        or st.session_state.get("image_findings", "")
    )
    st.session_state.saved_document_findings = (
        st.session_state.get("medical_analysis_document_findings_input")
        or st.session_state.get("document_findings", "")
    )
    refresh_integrated_clinical_summary()


def get_effective_other_symptoms():
    saved = (st.session_state.get("saved_other_symptoms") or "").strip()
    if saved:
        return saved
    return (
        st.session_state.get("symptoms")
        or st.session_state.get("symptoms_input")
        or ""
    ).strip()


def get_effective_image_findings():
    saved = (st.session_state.get("saved_image_findings") or "").strip()
    if saved:
        return saved
    return (
        st.session_state.get("image_findings")
        or st.session_state.get("medical_analysis_image_findings_input")
        or ""
    ).strip()


def get_effective_document_findings():
    saved = (st.session_state.get("saved_document_findings") or "").strip()
    if saved:
        return saved
    return (
        st.session_state.get("document_findings")
        or st.session_state.get("medical_analysis_document_findings_input")
        or ""
    ).strip()



def save_clinical_notes_snapshot():
    sync_clinical_widget_state()
    st.session_state.saved_clinical_notes = st.session_state.get("clinical_notes_input", "")
    st.session_state.saved_clinical_case = st.session_state.get("clinical_case_input", "")


def save_patient_info_snapshot():
    sync_clinical_widget_state()
    st.session_state.saved_last_visit = (
        st.session_state.get("last_visit_input")
        or st.session_state.get("last_visit", "")
    )


def get_effective_last_visit():
    saved = (st.session_state.get("saved_last_visit") or "").strip()
    if saved:
        return saved
    return (
        st.session_state.get("last_visit")
        or st.session_state.get("last_visit_input")
        or ""
    ).strip()


def get_effective_chief_complaint():
    persist_all_widget_state()
    saved = (st.session_state.get("saved_chief_complaint") or "").strip()
    if saved:
        return saved
    return (
        st.session_state.get("chief_complaint")
        or st.session_state.get("chief_complaint_input")
        or ""
    ).strip()


def build_effective_medical_history_summary():
    saved = st.session_state.get("saved_medical_history")
    if isinstance(saved, dict) and any(str(v).strip() for v in saved.values()):
        return _build_medical_history_summary_from_values(saved)
    return build_medical_history_summary()


def build_effective_symptoms_summary():
    saved = (st.session_state.get("saved_symptoms") or "").strip()
    if saved:
        return saved
    return build_symptoms_summary()


def get_effective_clinical_notes():
    saved = (st.session_state.get("saved_clinical_notes") or "").strip()
    if saved:
        return saved
    return (st.session_state.get("clinical_notes_input") or "").strip()


def get_effective_clinical_case():
    saved = (st.session_state.get("saved_clinical_case") or "").strip()
    if saved:
        return saved
    return (st.session_state.get("clinical_case_input") or "").strip()


def get_or_generate_clinical_case_and_notes():
    """Return existing case/notes, generating only the missing fields.

    Manual Clinical Case Description / Clinical Notes always take priority.
    The integrated summary is used only when the corresponding field is empty.
    """
    case_description = get_effective_clinical_case()
    clinical_notes = get_effective_clinical_notes()

    if not case_description or not clinical_notes:
        generated_case, generated_notes = build_integrated_clinical_summary_and_notes()
        if not case_description:
            case_description = generated_case
        if not clinical_notes:
            clinical_notes = generated_notes

    return case_description, clinical_notes


def build_clinical_intake_dict():
    persist_all_widget_state()
    clinical_case_description, clinical_notes = get_or_generate_clinical_case_and_notes()
    return {
        "chief_complaint": (
            st.session_state.get("chief_complaint")
            or st.session_state.get("chief_complaint_input")
            or ""
        ),
        "clinical_case_description": clinical_case_description,
        "clinical_notes": clinical_notes,
        "medical_history": build_effective_medical_history_summary(),
    }


def build_report_patient_info():
    """Collect all patient and clinical data for on-screen report and PDF export."""
    persist_all_widget_state()

    dob = st.session_state.get("patient_date_of_birth")
    if dob and hasattr(dob, "strftime"):
        dob_str = dob.strftime("%d-%m-%Y")
    else:
        dob_str = str(dob) if dob else ""

    selected = st.session_state.get("selected_date")
    if selected and hasattr(selected, "strftime"):
        selected_str = selected.strftime("%d-%m-%Y")
    else:
        selected_str = str(selected) if selected else ""

    return {
        "patient_id": st.session_state.get("patient_id", ""),
        "patient_name": st.session_state.get("patient_name", ""),
        "patient_age": st.session_state.get("patient_age", ""),
        "patient_gender": st.session_state.get("patient_gender", ""),
        "phone": st.session_state.get("patient_phone", ""),
        "address": st.session_state.get("patient_address", ""),
        "email": st.session_state.get("patient_email", ""),
        "cnic": st.session_state.get("patient_cnic", ""),
        "emergency_contact_phone": st.session_state.get("emergency_contact_phone", ""),
        "emergency_contact_relationship": st.session_state.get(
            "emergency_contact_relationship", ""
        ),
        "member_type": st.session_state.get("member_type", ""),
        "date_of_birth": dob_str,
        "payment_method": st.session_state.get("payment_method", ""),
        "selected_date": selected_str,
        "last_visit": get_effective_last_visit(),
        "chief_complaint": get_effective_chief_complaint(),
        "symptoms": build_effective_symptoms_summary(),
        "medical_history": build_effective_medical_history_summary(),
        "clinical_intake": build_clinical_intake_dict(),
        "clinical_examination": {
            "other_symptoms": get_effective_other_symptoms(),
            "image_findings": get_effective_image_findings(),
            "document_findings": get_effective_document_findings(),
        },
        "vitals": {
            "blood_pressure": st.session_state.get("blood_pressure", ""),
            "heart_rate": st.session_state.get("heart_rate", 0),
            "respiratory_rate": st.session_state.get("respiratory_rate", 0),
            "weight": st.session_state.get("patient_weight", 0.0),
            "height_feet": st.session_state.get("patient_height_feet", 0),
            "height_inches": st.session_state.get("patient_height_inches", 0),
            "blood_glucose": st.session_state.get("blood_glucose", 0.0),
            "temperature": st.session_state.get("body_temperature", 0.0),
            "temperature_scale": st.session_state.get("temperature_scale", "°F"),
            "oxygen_saturation": st.session_state.get("oxygen_saturation", 0),
            "waist_circumference": st.session_state.get("waist_circumference", 0.0),
            "blood_group": st.session_state.get("blood_group", "A+"),
        },
        "safety_result": st.session_state.get("safety_result"),
        "rag_evidence": st.session_state.get("rag_evidence"),
        "readmission_result": st.session_state.get("readmission_result"),
        "clinical_classification_result": st.session_state.get(
            "clinical_classification_result"
        ),
        "heart_disease_result": st.session_state.get(
            "heart_disease_result"
        ),
        "medication_result": st.session_state.get("medication_result"),
        "stroke_result": st.session_state.get("stroke_result"),
        "breast_cancer_result": st.session_state.get("breast_cancer_result"),
    }


def clear_all_session_inputs():
    new_version = st.session_state.get("uploader_version", 0) + 1
    for key in list(st.session_state.keys()):
        del st.session_state[key]

    st.session_state.uploader_version = new_version
    st.session_state.show_results = False
    st.session_state.analysis_results = None
    st.session_state.readmission_result = None
    st.session_state.clinical_classification_result = None
    st.session_state.safety_result = None
    st.session_state.medication_result = None
    st.session_state.stroke_result = None
    st.session_state.uploaded_files_info = []
    st.session_state.breast_cancer_result = None
    st.session_state.last_breast_us_signature = None
    st.session_state.breast_cancer_image_path = None
    st.session_state.breast_cancer_note_added_signature = None
    reset_session_defaults()


def reset_session_defaults():
    """Restore default session-state values after a full clear."""
    defaults = {
        # Reset the sidebar navigation itself. Without this, its key gets
        # wiped along with everything else during a full clear, and the
        # radio widget and the rendered section can end up out of sync
        # after the immediate rerun (sidebar highlighting one section while
        # the page shows another).
        "dashboard_navigation": "Patient Information",
        "clinical_case_input": "",
        "chief_complaint": "",
        "chief_complaint_input": "",
        "symptoms": "",
        "symptoms_input": "",
        "clinical_notes_input": "",
        "lab_data": "",
        "medical_analysis_image_findings_input": "",
        "medical_analysis_document_findings_input": "",
        "analysis_results": None,
        "readmission_result": None,
        "saved_readmission_inputs": {},
        "clinical_classification_result": None,
        "heart_disease_result": None,
        "breast_cancer_result": None,
        "last_breast_us_signature": None,
        "breast_cancer_image_path": None,
        "breast_cancer_note_added_signature": None,
        # Stroke model state
        "stroke_result": None,
        "stroke_hypertension": "No",
        "stroke_heart_disease": "No",
        "stroke_ever_married": "No",
        "stroke_work_type": "Private",
        "stroke_residence_type": "Urban",
        "stroke_smoking_status": "never smoked",
        "stroke_glucose": 100.0,
        "stroke_weight": 70.0,
        "stroke_height": 170.0,
        "uploaded_files_info": [],
        "document_types": {},
        "selected_common_symptoms": [],
        "safety_result": None,
        "medication_result": None,
        "saved_medical_history": None,
        "saved_chief_complaint": "",
        "saved_symptoms": "",
        "saved_selected_common_symptoms": [],
        "saved_clinical_notes": "",
        "saved_clinical_case": "",
        "pending_generated_clinical_case": "",
        "pending_generated_clinical_notes": "",
        "patient_id": "",
        "patient_id_input": "",
        "patient_name": "",
        "patient_name_input": "",
        "patient_age": 30,
        "patient_age_input": 30,
        "patient_gender": "Male",
        "patient_phone": "",
        "patient_phone_input": "",
        "patient_address": "",
        "patient_address_input": "",
        "patient_cnic": "",
        "patient_cnic_input": "",
        "patient_email": "",
        "patient_email_input": "",
        "emergency_contact_phone": "",
        "emergency_contact_phone_input": "",
        "emergency_contact_relationship": "",
        "emergency_contact_relationship_input": "",
        "member_type": "Regular",
        "member_type_input": "Regular",
        "patient_date_of_birth": None,
        "patient_date_of_birth_input": None,
        "payment_method": "Cash",
        "payment_method_input": "Cash",
        "selected_date": None,
        "selected_date_input": None,
        "saved_last_visit": "",
        "last_visit": "",
        "last_visit_input": "",
        "saved_other_symptoms": "",
        "saved_image_findings": "",
        "medical_history": "",
        "medical_history_input": "",
        "chronic_conditions": "",
        "chronic_conditions_input": "",
        "past_surgeries": "",
        "past_surgeries_input": "",
        "known_allergies": "",
        "known_allergies_input": "",
        "current_medications": "",
        "current_medications_input": "",
        "family_history": "",
        "family_history_input": "",
        "immunization_history": "",
        "immunization_history_input": "",
        "previous_admissions": 0,
        "previous_admissions_input": 0,
        "length_of_stay": 1,
        "length_of_stay_input": 1,
        "emergency_visits": 0,
        "emergency_visits_input": 0,
        "number_of_medications": 0,
        "number_of_medications_input": 0,
        "discharge_disposition": "Home / Self Care",
        "discharge_disposition_input": "Home / Self Care",

        "blood_pressure": "",
        "blood_pressure_input": "",
        "heart_rate": 0,
        "heart_rate_input": 0,
        "respiratory_rate": 0,
        "respiratory_rate_input": 0,
        "patient_weight": 0.0,
        "patient_weight_input": 0.0,
        "patient_height_feet": 0,
        "patient_height_feet_input": 0,
        "patient_height_inches": 0,
        "patient_height_inches_input": 0,
        "blood_glucose": 0.0,
        "blood_glucose_input": 0.0,
        "body_temperature": 0.0,
        "body_temperature_input": 0.0,
        "temperature_scale": "°F",
        "temperature_scale_input": "°F",
        "oxygen_saturation": 0,
        "oxygen_saturation_input": 0,
        "waist_circumference": 0.0,
        "waist_circumference_input": 0.0,
        "blood_group": "A+",
        "blood_group_input": "A+",
        "show_results": False,
    }

    for index in range(len(COMMON_SYMPTOMS)):
        defaults[f"symptom_checkbox_{index}"] = False

    for key, value in defaults.items():
        st.session_state[key] = value


def generate_diagnostic_report_compat(diagnostic_agent, **report_inputs):
    required_report_structure = """
Return only one structured medical diagnostic report.

Use every heading below exactly once and in this exact order:

Comprehensive Diagnostic Analysis
Potential Diagnoses Ordered by Likelihood
Critical Findings Requiring Immediate Attention
Evidence-Based Recommendations for Treatment or Further Testing
Impression
Recommendations
Clinical Summary
Risk Assessment and Urgency Level Determination
30-Day Readmission Risk Assessment
AI-Assisted Clinical Note Classification

Formatting requirements:
- Put every heading on its own line.
- Under Comprehensive Diagnostic Analysis, write one clear paragraph combining the patient's main symptoms, relevant clinical notes, important laboratory findings, and medical imaging findings. Use the actual available patient data. Follow this style: "Based on the provided information, the patient presents with [main symptoms]. The clinical notes mention [relevant history]. Laboratory results indicate [important findings], suggesting [clinical meaning]. The imaging findings show [important imaging findings], indicating [possible condition]."
- Do not repeat Patient Information because the app displays it separately.
- Do not repeat findings, diagnoses, recommendations, or urgency information.
- Use numbered items where appropriate.
- Do not return a Python dictionary, JSON, code block, or duplicate report.
- Do not include keys such as diagnostic_analysis, formatted_report, or summary.
- Write in formal, professional clinical English suitable for a report read by physicians. Use precise medical and diagnostic terminology rather than simplified, casual, or layman phrasing.
- Do not use asterisks, underscores, backticks, or any other markdown emphasis characters (no *, **, _, `) anywhere in the report. Write plain sentences instead of markdown-styled emphasis.
- Under 30-Day Readmission Risk Assessment and AI-Assisted Clinical Note Classification, briefly restate the prediction, score/confidence, and basis exactly as given in the input data below; do not invent or alter the numbers. The application will overwrite these two sections with the authoritative computed values regardless, so keep this brief.
""".strip()

    report_inputs["report_instructions"] = required_report_structure

    candidate_methods = [
        "generate_comprehensive_report",
        "generate_diagnostic_report",
        "generate_report",
        "analyze_clinical_case",
        "analyze_case",
        "analyze",
    ]

    available_method = None
    available_name = None

    for method_name in candidate_methods:
        method = getattr(diagnostic_agent, method_name, None)
        if callable(method):
            available_method = method
            available_name = method_name
            break

    if available_method is None:
        public_methods = [
            name for name in dir(diagnostic_agent)
            if not name.startswith("_") and callable(getattr(diagnostic_agent, name, None))
        ]
        raise AttributeError(
            "DiagnosticAgent has no supported report-generation method. "
            f"Available public methods: {', '.join(public_methods) or 'none'}"
        )

    try:
        signature = inspect.signature(available_method)
        parameters = signature.parameters
        accepts_all_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )

        if accepts_all_kwargs:
            return available_method(**report_inputs)

        accepted_kwargs = {
            key: value
            for key, value in report_inputs.items()
            if key in parameters
        }

        if accepted_kwargs:
            return available_method(**accepted_kwargs)
    except (TypeError, ValueError):
        pass

    patient = report_inputs.get("patient_info", {}) or {}
    combined_prompt = f"""
Patient Information
Patient ID: {patient.get('id', '')}
Name: {patient.get('name', '')}
Age: {patient.get('age', '')}
Gender: {patient.get('gender', '')}
Last Visit: {patient.get('last_visit', '')}
Medical History: {patient.get('medical_history', '')}

Clinical Case
{report_inputs.get('clinical_case', '')}

Clinical Notes
{report_inputs.get('clinical_notes', '')}

Laboratory Data
{report_inputs.get('lab_data', '')}

Medical Imaging Findings
{report_inputs.get('image_findings', '')}

Clinical Documents and Data
{report_inputs.get('documents', '')}

Vitals
{report_inputs.get('vitals', '')}

30-Day Readmission Risk Assessment
{report_inputs.get('readmission_summary', '')}

Required Report Structure
{required_report_structure}
""".strip()

    try:
        return available_method(combined_prompt)
    except TypeError as error:
        raise TypeError(
            f"The method '{available_name}' exists, but its parameters do not match "
            "the report inputs used by app.py. Update its signature or add one of the "
            "supported report methods."
        ) from error


def extract_report_text(report_result):
    if report_result is None:
        return ""

    if isinstance(report_result, str):
        return report_result

    if isinstance(report_result, dict):
        preferred_keys = (
            "formatted_report",
            "report",
            "report_text",
            "diagnostic_report",
            "diagnostic_analysis",
            "analysis",
            "content",
            "text",
        )

        for key in preferred_keys:
            value = report_result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        for value in report_result.values():
            if isinstance(value, str) and value.strip():
                return value.strip()

        return ""

    return str(report_result)


def ensure_required_diagnostic_headings(report_text):
    if not isinstance(report_text, str):
        report_text = str(report_text or "")

    text = report_text.replace("\r\n", "\n").replace("\\n", "\n").strip()

    text = re.sub(r"\*+", "", text)
    text = re.sub(r"__+", "", text)
    text = text.replace("`", "")

    heading_order = [
        "Comprehensive Diagnostic Analysis",
        "Potential Diagnoses Ordered by Likelihood",
        "Critical Findings Requiring Immediate Attention",
        "Evidence-Based Recommendations for Treatment or Further Testing",
        "Impression",
        "Recommendations",
        "Clinical Summary",
        "Risk Assessment and Urgency Level Determination",
        "30-Day Readmission Risk Assessment",
        "AI-Assisted Clinical Note Classification",
    ]

    aliases = {
        "Comprehensive Diagnostic Analysis": [
            "Comprehensive Diagnostic Analysis",
            "Detailed Diagnostic Analysis",
            "Clinical Presentation",
        ],
        "Potential Diagnoses Ordered by Likelihood": [
            "Potential Diagnoses Ordered by Likelihood",
            "Potential Diagnoses ordered by likelihood",
            "Potential Diagnoses",
            "Diagnostic Impressions",
            "Diagnostic Impression",
            "Differential Diagnoses",
            "Differential Diagnosis",
            "Working Diagnosis",
        ],
        "Critical Findings Requiring Immediate Attention": [
            "Critical Findings Requiring Immediate Attention",
            "Critical Findings",
            "Red Flags",
        ],
        "Evidence-Based Recommendations for Treatment or Further Testing": [
            "Evidence-Based Recommendations for Treatment or Further Testing",
            "Evidence-Based Recommendations",
            "Treatment Recommendations",
            "Further Testing",
        ],
        "Risk Assessment and Urgency Level Determination": [
            "Risk Assessment and Urgency Level Determination",
            "Risk Assessment and Urgency",
            "Risk Assessment",
        ],
        "Clinical Summary": [
            "Clinical Summary",
            "Clinical Findings",
            "Clinical Assessment",
            "Assessment",
            "History",
        ],
        "Impression": [
            "Impression",
            "Clinical Impression",
            "Clinical Impressions",
        ],
        "Recommendations": [
            "Recommendations",
            "Management Recommendations",
        ],
        "30-Day Readmission Risk Assessment": [
            "30-Day Readmission Risk Assessment",
            "30 Day Readmission Risk Assessment",
            "Readmission Risk Assessment",
            "Readmission Prediction",
            "Readmission Risk",
        ],
        "AI-Assisted Clinical Note Classification": [
            "AI-Assisted Clinical Note Classification",
            "Clinical Note Classification",
            "Clinical Classification",
            "Predicted Clinical Category",
        ],
    }

    text = re.sub(
        r"(?is)(?:^|\n)\s*(?:[-#*\s]*Patient Information[-#*\s]*:?\s*\n)"
        r".*?(?=\n\s*(?:[-#*\s]*(?:Comprehensive Diagnostic Analysis|"
        r"Potential Diagnoses|Critical Findings|Evidence-Based Recommendations|"
        r"Risk Assessment|Medical Report|Clinical Summary|Impression|"
        r"Recommendations|Urgency Level)[-#*\s]*:?\s*$)|\Z)",
        "\n",
        text,
    )

    alias_to_heading = {}
    alias_names = []

    for heading, names in aliases.items():
        for name in names:
            alias_to_heading[name.lower()] = heading
            alias_names.append(name)

    alias_names.sort(key=len, reverse=True)
    alias_pattern = "|".join(re.escape(name) for name in alias_names)

    heading_regex = re.compile(
        rf"(?im)^\s*(?:[-*#\s]*)(?P<heading>{alias_pattern})"
        rf"\s*:?\s*(?P<inline>[^\n]*)$"
    )

    matches = list(heading_regex.finditer(text))
    sections = {heading: [] for heading in heading_order}

    if matches:
        prefix = text[:matches[0].start()].strip()
        if prefix:
            sections["Comprehensive Diagnostic Analysis"].append(prefix)

        for index, match in enumerate(matches):
            heading = alias_to_heading[
                match.group("heading").strip().lower()
            ]

            inline_text = match.group("inline").strip()
            body_start = match.end()
            body_end = (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(text)
            )
            body_text = text[body_start:body_end].strip()

            combined_text = "\n".join(
                item for item in (inline_text, body_text) if item
            ).strip()

            if combined_text:
                sections[heading].append(combined_text)
    elif text:
        sections["Comprehensive Diagnostic Analysis"].append(text)

    for heading in heading_order:
        unique_content = []
        seen_content = set()

        for item in sections[heading]:
            normalized_item = re.sub(
                r"\s+",
                " ",
                item
            ).strip().lower()

            if normalized_item and normalized_item not in seen_content:
                seen_content.add(normalized_item)
                unique_content.append(item.strip())

        sections[heading] = unique_content

    missing_text = {
        "Comprehensive Diagnostic Analysis":
            "No comprehensive diagnostic analysis was provided.",
        "Potential Diagnoses Ordered by Likelihood":
            "No potential diagnoses were provided.",
        "Critical Findings Requiring Immediate Attention":
            "No critical findings requiring immediate attention were identified.",
        "Evidence-Based Recommendations for Treatment or Further Testing":
            "No treatment or further-testing recommendations were provided.",
        "Risk Assessment and Urgency Level Determination":
            "No risk assessment or urgency determination was provided.",
        "Clinical Summary":
            "No clinical summary was provided.",
        "Impression":
            "No impression was provided.",
        "Recommendations":
            "No additional recommendations were provided.",
        "30-Day Readmission Risk Assessment":
            "No 30-day readmission risk assessment was provided.",
        "AI-Assisted Clinical Note Classification":
            "No clinical note classification was performed.",
    }

    formatted_report = []

    for heading in heading_order:
        section_content = "\n\n".join(
            item.strip()
            for item in sections[heading]
            if item.strip()
        ).strip()

        if not section_content:
            section_content = missing_text[heading]

        formatted_report.append(
            f"### {heading}\n{section_content}"
        )

    return "\n\n".join(formatted_report).strip()


def fill_report_patient_details(
    report_content,
    patient_name,
    patient_age,
    patient_gender,
    evaluation_date,
):
    if not isinstance(report_content, str):
        return report_content

    replacements = {
        "[Patient's Name]": patient_name or "Not provided",
        "[Patient Name]": patient_name or "Not provided",
        "[Patient]": patient_name or "Not provided",
        "[Age]": str(patient_age),
        "[Patient's Age]": str(patient_age),
        "[Gender]": patient_gender or "Not provided",
        "[Patient's Gender]": patient_gender or "Not provided",
        "[Date]": evaluation_date,
        "[Evaluation Date]": evaluation_date,
        "[Date of Evaluation]": evaluation_date,
        "[Date of Consultation]": evaluation_date,
    }

    updated_content = report_content

    for placeholder, actual_value in replacements.items():
        updated_content = updated_content.replace(
            placeholder,
            str(actual_value)
        )

    return updated_content


def remove_duplicate_patient_placeholder(report_content):
    if not isinstance(report_content, str):
        return report_content

    cleaned_lines = []

    for raw_line in report_content.replace("\r\n", "\n").split("\n"):
        plain_line = re.sub(r"[*_#`]", "", raw_line).strip()
        normalised = re.sub(r"\s+", " ", plain_line).lower()

        contains_patient_placeholders = (
            "patient name:" in normalised
            and "age:" in normalised
            and "gender:" in normalised
            and (
                "date of consultation:" in normalised
                or "date of admission:" in normalised
                or "[patient]" in normalised
                or "[age]" in normalised
                or "[gender]" in normalised
                or "[date]" in normalised
            )
        )

        if contains_patient_placeholders:
            continue

        cleaned_lines.append(raw_line)

    return "\n".join(cleaned_lines).strip()


def standardize_report_headings(report_content):
    if not isinstance(report_content, str):
        return report_content

    report_content = fill_report_patient_details(
        report_content,
        patient_name=st.session_state.patient_name,
        patient_age=st.session_state.patient_age,
        patient_gender=st.session_state.patient_gender,
        evaluation_date=datetime.now().strftime("%d-%m-%Y"),
    )

    known_headings = {
        "Comprehensive Diagnostic Analysis",
        "Potential Diagnoses Ordered by Likelihood",
        "Critical Findings Requiring Immediate Attention",
        "Evidence-Based Recommendations for Treatment or Further Testing",
        "Impression",
        "Recommendations",
        "Clinical Summary",
        "Risk Assessment and Urgency Level Determination",
        "30-Day Readmission Risk Assessment",
        "AI-Assisted Clinical Note Classification",
    }

    bullet_headings = {
        "Potential Diagnoses Ordered by Likelihood",
        "Critical Findings Requiring Immediate Attention",
        "Evidence-Based Recommendations for Treatment or Further Testing",
        "Recommendations",
    }

    normal_text_headings = {
        "Risk Assessment and Urgency Level Determination",
    }

    formatted_lines = []
    current_section = None

    for raw_line in report_content.replace("\r\n", "\n").split("\n"):
        stripped = raw_line.strip()

        if not stripped:
            formatted_lines.append("")
            continue

        plain_line = stripped.lstrip("#").strip()
        heading_text = plain_line.rstrip(":")

        if heading_text in known_headings:
            current_section = heading_text
            formatted_lines.append(f"### {heading_text}")
            continue

        if current_section in bullet_headings:
            bullet_text = re.sub(r"^\d+[.)]\s*", "", plain_line)
            if bullet_text.startswith(("-", "*", "•")):
                bullet_text = bullet_text[1:].strip()
            if bullet_text:
                formatted_lines.append(f"- {bullet_text}")

        elif current_section in normal_text_headings:
            urgency_text = re.sub(
                r"^[#\s]+",
                "",
                stripped
            )
            urgency_text = urgency_text.replace("**", "")
            urgency_text = urgency_text.replace("__", "")
            urgency_text = urgency_text.replace("`", "")
            urgency_text = urgency_text.strip()

            if urgency_text:
                formatted_lines.append(
                    '<div style="font-size:0.95rem; '
                    'font-weight:400; line-height:1.65; '
                    'margin:0.25rem 0 0.75rem 0;">'
                    f'{urgency_text}'
                    '</div>'
                )
        else:
            formatted_lines.append(plain_line)

    return "\n".join(formatted_lines)


def build_medical_history_summary():
    sections = [
        ("Health Issues", st.session_state.get("chronic_conditions", "")),
        ("Past Surgeries", st.session_state.get("past_surgeries", "")),
        ("Known Allergies", st.session_state.get("known_allergies", "")),
        ("Current Medications", st.session_state.get("current_medications", "")),
        ("Family History", st.session_state.get("family_history", "")),
        ("Immunization History", st.session_state.get("immunization_history", "")),
        ("Additional Notes", st.session_state.get("medical_history", "")),
    ]

    formatted_sections = [
        f"{label}: {value.strip()}"
        for label, value in sections
        if value and value.strip()
    ]

    return "\n".join(formatted_sections)


def build_symptoms_summary():
    sync_clinical_widget_state()

    checked_symptoms = st.session_state.get("selected_common_symptoms", [])
    if not checked_symptoms:
        checked_symptoms = [
            name
            for index, name in enumerate(COMMON_SYMPTOMS)
            if st.session_state.get(f"symptom_checkbox_{index}", False)
        ]

    other_symptoms = (
        st.session_state.get("symptoms")
        or st.session_state.get("symptoms_input")
        or ""
    ).strip()

    parts = []
    if checked_symptoms:
        parts.append(", ".join(checked_symptoms))
    if other_symptoms:
        parts.append(other_symptoms)

    return "; ".join(parts)


def build_vitals_dict_for_safety_agent():
    vitals = {}

    bp_raw = (st.session_state.get("blood_pressure") or "").strip()
    if "/" in bp_raw:
        try:
            vitals["sbp"] = int(bp_raw.split("/")[0].strip())
        except ValueError:
            pass

    spo2 = st.session_state.get("oxygen_saturation")
    if spo2:
        vitals["spo2"] = spo2

    hr = st.session_state.get("heart_rate")
    if hr:
        vitals["hr"] = hr

    temp = st.session_state.get("body_temperature")
    if temp:
        if st.session_state.get("temperature_scale") == "\u00b0F":
            vitals["temp_c"] = SafetyAgent.fahrenheit_to_celsius(temp)
        else:
            vitals["temp_c"] = temp

    rr = st.session_state.get("respiratory_rate")
    if rr:
        vitals["rr"] = rr

    glucose = st.session_state.get("blood_glucose")
    if glucose:
        vitals["glucose"] = glucose

    return vitals


def build_readmission_report_section(readmission_data):
    rr = readmission_data or {}
    prediction = rr.get("prediction", "NO")
    probability = rr.get("probability", 0)
    risk_level = rr.get("risk_level", "Low")
    reason = rr.get("reason") or "Risk assessment based on clinical categories."
    summary = rr.get("summary") or f"Estimated 30-Day Risk: {probability}%. Risk Level: {risk_level}."
    recommendations = rr.get("recommendations") or []

    if risk_level == "High":
        pred_text = "High risk of readmission within 30 days (YES)."
    elif risk_level == "Medium":
        pred_text = "Moderate risk of readmission within 30 days (YES)."
    elif risk_level == "Low":
        pred_text = "Low risk of readmission within 30 days (NO)."
    else:
        pred_text = f"Readmission status: {prediction}."

    prev_adm = st.session_state.get("previous_admissions", 0)
    los = st.session_state.get("length_of_stay", 0)
    ed = st.session_state.get("emergency_visits", 0)
    meds = st.session_state.get("number_of_medications", 0)
    disp = st.session_state.get("discharge_disposition", "Home / Self Care")

    lines = [
        f"Prediction: {pred_text}",
        f"Estimated 30-Day Risk: {probability}%.",
        f"Risk Level: {risk_level}.",
        f"Evaluated Factors: Previous Admissions ({prev_adm}), Length of Stay ({los} days), Emergency Visits ({ed}), Medications ({meds}), Disposition ({disp}).",
        f"Summary: {summary}",
        f"AI Clinical Reasoning: {reason}",
    ]

    if recommendations:
        lines.append("Recommended Actions:")
        lines.extend(f"- {item}" for item in recommendations)

    return "### 30-Day Readmission Risk Assessment\n" + "\n".join(lines)


def build_classification_report_section(classification_result):
    if not classification_result:
        return (
            "### AI-Assisted Clinical Note Classification\n"
            "No clinical note classification was performed."
        )

    predicted_class = str(classification_result.get("predicted_class", "")).strip()
    confidence = classification_result.get("confidence", 0.0)
    recommendations = classification_result.get("recommendations") or []

    lines = [
        f"Condition Identified: {predicted_class.title() or 'Not available'}.",
        f"Confidence Score: {confidence:.0%}.",
    ]

    if recommendations:
        lines.append("Recommended Actions:")
        lines.extend(f"- {item}" for item in recommendations)

    return "### AI-Assisted Clinical Note Classification\n" + "\n".join(lines)


def build_heart_disease_report_section(heart_result):
    """Build the additional Heart Disease Analytics section for the report."""
    if not heart_result:
        return (
            "### Heart Disease Analytics\n"
            "No heart disease analytics were performed."
        )

    prediction = str(heart_result.get("prediction", "NO")).upper()
    probability = float(heart_result.get("probability", 0.0))
    risk_level = heart_result.get("risk_level", "Unavailable")
    description = heart_result.get(
        "description",
        "No heart disease description available.",
    )

    return (
        "### Heart Disease Analytics\n"
        f"Prediction: {prediction}.\n"
        f"Risk Level: {risk_level}.\n"
        f"Probability: {int(round(probability * 100))}%.\n"
        f"Assessment Details: {description}"
    )


def _medication_is_critical(med):
    """True when MedicationAgent deferred routine OTC due to SafetyAgent escalation."""
    if not med:
        return False
    return med.get("status") == "emergency" or bool(med.get("emergency_escalation_required"))


def _render_medication_section_block(title, content, *, icon="💊", style="default"):
    """Render a titled medication subsection when content is present."""
    if not (content or "").strip():
        return
    st.markdown(
        f"""
        <div class="section-header">
            <span class="icon-blue">{icon}</span>{title}
        </div>
        """,
        unsafe_allow_html=True,
    )
    if style == "warning":
        st.warning(content.strip())
    elif style == "error":
        st.error(content.strip())
    elif style == "info":
        st.info(content.strip())
    else:
        st.markdown(content.strip())
    st.markdown("<br>", unsafe_allow_html=True)


def render_medication_advisor_ui(med):
    """Render the Medication Advisor tab from a structured MedicationAgent result."""
    if _medication_is_critical(med):
        st.markdown(
            """
            <div class="section-header" style="margin-top:0;">
                <span class="icon-pink">🚨</span>Medication Recommendations
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.error("CRITICAL SAFETY FINDING")

        critical_text = (
            med.get("critical_safety_finding", "").strip()
            or med.get("reasoning_summary", "").strip()
        )
        if critical_text:
            st.markdown(critical_text)

        st.markdown("<br>", unsafe_allow_html=True)
        _render_medication_section_block(
            "Immediate Safety Recommendations",
            med.get("immediate_safety_recommendations", ""),
            icon="🆘",
            style="error",
        )
        _render_medication_section_block(
            "Why Routine OTC Medication Is Not Recommended",
            med.get("why_otc_withheld", ""),
            icon="⛔",
            style="info",
        )
        _render_medication_section_block(
            "Patient-Specific Warnings",
            med.get("patient_specific_warnings", ""),
            icon="⚠️",
            style="warning",
        )
        _render_medication_section_block(
            "Monitoring / Warning Signs",
            med.get("monitoring_requirements", ""),
            icon="👁️",
        )
        _render_medication_section_block(
            "When to Seek Emergency Care",
            med.get("when_to_escalate", "") or med.get("when_to_seek_prescription_care", ""),
            icon="🚑",
            style="error",
        )
    else:
        st.markdown(
            """
            <div class="section-header" style="margin-top:0;">
                <span class="icon-blue">💊</span>Medication Recommendations
            </div>
            """,
            unsafe_allow_html=True,
        )

        structured_meds = med.get("medications") or []
        st.markdown(
            """
            <div class="section-header">
                <span class="icon-blue">💊</span>OTC Medication Options
            </div>
            """,
            unsafe_allow_html=True,
        )
        if structured_meds:
            for entry in structured_meds:
                name = entry.get("name", "").strip()
                if not name:
                    continue
                detail_lines = [f"**{name}**"]
                if entry.get("purpose"):
                    detail_lines.append(f"- Purpose: {entry['purpose']}")
                if entry.get("dosage_information"):
                    detail_lines.append(f"- Dosage: {entry['dosage_information']}")
                if entry.get("route"):
                    detail_lines.append(f"- Route: {entry['route']}")
                if entry.get("frequency"):
                    detail_lines.append(f"- Frequency: {entry['frequency']}")
                if entry.get("duration"):
                    detail_lines.append(f"- Duration: {entry['duration']}")
                if entry.get("important_precautions"):
                    detail_lines.append(f"- Precautions: {entry['important_precautions']}")
                st.markdown("  \n".join(detail_lines))
        elif med.get("otc_medications", "").strip():
            st.markdown(med["otc_medications"].strip())
        else:
            st.caption(
                "No OTC medication options were identified for the confirmed diagnoses. "
                "See self-care, monitoring, and escalation guidance below."
            )
        st.markdown("<br>", unsafe_allow_html=True)

        _render_medication_section_block(
            "Self-Care and Supportive Recommendations",
            med.get("self_care_recommendations", ""),
            icon="🧘",
        )

        dose_text = med.get("dosage_duration", "").strip() or med.get("dosage_information", "").strip()
        st.markdown(
            """
            <div class="section-header">
                <span class="icon-teal">⏱️</span>Dosage and Duration
            </div>
            """,
            unsafe_allow_html=True,
        )
        if dose_text:
            st.markdown(dose_text)
        else:
            st.caption("No dosage guidance generated.")
        st.markdown("<br>", unsafe_allow_html=True)

        # Medication Safety Overview
        contra_text = med.get("contraindications", "")
        contra_bullets = med.get("contraindication_bullets", [])
        interact_text = med.get("drug_interactions", "").strip()
        warnings_text = med.get("patient_warnings", "")
        warning_bullets = med.get("patient_warning_bullets", [])

        st.markdown(
            """
            <div class="section-header">
                <span class="icon-orange">🛡️</span>Medication Safety Summary
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Contraindications")
        if contra_text or contra_bullets:
            if contra_text:
                st.write(contra_text)
            for bullet in contra_bullets:
                st.markdown(f"- {bullet}")
        else:
            st.success("No contraindications identified.")

        st.markdown("#### Drug Interactions")
        if interact_text:
            st.info(interact_text)
        else:
            st.success("No potential drug interactions identified.")

        st.markdown("#### Patient-Specific Warnings")
        if warnings_text or warning_bullets:
            if warnings_text:
                st.warning(warnings_text)
            for bullet in warning_bullets:
                st.markdown(f"- {bullet}")
        else:
            st.success("No patient-specific warnings generated.")

        st.markdown("<br>", unsafe_allow_html=True)

        _render_medication_section_block(
            "Monitoring Requirements",
            med.get("monitoring_requirements", ""),
            icon="👁️",
        )

        st.markdown(
            """
            <div class="section-header">
                <span class="icon-pink">🚨</span>When to Seek Prescription or Emergency Care
            </div>
            """,
            unsafe_allow_html=True,
        )
        escalate_text = (
            med.get("when_to_escalate", "").strip()
            or med.get("when_to_seek_prescription_care", "").strip()
        )
        if escalate_text:
            st.error(escalate_text)
        else:
            st.caption("No specific escalation triggers identified.")
        st.markdown("<br>", unsafe_allow_html=True)

    disclaimer_text = med.get("pharmacist_disclaimer", "").strip()
    if disclaimer_text:
        st.markdown(
            f"""
            <div style="
                background: #f7f7ff;
                border: 1px solid #c7bdfb;
                border-radius: 12px;
                padding: 14px 18px;
                color: #3730a3;
                font-size: 0.88rem;
                line-height: 1.55;
            ">
                <b>⚕️ Pharmacist Disclaimer</b><br><br>
                {disclaimer_text}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_medication_report_ui(med):
    """Compact medication rendering for the Diagnostic Report tab."""
    if _medication_is_critical(med):
        st.error("CRITICAL SAFETY FINDING")
        critical_text = (
            med.get("critical_safety_finding", "").strip()
            or med.get("reasoning_summary", "").strip()
        )
        if critical_text:
            st.markdown(critical_text)
        if med.get("immediate_safety_recommendations", "").strip():
            st.markdown(
                f"**Immediate Safety Recommendations:** "
                f"{med['immediate_safety_recommendations'].strip()}"
            )
        if med.get("why_otc_withheld", "").strip():
            st.markdown(
                f"**Why Routine OTC Medication Is Not Recommended:** "
                f"{med['why_otc_withheld'].strip()}"
            )
        if med.get("patient_specific_warnings", "").strip():
            st.markdown(
                f"**Patient-Specific Warnings:** {med['patient_specific_warnings'].strip()}"
            )
        if med.get("monitoring_requirements", "").strip():
            st.markdown(
                f"**Monitoring / Warning Signs:** {med['monitoring_requirements'].strip()}"
            )
        if med.get("when_to_escalate", "").strip():
            st.markdown(
                f"**When to Seek Emergency Care:** {med['when_to_escalate'].strip()}"
            )
        return

    structured_meds = med.get("medications") or []
    if structured_meds:
        for entry in structured_meds:
            name = entry.get("name", "").strip()
            if not name:
                continue
            detail_lines = [f"**{name}**"]
            if entry.get("purpose"):
                detail_lines.append(f"- Purpose: {entry['purpose']}")
            if entry.get("dosage_information"):
                detail_lines.append(f"- Dosage: {entry['dosage_information']}")
            if entry.get("route"):
                detail_lines.append(f"- Route: {entry['route']}")
            if entry.get("frequency"):
                detail_lines.append(f"- Frequency: {entry['frequency']}")
            if entry.get("duration"):
                detail_lines.append(f"- Duration: {entry['duration']}")
            if entry.get("important_precautions"):
                detail_lines.append(f"- Precautions: {entry['important_precautions']}")
            st.markdown("  \n".join(detail_lines))
    elif med.get("otc_medications", "").strip():
        st.markdown(med["otc_medications"].strip())
    else:
        st.caption(
            "No OTC medication options were identified for the confirmed diagnoses. "
            "See self-care, monitoring, and escalation guidance below."
        )

    if med.get("self_care_recommendations", "").strip():
        st.markdown(
            f"**Self-Care and Supportive Recommendations:** "
            f"{med['self_care_recommendations'].strip()}"
        )
    dosage_text = (med.get("dosage_duration") or med.get("dosage_information") or "").strip()
    if dosage_text:
        st.markdown(f"**Dosage and Duration:** {dosage_text}")
    if med.get("contraindications", "").strip():
        st.markdown(f"**Contraindications:** {med['contraindications'].strip()}")
    if med.get("drug_interactions", "").strip():
        st.markdown(f"**Drug Interactions:** {med['drug_interactions'].strip()}")
    if med.get("patient_specific_warnings", "").strip():
        st.markdown(f"**Patient-Specific Warnings:** {med['patient_specific_warnings'].strip()}")
    if med.get("monitoring_requirements", "").strip():
        st.markdown(f"**Monitoring Requirements:** {med['monitoring_requirements'].strip()}")
    if med.get("when_to_escalate", "").strip() or med.get("when_to_seek_prescription_care", "").strip():
        st.markdown(
            f"**When to Seek Prescription or Emergency Care:** "
            f"{(med.get('when_to_escalate') or med.get('when_to_seek_prescription_care') or '').strip()}"
        )


def build_medication_report_section(medication_result):
    """Build the Medication Recommendations section for the report/PDF.

    Follows the same pattern as build_readmission_report_section() /
    build_classification_report_section() / build_heart_disease_report_section():
    it only formats the already-structured MedicationAgent result — it never
    invents or alters medication content.
    """
    med = medication_result or {}

    if not med or med.get("error") or med.get("status") == "unavailable":
        reason = (
            med.get("reason")
            or med.get("reasoning_summary")
            or med.get("error")
            or "Medication recommendations are currently unavailable."
        )
        return (
            "### Medication Recommendations\n"
            f"Status: Unavailable.\n"
            f"Reason: {reason}"
        )

    lines = []
    status = med.get("status", "ok")

    if _medication_is_critical(med):
        lines.append("Status: Emergency escalation required — routine OTC recommendations withheld.")
        if med.get("critical_safety_finding", "").strip():
            lines.append(f"Critical Safety Finding: {med['critical_safety_finding'].strip()}")
        if med.get("immediate_safety_recommendations", "").strip():
            lines.append(
                f"Immediate Safety Recommendations: {med['immediate_safety_recommendations'].strip()}"
            )
        if med.get("why_otc_withheld", "").strip():
            lines.append(
                f"Why Routine OTC Medication Is Not Recommended: {med['why_otc_withheld'].strip()}"
            )
        warnings = (med.get("patient_specific_warnings") or "").strip()
        if warnings:
            lines.append(f"Patient-Specific Warnings: {warnings}")
        monitoring = (med.get("monitoring_requirements") or "").strip()
        if monitoring:
            lines.append(f"Monitoring / Warning Signs: {monitoring}")
        escalation = (med.get("when_to_escalate") or med.get("when_to_seek_prescription_care") or "").strip()
        if escalation:
            lines.append(f"When to Seek Emergency Care: {escalation}")
    else:
        if status == "emergency" or med.get("emergency_escalation_required"):
            lines.append(
                "Status: Emergency escalation required — routine OTC recommendations withheld."
            )

        structured_meds = med.get("medications") or []
        if structured_meds:
            for entry in structured_meds:
                name = entry.get("name", "").strip()
                if not name:
                    continue
                lines.append(f"- Medication: {name}")
                if entry.get("purpose"):
                    lines.append(f"  Purpose / Indication: {entry['purpose']}")
                if entry.get("dosage_information"):
                    lines.append(f"  Dosage Information: {entry['dosage_information']}")
                if entry.get("route"):
                    lines.append(f"  Route: {entry['route']}")
                if entry.get("frequency"):
                    lines.append(f"  Frequency: {entry['frequency']}")
                if entry.get("duration"):
                    lines.append(f"  Duration: {entry['duration']}")
                if entry.get("important_precautions"):
                    lines.append(f"  Important Precautions: {entry['important_precautions']}")
        elif med.get("otc_medications", "").strip():
            lines.append(f"OTC Medication Options: {med['otc_medications'].strip()}")
        else:
            lines.append(
                "No OTC medication options were identified for the confirmed diagnoses. "
                "See self-care, monitoring, and escalation guidance below."
            )

        self_care = (med.get("self_care_recommendations") or "").strip()
        if self_care:
            lines.append(f"Self-Care and Supportive Recommendations: {self_care}")

        dosage_text = (med.get("dosage_duration") or med.get("dosage_information") or "").strip()
        if dosage_text:
            lines.append(f"Dosage and Duration: {dosage_text}")

        contraindications = (med.get("contraindications") or "").strip()
        if contraindications:
            lines.append(f"Contraindications: {contraindications}")

        interactions = (med.get("drug_interactions") or "").strip()
        if interactions:
            lines.append(f"Drug Interactions: {interactions}")

        warnings = (med.get("patient_specific_warnings") or "").strip()
        if warnings:
            lines.append(f"Patient-Specific Warnings: {warnings}")

        monitoring = (med.get("monitoring_requirements") or "").strip()
        if monitoring:
            lines.append(f"Monitoring Requirements: {monitoring}")

        escalation = (med.get("when_to_escalate") or med.get("when_to_seek_prescription_care") or "").strip()
        if escalation:
            lines.append(f"When to Seek Prescription or Emergency Care: {escalation}")

    lines.append(
        f"Physician/Pharmacist Review Required: "
        f"{'Yes' if med.get('physician_review_required') else 'No'}"
    )
    lines.append(
        f"Emergency Escalation Required: "
        f"{'Yes' if med.get('emergency_escalation_required') else 'No'}"
    )

    disclaimer = (med.get("pharmacist_disclaimer") or "").strip()
    if disclaimer:
        lines.append(f"Pharmacist Disclaimer: {disclaimer}")

    return "### Medication Recommendations\n" + "\n".join(lines)


def inject_authoritative_sections(report_text, readmission_data, classification_result):
    readmission_section = build_readmission_report_section(readmission_data)
    classification_section = build_classification_report_section(classification_result)

    if report_text:
        if "### 30-Day Readmission Risk Assessment" in report_text:
            report_text = re.sub(
                r"### 30-Day Readmission Risk Assessment\n.*?"
                r"(?=\n### AI-Assisted Clinical Note Classification|\Z)",
                readmission_section + "\n\n",
                report_text,
                flags=re.DOTALL,
            )
        elif "### AI-Assisted Clinical Note Classification" in report_text:
            report_text = report_text.replace(
                "### AI-Assisted Clinical Note Classification",
                readmission_section + "\n\n### AI-Assisted Clinical Note Classification"
            )
        else:
            report_text = report_text.rstrip() + "\n\n" + readmission_section + "\n\n"

        if "### AI-Assisted Clinical Note Classification" in report_text:
            report_text = re.sub(
                r"### AI-Assisted Clinical Note Classification\n.*\Z",
                classification_section,
                report_text,
                flags=re.DOTALL,
            )
        else:
            report_text = report_text.rstrip() + "\n\n" + classification_section

    return report_text


@st.cache_resource
def get_readmission_predictor_instance():
    """Single shared ReadmissionPredictor instance across the Streamlit app.
    Module-level @st.cache_resource ensures self._cache is preserved across
    all reruns and button clicks during the session."""
    return ReadmissionPredictor()



def _extract_diagnostic_sections(report_text: str) -> dict:
    """
    Parse the structured DiagnosticAgent report text (which uses '### Heading'
    markers guaranteed by ensure_required_diagnostic_headings) into a typed
    dict of section_name → section_content.

    Only the sections relevant to the MedicationAgent are extracted; the rest
    are included under their exact heading name for completeness.
    """
    target_headings = {
        "Potential Diagnoses Ordered by Likelihood",
        "Critical Findings Requiring Immediate Attention",
        "Evidence-Based Recommendations for Treatment or Further Testing",
        "Impression",
        "Recommendations",
        "Clinical Summary",
        "Risk Assessment and Urgency Level Determination",
    }

    sections: dict = {h: "" for h in target_headings}
    current_heading = None
    buffer: list = []

    for line in (report_text or "").splitlines():
        # Strip leading #, *, spaces so both '### Heading' and 'Heading' match
        plain = line.strip().lstrip("#").strip()
        if plain in target_headings:
            if current_heading is not None:
                sections[current_heading] = "\n".join(buffer).strip()
            current_heading = plain
            buffer = []
        elif current_heading is not None:
            buffer.append(line)

    # Flush last section
    if current_heading is not None:
        sections[current_heading] = "\n".join(buffer).strip()

    return sections


def build_combined_clinical_text():
    return " ".join(filter(None, [
        get_effective_chief_complaint(),
        build_effective_symptoms_summary(),
        get_effective_clinical_case(),
        get_effective_clinical_notes(),
        st.session_state.get("known_allergies", ""),
        st.session_state.get("chronic_conditions", ""),
        st.session_state.get("current_medications", ""),
    ]))



def calculate_stroke_bmi(weight_kg, height_cm):
    """Calculate BMI for the stroke model input."""
    try:
        height_m = float(height_cm) / 100.0
        if height_m <= 0:
            return 0.0
        return round(float(weight_kg) / (height_m * height_m), 2)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def get_stroke_risk_level(probability):
    """Use the same risk bands as the supplied StrokeGuard model app."""
    probability = float(probability)
    if probability < 0.30:
        return "LOW RISK"
    if probability < 0.60:
        return "MODERATE RISK"
    return "HIGH RISK"


def get_stroke_recommendations(probability):
    """Return the recommendations used by the supplied stroke application."""
    probability = float(probability)
    if probability >= 0.60:
        return [
            "Consult a healthcare professional",
            "Monitor blood pressure regularly",
            "Follow a heart-healthy diet",
            "Avoid smoking",
            "Exercise consistently",
        ]
    if probability >= 0.30:
        return [
            "Improve daily lifestyle habits",
            "Reduce processed foods",
            "Maintain a healthy weight",
            "Schedule regular health checkups",
        ]
    return [
        "Continue healthy habits",
        "Stay physically active",
        "Maintain balanced nutrition",
    ]


def run_stroke_risk_prediction(
    model,
    *,
    age,
    gender,
    hypertension,
    heart_disease,
    ever_married,
    work_type,
    residence_type,
    avg_glucose,
    weight,
    height,
    smoking_status,
):
    """Run the supplied stroke_model.pkl using its original feature schema."""
    if model is None:
        raise FileNotFoundError(
            "stroke_model.pkl was not loaded. Place stroke_model.pkl beside app.py."
        )

    gender_value = 1 if gender == "Male" else 0
    hypertension_value = 1 if hypertension == "Yes" else 0
    heart_value = 1 if heart_disease == "Yes" else 0
    married_value = 1 if ever_married == "Yes" else 0

    work_mapping = {
        "Government Job": 0,
        "Children": 1,
        "Private": 2,
        "Self-employed": 3,
        "Never Worked": 4,
    }
    smoking_mapping = {
        "never smoked": 0,
        "formerly smoked": 1,
        "smokes": 2,
    }

    bmi = calculate_stroke_bmi(weight, height)

    input_data = pd.DataFrame(
        [[
            gender_value,
            age,
            hypertension_value,
            heart_value,
            married_value,
            work_mapping[work_type],
            1 if residence_type == "Urban" else 0,
            avg_glucose,
            bmi,
            smoking_mapping[smoking_status],
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
            "smoking_status",
        ],
    )

    prediction = int(model.predict(input_data)[0])
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(input_data)[0][1])
    else:
        # Some estimators expose only predict(). Keep the app functional.
        probability = float(prediction)

    return {
        "prediction": prediction,
        "probability": probability,
        "risk_level": get_stroke_risk_level(probability),
        "bmi": bmi,
        "features": input_data.iloc[0].to_dict(),
        "recommendations": get_stroke_recommendations(probability),
    }


def main():
    try:
        with open("static/css/style.css", "r") as f:
            css = f.read()

        st.markdown(
            f"<style>{css}</style>",
            unsafe_allow_html=True
        )

    except FileNotFoundError:
        pass

    sync_clinical_widget_state()

    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary: #4f46e5;
                --primary-dark: #3730a3;
                --primary-soft: #eef0ff;
                --accent: #0ea5b7;
                --accent-soft: #e6f8fa;
                --success: #16a34a;
                --warning: #d97706;
                --danger: #dc2626;
                --text: #1e2433;
                --muted: #6b7280;
                --border: #e7e9f2;
                --surface: #ffffff;
                --page: #f6f7fb;
                --shadow-sm: 0 2px 8px rgba(30, 36, 51, 0.05);
                --shadow: 0 12px 30px rgba(30, 36, 51, 0.08);
                --radius: 18px;
            }

            html, body, [class*="css"] {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            }

            h1, h2, h3, h4, .section-header {
                font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
            }

            .stApp {
                background:
                    radial-gradient(1200px 500px at 10% -5%, #eef1ff 0%, rgba(238,241,255,0) 60%),
                    radial-gradient(1000px 500px at 100% 0%, #e7fbfd 0%, rgba(231,251,253,0) 55%),
                    var(--page);
            }

            .block-container {
                max-width: 1480px;
                padding-top: 2.2rem;
                padding-bottom: 3.5rem;
            }

            .header-meta {
                display: flex;
                align-items: center;
                justify-content: flex-end;
                gap: 22px;
                height: 100%;
            }

            .header-meta-item {
                display: flex;
                align-items: center;
                gap: 10px;
            }

            .header-meta-icon {
                width: 38px;
                height: 38px;
                border-radius: 10px;
                background: var(--primary-soft);
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.05rem;
                flex-shrink: 0;
            }

            .header-meta-title {
                font-size: 0.78rem;
                color: var(--muted);
                line-height: 1.2;
                font-weight: 500;
            }

            .header-meta-value {
                font-size: 0.92rem;
                color: var(--text);
                font-weight: 700;
                line-height: 1.25;
            }

            .header-meta-value b {
                color: var(--primary-dark);
            }

            div[data-testid="stHorizontalBlock"]:has(.single-header-marker) {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 18px;
                padding: 16px 20px;
                margin-bottom: 22px;
                box-shadow: var(--shadow-sm);
                align-items: center;
            }

            div[data-testid="stHorizontalBlock"]:has(.single-header-marker)
            > div[data-testid="stColumn"] {
                background: transparent;
                border: none;
                border-radius: 0;
                padding: 0 12px;
                box-shadow: none;
            }

            .icon-blue { background: #dbeafe !important; }
            .icon-indigo { background: #ede9fe !important; }
            .icon-teal { background: #ccfbf1 !important; }
            .icon-amber { background: #fef3c7 !important; }
            .icon-green { background: #d1fae5 !important; }
            .icon-orange { background: #ffedd5 !important; }
            .icon-pink { background: #fce7f3 !important; }

            div[data-testid="stColumn"] {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: var(--radius);
                padding: 20px 20px 16px 20px;
                box-shadow: var(--shadow-sm);
                transition: box-shadow 0.2s ease;
            }

            div[data-testid="stColumn"] div[data-testid="stColumn"] {
                background: transparent !important;
                border: none !important;
                border-radius: 0 !important;
                box-shadow: none !important;
                padding: 0 !important;
            }

            div[data-testid="stColumn"] div[data-testid="stHorizontalBlock"] {
                gap: 14px;
            }

            div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:empty {
                background: transparent;
                border: none;
                box-shadow: none;
                padding: 0;
            }

            .section-header {
                color: var(--primary-dark);
                font-size: 1.15rem;
                font-weight: 800;
                display: flex;
                align-items: center;
                gap: 10px;
                padding-bottom: 12px;
                margin-bottom: 14px;
                border-bottom: 1px solid var(--border);
                letter-spacing: -0.01em;
            }

            .section-header span {
                background: var(--primary-soft);
                border-radius: 10px;
                padding: 6px 9px;
                font-size: 1.05rem;
            }

            div[data-baseweb="input"] > div,
            div[data-baseweb="select"] > div,
            textarea,
            .stTextInput input,
            .stNumberInput input {
                background: #fbfbfe !important;
                border: 1px solid #e2e5f0 !important;
                border-radius: 12px !important;
                color: var(--text) !important;
            }

            div[data-baseweb="select"] * {
                color: var(--text) !important;
                -webkit-text-fill-color: var(--text) !important;
            }

            div[data-baseweb="select"] > div > div {
                background: #fbfbfe !important;
                color-scheme: light;
            }

            div[data-baseweb="input"] input {
                background: #fbfbfe !important;
                color-scheme: light;
            }

            div[data-baseweb="select"] svg {
                fill: var(--text) !important;
            }

            textarea:focus,
            .stTextInput input:focus,
            .stNumberInput input:focus {
                border-color: var(--primary) !important;
                box-shadow: 0 0 0 4px rgba(79, 70, 229, 0.12) !important;
            }

            label, .stMarkdown, p {
                color: var(--text);
            }

            div.stButton > button,
            div[data-testid="stDownloadButton"] > button {
                border-radius: 12px;
                min-height: 46px;
                border: 1.5px solid #d9d4fb;
                font-weight: 700;
                letter-spacing: 0.01em;
                background: #ede9fe;
                color: var(--primary-dark);
                box-shadow: var(--shadow-sm);
                transition: all 0.18s ease;
            }

            div.stButton > button:hover,
            div[data-testid="stDownloadButton"] > button:hover {
                transform: translateY(-2px);
                background: #e0d9fd;
                border-color: #c7bdfb;
                box-shadow: 0 8px 18px rgba(79, 70, 229, 0.16);
            }

            div.stButton > button[kind="primary"],
            div[data-testid="stDownloadButton"] > button {
                background: #dcd8ff;
                color: #2f267f !important;
                border: 1.5px solid #bdb5ff;
                box-shadow: 0 8px 18px rgba(79, 70, 229, 0.24);
            }

            div.stButton > button[kind="primary"] p,
            div[data-testid="stDownloadButton"] > button p {
                color: #2f267f !important;
            }

            section[data-testid="stFileUploaderDropzone"] {
                background: #fbfbfe;
                border: 1.5px dashed #cdd3e6;
                border-radius: 14px;
                padding: 18px;
            }

            div[data-testid="stMetric"],
            div[data-testid="stAlert"] {
                border-radius: 14px;
                border: 1px solid var(--border);
                box-shadow: var(--shadow-sm);
            }

            div[data-testid="stMetric"] {
                background: linear-gradient(135deg, #ffffff, #f7f7ff);
                padding: 16px;
            }

            .diagnostic-results-section {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: var(--radius);
                padding: 22px;
                margin-top: 14px;
                box-shadow: var(--shadow);
            }

            .summary-box {
                background: #eef2ff;
                border-radius: 12px;
                padding: 16px;
                color: #1e1b4b;
                margin-top: 14px;
                border-left: 4px solid #4f46e5;
            }


            section[data-testid="stSidebar"] {
                background: #ffffff;
                border-right: 1px solid var(--border);
            }

            section[data-testid="stSidebar"] > div {
                padding-top: 1.1rem;
            }

            .sidebar-brand {
                padding: 8px 4px 18px 4px;
                border-bottom: 1px solid var(--border);
                margin-bottom: 14px;
            }

            .sidebar-brand-title {
                font-family: 'Plus Jakarta Sans', sans-serif;
                font-size: 1.2rem;
                font-weight: 800;
                color: var(--primary-dark);
            }

            .sidebar-brand-subtitle {
                color: var(--muted);
                font-size: 0.8rem;
                margin-top: 4px;
            }

            .sidebar-help {
                margin-top: 22px;
                padding: 13px;
                background: #f7f7ff;
                border: 1px solid var(--border);
                border-radius: 12px;
                color: var(--muted);
                font-size: 0.78rem;
                line-height: 1.45;
            }

            section[data-testid="stSidebar"] div[role="radiogroup"] {
                gap: 7px;
            }

            section[data-testid="stSidebar"] div[role="radiogroup"] label {
                background: #fbfbfe;
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 11px 14px;
                transition: all 0.18s ease;
                display: flex;
                align-items: center;
                cursor: pointer;
            }

            section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
                background: var(--primary-soft);
                border-color: #c9c5ff;
                transform: translateX(2px);
            }

            section[data-testid="stSidebar"] div[role="radiogroup"] input[type="radio"] {
                position: absolute;
                opacity: 0;
                width: 0;
                height: 0;
                pointer-events: none;
            }

            section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
                background: linear-gradient(135deg, var(--primary), var(--primary-dark)) !important;
                border-color: var(--primary) !important;
                box-shadow: 0 6px 14px rgba(79, 70, 229, 0.28);
                transform: translateX(2px);
            }

            section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p {
                color: #ffffff !important;
                font-weight: 700 !important;
            }

            section[data-testid="stSidebar"] div[role="radiogroup"] label p {
                font-size: 0.92rem !important;
                font-weight: 600 !important;
                color: var(--text) !important;
                margin: 0 !important;
                letter-spacing: 0.1px;
            }

            div[data-testid="stRadio"] {
                width: 100% !important;
                overflow: hidden;
            }

            div[data-testid="stRadio"] div[role="radiogroup"] {
                gap: 8px;
                flex-wrap: wrap;
                row-gap: 8px;
                width: 100%;
                max-width: 100%;
            }

            div[data-testid="stRadio"] div[role="radiogroup"] label {
                background: #fbfbfe;
                border: 1px solid #e2e5f0;
                border-radius: 10px;
                padding: 6px 10px;
                margin: 0 !important;
                box-sizing: border-box;
                max-width: 100%;
                flex: 0 1 auto;
                transition: all 0.18s ease;
            }

            div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
                background: var(--primary-soft);
                border-color: #c9c5ff;
            }

            div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
                background: var(--primary-soft) !important;
                border-color: var(--primary) !important;
            }

            div[data-testid="stRadio"] input[type="radio"] {
                accent-color: var(--primary) !important;
            }

            div[data-testid="stRadio"] div[role="radiogroup"] label p {
                font-size: 0.9rem !important;
                font-weight: 600 !important;
                color: var(--text) !important;
                white-space: nowrap !important;
            }

            .footer {
                margin-top: 30px;
                padding-top: 16px;
                border-top: 1px solid var(--border);
                text-align: center;
                color: var(--muted);
                font-size: 0.85rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    file_handler = FileHandler()
    patient_manager = PatientManager()

    diagnostic_agent = DiagnosticAgent(
        api_key=settings.OPENAI_API_KEY
    )

    try:
        breast_cancer_classifier = load_breast_cancer_classifier()
    except Exception as error:
        breast_cancer_classifier = None
        _breast_cancer_load_error = str(error)
    else:
        _breast_cancer_load_error = None

    readmission_predictor = get_readmission_predictor_instance()
    safety_agent = SafetyAgent()
    medication_agent = MedicationAgent(api_key=settings.OPENAI_API_KEY)
    heart_analytics = HeartDiseaseAnalytics()
    @st.cache_resource
    def load_stroke_model():
        """Load the standalone stroke-risk model without affecting other agents."""
        try:
            model_path = PROJECT_DIR / "stroke_agent" / "stroke_model.pkl"
            return joblib.load(model_path)
        except Exception as error:
            print(f"[Stroke] Could not load {PROJECT_DIR / 'stroke_agent' / 'stroke_model.pkl'}: {error}")
            return None

    stroke_model = load_stroke_model()
    @st.cache_resource
    def load_rag_service():
        try:
            return get_rag_service()
        except Exception as error:
            print(f"[RAG] Could not load StatPearls RAG service: {error}")
            return None

    rag_service = load_rag_service()

    try:
        clinical_note_model = (
            load_clinical_note_model()
        )

    except Exception as error:
        clinical_note_model = None

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-brand-title">🏥 Clinical Dashboard</div>
                <div class="sidebar-brand-subtitle">Smart Clinic Diagnostic Agent</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        nav_icons = {
            "Patient Information": "👤",
            "Medical History": "📋",
            "Clinical Examination": "🩺",
            "Clinical Description and Notes": "📝",
            "Readmission Analysis": "📊",
            "Stroke Risk Assessment": "🧠",
            "Medication Advisor": "💊",
            "Diagnostic Report": "📑",
        }

        selected_section = st.radio(
            "Dashboard Navigation",
            list(nav_icons.keys()),
            format_func=lambda option: f"{nav_icons.get(option, '')}  {option}",
            key="dashboard_navigation",
            label_visibility="collapsed"
        )

        st.markdown(
            """
            <div class="sidebar-help">
                Select a section to review or update the patient's clinical record.
            </div>
            """,
            unsafe_allow_html=True
        )

    if selected_section == "Patient Information":
        st.markdown(
            """
            <div class="section-header">
                <span class="icon-blue">👤</span>Patient Information
            </div>
            """,
            unsafe_allow_html=True
        )

        p_row1_col1, p_row1_col2, p_row1_col3 = st.columns(3)

        with p_row1_col1:
            patient_id = st.text_input(
                "🆔 Patient ID",
                value=st.session_state.patient_id,
                key="patient_id_input"
            )
            st.session_state.patient_id = patient_id

        with p_row1_col2:
            patient_name = st.text_input(
                "👤 Patient Name",
                value=st.session_state.patient_name,
                key="patient_name_input"
            )
            st.session_state.patient_name = patient_name

        with p_row1_col3:
            patient_age = st.number_input(
                "🎂 Age",
                min_value=0,
                max_value=150,
                value=st.session_state.patient_age,
                key="patient_age_input"
            )
            st.session_state.patient_age = patient_age

        st.markdown("<br>", unsafe_allow_html=True)

        p_row2_col1, p_row2_col2, p_row2_col3 = st.columns(3)

        with p_row2_col1:
            gender_options = ["Male", "Female"]
            patient_gender = st.selectbox(
                "⚧ Gender",
                gender_options,
                index=gender_options.index(
                    st.session_state.patient_gender
                ),
                key="patient_gender_input"
            )
            st.session_state.patient_gender = patient_gender

        with p_row2_col2:
            patient_phone = st.text_input(
                "📞 Phone Number",
                value=st.session_state.patient_phone,
                placeholder="e.g. 03001234567",
                key="patient_phone_input"
            )
            st.session_state.patient_phone = patient_phone

        with p_row2_col3:
            patient_cnic = st.text_input(
                "🪪 CNIC Number",
                value=st.session_state.patient_cnic,
                placeholder="e.g. 12345-1234567-1",
                key="patient_cnic_input"
            )
            st.session_state.patient_cnic = patient_cnic

        p_row3_col1, p_row3_col2, p_row3_col3 = st.columns(3)

        with p_row3_col1:
            patient_email = st.text_input(
                "✉️ Email",
                value=st.session_state.patient_email,
                placeholder="e.g. patient@example.com",
                key="patient_email_input"
            )
            st.session_state.patient_email = patient_email

        with p_row3_col2:
            patient_address = st.text_input(
                "🏠 Address",
                value=st.session_state.patient_address,
                placeholder="e.g. House 12, Main Street, near City Hospital",
                key="patient_address_input",
                help="Enter the patient's address and a nearby landmark if available."
            )
            st.session_state.patient_address = patient_address

        with p_row3_col3:
            emergency_contact_phone = st.text_input(
                "🚨 Emergency Contact Phone Number",
                value=st.session_state.emergency_contact_phone,
                placeholder="e.g. 03001234567",
                key="emergency_contact_phone_input"
            )
            st.session_state.emergency_contact_phone = emergency_contact_phone

        p_row4_col1, p_row4_col2, p_row4_col3 = st.columns(3)

        with p_row4_col1:
            emergency_contact_relationship = st.text_input(
                "🤝 Emergency Contact Relationship",
                value=st.session_state.emergency_contact_relationship,
                placeholder="e.g. Father, Mother, Brother, Friend or Other",
                key="emergency_contact_relationship_input"
            )
            st.session_state.emergency_contact_relationship = emergency_contact_relationship

        with p_row4_col2:
            member_type_options = [
                "Regular",
                "Premium",
                "Corporate",
                "Insurance",
                "Other"
            ]
            member_type = st.selectbox(
                "👥 Member Type",
                member_type_options,
                index=member_type_options.index(st.session_state.member_type)
                if st.session_state.member_type in member_type_options else 0,
                key="member_type_input"
            )
            st.session_state.member_type = member_type

        with p_row4_col3:
            default_dob = st.session_state.patient_date_of_birth
            patient_date_of_birth = st.date_input(
                "📅 Date of Birth",
                value=default_dob,
                min_value=datetime(1900, 1, 1).date(),
                max_value=datetime.now().date(),
                key="patient_date_of_birth_input"
            )
            st.session_state.patient_date_of_birth = patient_date_of_birth

        p_row5_col1, p_row5_col2, p_row5_col3 = st.columns(3)

        with p_row5_col1:
            payment_method_options = [
                "Cash",
                "Credit Card",
                "Debit Card",
                "Bank Transfer",
                "JazzCash",
                "EasyPaisa",
                "Insurance",
                "Other"
            ]
            payment_method = st.selectbox(
                "💳 Payment Method",
                payment_method_options,
                index=payment_method_options.index(st.session_state.payment_method)
                if st.session_state.payment_method in payment_method_options else 0,
                key="payment_method_input"
            )
            st.session_state.payment_method = payment_method

        with p_row5_col2:
            selected_date = st.date_input(
                "🗓️ Select Date",
                value=st.session_state.selected_date,
                key="selected_date_input"
            )
            st.session_state.selected_date = selected_date

        with p_row5_col3:
            # Streamlit resets a widget's own session_state value whenever
            # that widget wasn't rendered on the previous run (e.g. the user
            # was on a different tab), so this field would otherwise appear
            # blank even though the saved last visit is still available.
            # Restore it here, right before the widget is created, whenever
            # it's been cleared.
            _effective_last_visit = get_effective_last_visit()
            if _effective_last_visit and not st.session_state.get("last_visit_input"):
                st.session_state.last_visit_input = _effective_last_visit

            last_visit_val = st.text_input(
                "🕓 Last Visit",
                value=get_effective_last_visit(),
                key="last_visit_input",
                placeholder="e.g. 12 May 2026",
                help="Date of the patient's last visit."
            )
            st.session_state.last_visit = last_visit_val

        st.info(
            "📋 Detailed medical history (health issues, past surgeries, "
            "allergies, medications, family history) can be entered on the "
            "**Medical History** tab in the sidebar."
        )

        if patient_id and patient_name:
            patient_manager.create_patient(
                patient_id,
                patient_name,
                patient_age,
                patient_gender
            )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="section-header">
                <span class="icon-pink">❤️</span>Vitals
            </div>
            """,
            unsafe_allow_html=True
        )

        vital_row1_col1, vital_row1_col2, vital_row1_col3 = st.columns(3)

        with vital_row1_col1:
            st.markdown("#### ↕️ Height")
            st.caption("Feet and inches")

            height_col1, height_col2 = st.columns(2)

            with height_col1:
                patient_height_feet = st.number_input(
                    "Feet",
                    min_value=0,
                    max_value=8,
                    value=int(st.session_state.get("patient_height_feet", 0)),
                    step=1,
                    key="patient_height_feet_input"
                )

            with height_col2:
                patient_height_inches = st.number_input(
                    "Inches",
                    min_value=0,
                    max_value=11,
                    value=int(st.session_state.get("patient_height_inches", 0)),
                    step=1,
                    key="patient_height_inches_input"
                )

            st.session_state.patient_height_feet = patient_height_feet
            st.session_state.patient_height_inches = patient_height_inches

        with vital_row1_col2:
            st.markdown("#### ⚖️ Weight")
            st.caption("Body weight")

            patient_weight = st.number_input(
                "Weight (kg)",
                min_value=0.0,
                max_value=500.0,
                value=float(st.session_state.patient_weight),
                step=0.1,
                key="patient_weight_input"
            )
            st.session_state.patient_weight = patient_weight

        with vital_row1_col3:
            st.markdown("#### 💙 Blood Pressure")
            st.caption("Blood pressure reading (mmHg)")

            blood_pressure = st.text_input(
                "Blood Pressure (mmHg)",
                value=st.session_state.blood_pressure,
                placeholder="e.g. 120/80",
                key="blood_pressure_input"
            )
            st.session_state.blood_pressure = blood_pressure

        st.markdown("<br>", unsafe_allow_html=True)

        vital_row2_col1, vital_row2_col2, vital_row2_col3 = st.columns(3)

        with vital_row2_col1:
            st.markdown("#### 🌡️ Temperature")
            st.caption("Body temperature")

            temperature_col1, temperature_col2 = st.columns([3, 3])

            with temperature_col1:
                body_temperature = st.number_input(
                    "Reading",
                    min_value=0.0,
                    max_value=120.0,
                    value=float(st.session_state.body_temperature),
                    step=0.1,
                    key="body_temperature_input"
                )

            with temperature_col2:
                temperature_scale = st.radio(
                    "Scale",
                    ["°F", "°C"],
                    index=0 if st.session_state.temperature_scale == "°F" else 1,
                    key="temperature_scale_input",
                    horizontal=True,
                )

            st.session_state.body_temperature = body_temperature
            st.session_state.temperature_scale = temperature_scale

        with vital_row2_col2:
            st.markdown("#### 🫁 Oxygen (SpO₂)")
            st.caption("Oxygen saturation percentage")

            oxygen_saturation = st.number_input(
                "SpO₂ (%)",
                min_value=0,
                max_value=100,
                value=int(st.session_state.oxygen_saturation),
                step=1,
                key="oxygen_saturation_input"
            )
            st.session_state.oxygen_saturation = oxygen_saturation

        with vital_row2_col3:
            st.markdown("#### 💓 Heart Rate")
            st.caption("Beats per minute")

            heart_rate = st.number_input(
                "Heart Rate (bpm)",
                min_value=0,
                max_value=300,
                value=int(st.session_state.heart_rate),
                step=1,
                key="heart_rate_input"
            )
            st.session_state.heart_rate = heart_rate

        st.markdown("<br>", unsafe_allow_html=True)

        vital_row3_col1, vital_row3_col2, vital_row3_col3 = st.columns(3)

        with vital_row3_col1:
            st.markdown("#### 💨 Respiratory Rate")
            st.caption("Breaths per minute")

            respiratory_rate = st.number_input(
                "Respiratory Rate (breaths/min)",
                min_value=0,
                max_value=100,
                value=int(st.session_state.respiratory_rate),
                step=1,
                key="respiratory_rate_input"
            )
            st.session_state.respiratory_rate = respiratory_rate

        with vital_row3_col2:
            st.markdown("#### 💧 Blood Glucose")
            st.caption("Blood glucose level")

            blood_glucose = st.number_input(
                "Blood Glucose (mg/dL)",
                min_value=0.0,
                max_value=1000.0,
                value=float(st.session_state.blood_glucose),
                step=0.1,
                key="blood_glucose_input"
            )
            st.session_state.blood_glucose = blood_glucose

        with vital_row3_col3:
            st.markdown("#### 📏 Waist Circumference")
            st.caption("Waist measurement")

            waist_circumference = st.number_input(
                "Waist Circumference (cm)",
                min_value=0.0,
                max_value=300.0,
                value=float(st.session_state.waist_circumference),
                step=0.1,
                key="waist_circumference_input"
            )
            st.session_state.waist_circumference = waist_circumference

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("#### 🩸 Blood Group")
        st.caption("Select the patient's blood group")

        blood_group_options = [
            "A+",
            "A-",
            "B+",
            "B-",
            "AB+",
            "AB-",
            "O+",
            "O-"
        ]

        blood_group = st.selectbox(
            "Blood Group",
            blood_group_options,
            index=blood_group_options.index(st.session_state.blood_group)
            if st.session_state.blood_group in blood_group_options else 0,
            key="blood_group_input"
        )
        st.session_state.blood_group = blood_group

        if st.button(
            "💾 SAVE INFORMATION",
            type="primary",
            key="save_patient_information_btn",
            use_container_width=True
        ):
            persist_all_widget_state()
            save_patient_info_snapshot()
            if patient_id and patient_name:
                patient_manager.create_patient(
                    patient_id,
                    patient_name,
                    patient_age,
                    patient_gender
                )
                st.success("Saved successfully.")
            else:
                st.warning(
                    "Please enter both Patient ID and Patient Name before saving."
                )

        st.markdown("<br>", unsafe_allow_html=True)

    elif selected_section == "Medical History":
        st.markdown(
            """
            <div class="section-header">
                <span class="icon-blue">📋</span>Medical History
            </div>
            """,
            unsafe_allow_html=True
        )

        mh_row1_col1, mh_row1_col2 = st.columns(2)

        with mh_row1_col1:
            chronic_conditions = st.text_area(
                "🩺 Health Issues",
                value=st.session_state.chronic_conditions,
                key="chronic_conditions_input",
                height=110,
                placeholder="e.g. Diabetes, High Blood Pressure, Asthma",
                help="Long-term or ongoing medical conditions."
            )
            st.session_state.chronic_conditions = chronic_conditions

        with mh_row1_col2:
            past_surgeries = st.text_area(
                "🔪 Past Surgeries",
                value=st.session_state.past_surgeries,
                key="past_surgeries_input",
                height=110,
                placeholder="e.g. Appendix Removal (2018), C-Section (2021)",
                help="Previous surgical procedures with approximate dates."
            )
            st.session_state.past_surgeries = past_surgeries

        mh_row2_col1, mh_row2_col2 = st.columns(2)

        with mh_row2_col1:
            known_allergies = st.text_area(
                "⚠️ Known Allergies",
                value=st.session_state.known_allergies,
                key="known_allergies_input",
                height=110,
                placeholder="e.g. Penicillin, Peanuts, Dust",
                help="Drug, food, or environmental allergies."
            )
            st.session_state.known_allergies = known_allergies

        with mh_row2_col2:
            current_medications = st.text_area(
                "💊 Current Medications",
                value=st.session_state.current_medications,
                key="current_medications_input",
                height=110,
                placeholder="e.g. Panadol 500mg, twice daily",
                help="Medications the patient is currently taking, with dosage if known."
            )
            st.session_state.current_medications = current_medications

        mh_row3_col1, mh_row3_col2 = st.columns(2)

        with mh_row3_col1:
            family_history = st.text_area(
                "👪 Family History",
                value=st.session_state.family_history,
                key="family_history_input",
                height=110,
                placeholder="e.g. Father - Heart Disease, Mother - Diabetes",
                help="Relevant medical conditions in the patient's immediate family."
            )
            st.session_state.family_history = family_history

        with mh_row3_col2:
            immunization_history = st.text_area(
                "💉 Immunization History",
                value=st.session_state.immunization_history,
                key="immunization_history_input",
                height=110,
                placeholder="e.g. Polio Vaccine, Tetanus (2023), Flu Shot (2025)",
                help="Vaccines the patient has received."
            )
            st.session_state.immunization_history = immunization_history

        st.markdown("<br>", unsafe_allow_html=True)

        additional_notes = st.text_area(
            "📝 Additional Notes",
            value=st.session_state.medical_history,
            key="medical_history_input",
            height=100,
            placeholder="Any other relevant medical history, previous visit notes, etc.",
            help="Anything not covered by the fields above."
        )
        st.session_state.medical_history = additional_notes

        if st.button(
            "💾 SAVE MEDICAL HISTORY",
            type="primary",
            key="save_medical_history_btn",
            use_container_width=True
        ):
            save_medical_history_snapshot()
            st.success("Medical history saved successfully.")

        st.markdown("<br>", unsafe_allow_html=True)

    elif selected_section == "Clinical Examination":
        st.markdown(
            """
            <div class="section-header">
                <span class="icon-teal">🤒</span>Symptoms
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("**Select common symptoms**")

        symptom_cols = st.columns(4)
        # Streamlit drops a widget's session_state entry whenever that widget
        # isn't re-created on a run (e.g. the user is on a different tab), so
        # symptom_checkbox_* keys don't survive tab switches on their own.
        # Fall back to the last-known / saved selections so the checkboxes
        # come back checked when this tab is revisited. Prefer the live,
        # continuously-updated selection over the saved snapshot -- the
        # snapshot is only taken when "Save Symptoms" is clicked, so relying
        # on it first would make an unsaved uncheck silently reappear as
        # checked the next time this tab is opened.
        previously_selected_symptoms = (
            st.session_state.get("selected_common_symptoms")
            or st.session_state.get("saved_selected_common_symptoms")
            or []
        )
        for symptom_index, symptom_name in enumerate(COMMON_SYMPTOMS):
            with symptom_cols[symptom_index % 4]:
                checkbox_key = f"symptom_checkbox_{symptom_index}"
                default_checked = symptom_name in previously_selected_symptoms
                st.checkbox(
                    symptom_name,
                    value=default_checked,
                    key=checkbox_key,
                    on_change=_on_symptom_checkbox_change,
                    args=(symptom_index,),
                )

        # Belt-and-braces: also resync from whatever checkbox keys are
        # actually present this run (covers the very first render, before
        # any on_change has fired).
        selected_common_symptoms = [
            symptom_name
            for symptom_index, symptom_name in enumerate(COMMON_SYMPTOMS)
            if st.session_state.get(
                f"symptom_checkbox_{symptom_index}",
                symptom_name in previously_selected_symptoms,
            )
        ]
        st.session_state.selected_common_symptoms = selected_common_symptoms

        st.markdown("<br>", unsafe_allow_html=True)

        # Streamlit resets a widget's own session_state value whenever that
        # widget wasn't rendered on the previous run (e.g. the user was on a
        # different tab), so this text area would otherwise appear blank
        # even though the saved text is still available. Restore it here,
        # right before the widget is created, whenever it's been cleared.
        _effective_other_symptoms = get_effective_other_symptoms()
        if _effective_other_symptoms and not st.session_state.get("symptoms_input"):
            st.session_state.symptoms_input = _effective_other_symptoms

        symptoms_val = st.text_area(
            "Other Symptoms / Additional Details",
            value=get_effective_other_symptoms(),
            key="symptoms_input",
            height=100,
            placeholder="e.g. Mild dizziness on standing, symptoms worse at night",
            help="Add any symptoms not listed above, or extra detail (duration, severity, etc.)."
        )
        st.session_state.symptoms = symptoms_val

        if st.button(
            "💾 SAVE SYMPTOMS",
            type="primary",
            key="save_symptoms_btn",
            use_container_width=True
        ):
            persist_all_widget_state()
            save_symptoms_snapshot()
            st.success("Symptoms saved successfully.")

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="section-header">
                <span class="icon-teal">🖼️</span>Medical Image Analysis
            </div>
            """,
            unsafe_allow_html=True
        )

        uploaded_images = st.file_uploader(
            "Upload Imaging Scans (X-Ray, MRI, CT)",
            type=["png", "jpg", "jpeg", "webp", "bmp", "tiff", "gif", "dicom", "dcm"],
            accept_multiple_files=True,
            key=f"image_uploader_{st.session_state.uploader_version}"
        )

        if uploaded_images:
            for image_file in uploaded_images:
                file_signature = f"image::{image_file.name}::{image_file.size}"

                if file_signature in st.session_state.uploaded_files_info:
                    continue

                saved_path, save_msg = file_handler.save_uploaded_file(
                    image_file,
                    folder="images",
                    patient_id=st.session_state.patient_id
                )

                if not saved_path and save_msg == "File already exists":
                    existing_record = _get_existing_file_record(
                        file_handler, image_file.name
                    )
                    if existing_record and os.path.exists(
                        existing_record.get("file_path", "")
                    ):
                        saved_path = existing_record["file_path"]
                    else:
                        st.error(
                            f"{image_file.name} was already uploaded before, but its "
                            "saved copy could not be found on disk. Rename the file "
                            "slightly and re-upload, or delete medical_data.db to reset."
                        )

                if saved_path:
                    try:
                        analysis = diagnostic_agent.analyze_image(saved_path)
                    except Exception as e:
                        analysis = f"[ERROR analyzing image: {e}]"
                    st.session_state.medical_analysis_image_findings_input += (
                        f"\n\n--- Image: {image_file.name} ---\n{analysis}"
                    )
                    st.session_state.uploaded_files_info.append(file_signature)
                elif save_msg and save_msg != "File already exists":
                    st.error(f"Failed to save {image_file.name}: {save_msg}")

        image_findings = st.text_area(
            "Image Observations / Medical Imaging Findings",
            value=get_effective_image_findings(),
            height=150,
            key="medical_analysis_image_findings_input",
            help="Enter image observations manually or review findings extracted from uploaded scans."
        )
        st.session_state.image_findings = image_findings

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="section-header">
                <span class="icon-teal">🎗️</span>Breast Cancer Ultrasound Classification
            </div>
            """,
            unsafe_allow_html=True
        )

        breast_us_image = None

        if breast_cancer_classifier is None or not breast_cancer_classifier.is_ready:
            st.info(
                "Optional breast ultrasound classification is not installed "
                f"({_breast_cancer_load_error or 'model not loaded'}). "
                "The rest of Full Diagnostics remains available. Add the BUSI "
                "training artifact later to enable this optional module."
            )
        else:
            breast_us_image = st.file_uploader(
                "Upload Breast Ultrasound Scan",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=False,
                key=f"breast_cancer_uploader_{st.session_state.uploader_version}",
                help=(
                    "Upload a single breast ultrasound image for AI-assisted "
                    "benign / malignant / normal classification."
                )
            )

            if breast_us_image is not None:
                breast_us_signature = (
                    f"breast_us::{breast_us_image.name}::{breast_us_image.size}"
                )

                if breast_us_signature != st.session_state.last_breast_us_signature:
                    saved_breast_us_path, save_message = file_handler.save_uploaded_file(
                        breast_us_image,
                        folder="images",
                        patient_id=st.session_state.patient_id,
                    )

                    if not saved_breast_us_path and save_message == "File already exists":
                        existing_record = _get_existing_file_record(
                            file_handler, breast_us_image.name
                        )
                        if existing_record and os.path.exists(
                            existing_record.get("file_path", "")
                        ):
                            saved_breast_us_path = existing_record["file_path"]

                    if saved_breast_us_path:
                        try:
                            breast_cancer_prediction = breast_cancer_classifier.predict(
                                saved_breast_us_path
                            )
                            st.session_state.breast_cancer_result = breast_cancer_prediction
                            st.session_state.breast_cancer_image_path = saved_breast_us_path
                            st.session_state.last_breast_us_signature = breast_us_signature
                        except Exception as error:
                            st.error(f"Could not analyze the ultrasound image: {error}")
                    elif save_message and save_message != "File already exists":
                        st.warning(f"Could not save the ultrasound image: {save_message}")

            breast_cancer_result = st.session_state.get("breast_cancer_result")
            if breast_cancer_result:
                result_img_col, result_detail_col = st.columns([1, 1.2])

                with result_img_col:
                    display_path = st.session_state.get("breast_cancer_image_path")
                    if breast_us_image is not None:
                        st.image(
                            breast_us_image,
                            caption="Uploaded Ultrasound Scan",
                            use_container_width=True
                        )
                    elif display_path and os.path.exists(display_path):
                        st.image(
                            display_path,
                            caption="Uploaded Ultrasound Scan",
                            use_container_width=True
                        )

                with result_detail_col:
                    risk_icon = {
                        "low": "🟢",
                        "none": "🟢",
                        "high": "🔴",
                        "unknown": "⚪",
                    }.get(breast_cancer_result.get("risk", "unknown"), "⚪")

                    st.markdown(
                        f"### {risk_icon} {breast_cancer_result['predicted_label']}"
                    )
                    st.markdown(
                        f"**Confidence:** "
                        f"{breast_cancer_result['confidence'] * 100:.1f}%"
                    )
                    st.caption(breast_cancer_result.get("note", ""))

                    st.markdown("**Class Probabilities**")
                    sorted_probs = sorted(
                        breast_cancer_result["probabilities"].items(),
                        key=lambda item: item[1],
                        reverse=True
                    )
                    for class_name, probability in sorted_probs:
                        st.progress(
                            probability,
                            text=f"{class_name.title()}: {probability * 100:.1f}%"
                        )

                def _add_breast_cancer_note_to_findings():
                    result = st.session_state.get("breast_cancer_result")
                    if not result:
                        return
                    current_notes = st.session_state.get(
                        "medical_analysis_image_findings_input", ""
                    )
                    note_text = (
                        f"\n\n--- Breast Ultrasound AI Classification ---\n"
                        f"Prediction: {result['predicted_label']} "
                        f"(confidence: {result['confidence'] * 100:.1f}%)\n"
                        f"{result['note']}"
                    )
                    st.session_state.medical_analysis_image_findings_input = (
                        current_notes + note_text
                    )
                    st.session_state.breast_cancer_note_added_signature = (
                        st.session_state.get("last_breast_us_signature")
                    )

                already_added = (
                    st.session_state.get("breast_cancer_note_added_signature")
                    == st.session_state.get("last_breast_us_signature")
                )
                if already_added:
                    st.caption("✅ Added to Image Observations above.")
                else:
                    st.button(
                        "➕ Add this result to Image Observations",
                        key="add_breast_cancer_note_btn",
                        on_click=_add_breast_cancer_note_to_findings,
                    )

                if breast_cancer_result.get("risk") == "high":
                    st.error(
                        "⚠️ This scan is flagged as suspicious for malignancy by "
                        "the AI model. This is a decision-support tool, not a "
                        "diagnosis — urgent specialist review and biopsy "
                        "correlation are recommended."
                    )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="section-header">
                <span class="icon-amber">📂</span>Clinical Documents & Data
            </div>
            """,
            unsafe_allow_html=True
        )

        uploaded_docs = st.file_uploader(
            "Upload Medical Records, Lab Reports, or Clinical PDFs",
            type=["pdf", "txt", "docx", "csv"],
            accept_multiple_files=True,
            key=f"document_uploader_{st.session_state.uploader_version}"
        )

        if uploaded_docs:
            for doc_file in uploaded_docs:
                file_signature = f"doc::{doc_file.name}::{doc_file.size}"

                if file_signature in st.session_state.uploaded_files_info:
                    continue

                saved_path, save_msg = file_handler.save_uploaded_file(
                    doc_file,
                    folder="documents",
                    patient_id=st.session_state.patient_id
                )

                if not saved_path and save_msg == "File already exists":
                    existing_record = _get_existing_file_record(
                        file_handler, doc_file.name
                    )
                    if existing_record and os.path.exists(
                        existing_record.get("file_path", "")
                    ):
                        saved_path = existing_record["file_path"]
                    else:
                        st.error(
                            f"{doc_file.name} was already uploaded before, but its "
                            "saved copy could not be found on disk. Rename the file "
                            "slightly and re-upload, or delete medical_data.db to reset."
                        )

                if saved_path:
                    try:
                        extracted_text = file_handler.extract_text_from_file(saved_path)
                    except Exception as e:
                        extracted_text = f"[ERROR extracting text: {e}]"
                    st.session_state.medical_analysis_document_findings_input += (
                        f"\n\n--- Document: {doc_file.name} ---\n{extracted_text}"
                    )
                    st.session_state.uploaded_files_info.append(file_signature)
                elif save_msg and save_msg != "File already exists":
                    st.error(f"Failed to save {doc_file.name}: {save_msg}")

        documents_text = st.text_area(
            "Laboratory Results / Clinical Document Findings",
            value=get_effective_document_findings(),
            height=170,
            key="medical_analysis_document_findings_input",
            help="Enter laboratory results manually or review text extracted from uploaded reports."
        )
        st.session_state.document_findings = documents_text

        if st.button(
            "💾 SAVE MEDICAL ANALYSIS",
            type="primary",
            key="save_medical_analysis_btn",
            use_container_width=True
        ):
            persist_all_widget_state()
            save_medical_analysis_snapshot()
            st.success("Medical analysis saved successfully.")

        st.markdown("<br>", unsafe_allow_html=True)

    elif selected_section == "Clinical Description and Notes":
        st.markdown(
            """
            <div class="section-header">
                <span class="icon-indigo">📄</span>Clinical Input
            </div>
            """,
            unsafe_allow_html=True
        )

        # Streamlit resets a widget's own session_state value whenever that
        # widget wasn't rendered on the previous run (e.g. the user was on a
        # different tab), so this field would otherwise appear blank even
        # though the saved chief complaint is still available. Restore it
        # here, right before the widget is created, whenever it's been
        # cleared.
        _effective_chief_complaint = (
            st.session_state.get("saved_chief_complaint")
            or st.session_state.get("chief_complaint", "")
        )
        if _effective_chief_complaint and not st.session_state.get("chief_complaint_input"):
            st.session_state.chief_complaint_input = _effective_chief_complaint

        chief_complaint_val = st.text_input(
            "🗣️ Chief Complaint",
            value=st.session_state.get("chief_complaint", ""),
            key="chief_complaint_input",
            placeholder="e.g. Fever and cough for 3 days",
            help="The main reason the patient came in, in the patient's own words - usually one short line."
        )
        st.session_state.chief_complaint = chief_complaint_val

        if st.button(
            "💾 SAVE CHIEF COMPLAINT",
            type="primary",
            key="save_chief_complaint_btn",
            use_container_width=True
        ):
            persist_all_widget_state()
            save_chief_complaint_snapshot()
            st.success("Chief complaint saved successfully.")

        st.markdown("<br>", unsafe_allow_html=True)

        # Populate the clinical case description and clinical notes from the
        # integrated clinical record when those fields are empty. Existing
        # manual text is never overwritten. This keeps the current UI and
        # SAVE CLINICAL NOTES behavior unchanged while ensuring the two fields
        # are also available in the Diagnostic Report.
        if "pending_generated_clinical_case" in st.session_state:
            pending_case = st.session_state.pop("pending_generated_clinical_case")
            if not st.session_state.get("clinical_case_input", "").strip():
                st.session_state.clinical_case_input = pending_case

        if "pending_generated_clinical_notes" in st.session_state:
            pending_notes = st.session_state.pop("pending_generated_clinical_notes")
            if not st.session_state.get("clinical_notes_input", "").strip():
                st.session_state.clinical_notes_input = pending_notes

        # If no pending generated text exists (for example after switching
        # directly to this section), generate the missing fields from the
        # current Chief Complaint, Medical History, Vitals, Symptoms, and
        # Documents already present in session state.
        current_case = (st.session_state.get("clinical_case_input") or "").strip()
        current_notes = (st.session_state.get("clinical_notes_input") or "").strip()
        if not current_case or not current_notes:
            generated_case, generated_notes = build_integrated_clinical_summary_and_notes()
            if not current_case and generated_case:
                st.session_state.clinical_case_input = generated_case
            if not current_notes and generated_notes:
                st.session_state.clinical_notes_input = generated_notes

        c_col1, c_col2 = st.columns(2)

        with c_col1:
            clinical_case = st.text_area(
                "Clinical Case Description",
                value=st.session_state.get("clinical_case_input", ""),
                height=150,
                key="clinical_case_input",
                help="Enter clinical case details, patient symptoms, medical history, chief complaint, etc."
            )

        with c_col2:
            clinical_notes = st.text_area(
                "Clinical Messages/Notes",
                value=st.session_state.get("clinical_notes_input", ""),
                height=150,
                key="clinical_notes_input",
                help="Enter clinical messages, nursing notes, physician observations..."
            )

        if st.button(
            "💾 SAVE CLINICAL NOTES",
            type="primary",
            key="save_clinical_notes_btn",
            use_container_width=True
        ):
            save_clinical_notes_snapshot()
            st.success("Clinical notes saved successfully.")

        if st.button("🔍 Clinical Recommendations", key="classify_clinical_note_btn"):
            cleaned_note = st.session_state.clinical_notes_input.strip()

            if cleaned_note and len(cleaned_note.split()) >= 3 and clinical_note_model is not None:
                predicted_class = clinical_note_model.predict([cleaned_note])[0]
                predicted_class_key = str(predicted_class).strip().lower()

                recommendations = disease_recommendations.get(
                    predicted_class_key,
                    default_recommendations
                )

                probabilities = clinical_note_model.predict_proba([cleaned_note])[0]
                class_names = clinical_note_model.named_steps["classifier"].classes_

                clinical_results = pd.DataFrame({
                    "Clinical category": class_names,
                    "Probability": probabilities
                }).sort_values(by="Probability", ascending=False).reset_index(drop=True)

                confidence = float(clinical_results.loc[0, "Probability"])

                st.session_state["clinical_classification_result"] = {
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "recommendations": recommendations,
                    "results": clinical_results
                }

                # ---------------------------------------------------------
                # ADDITION ONLY: Heart Disease Analytics
                # Existing Clinical Note Classification above is unchanged.
                # ---------------------------------------------------------
                try:
                    heart_analytics = heart_analytics
                    heart_result = heart_analytics.predict(
                        age=st.session_state.get("patient_age", 0),
                        gender=st.session_state.get("patient_gender", "Male"),
                        blood_pressure=st.session_state.get("blood_pressure", ""),
                        heart_rate=st.session_state.get("heart_rate", 0),
                        blood_glucose=st.session_state.get("blood_glucose", 0),
                        clinical_text=build_combined_clinical_text(),
                        medical_history=build_effective_medical_history_summary(),
                    )
                    st.session_state["heart_disease_result"] = heart_result
                except Exception as error:
                    st.session_state["heart_disease_result"] = {
                        "prediction": "NO",
                        "probability": 0.0,
                        "risk_level": "Unavailable",
                        "description": (
                            "Heart Disease Analytics could not be calculated "
                            f"because of an internal error: {error}"
                        ),
                        "model_used": "Random Forest Classifier",
                    }

        if st.session_state["clinical_classification_result"] is not None:
            classification_result = st.session_state["clinical_classification_result"]
            predicted_class = classification_result["predicted_class"]
            confidence = classification_result["confidence"]
            recommendations = classification_result["recommendations"]

            st.markdown("<br>", unsafe_allow_html=True)
            st.info(f"**Predicted Condition:** {predicted_class.title()} (Confidence: {confidence:.2%})")


            # ---------------------------------------------------------
            # ADDITION ONLY: Heart Disease Analytics display
            # ---------------------------------------------------------
            heart_result = st.session_state.get("heart_disease_result")
            if heart_result:
                st.markdown("### ❤️ Heart Disease Analytics")
                st.markdown(
                    f"**Prediction:** {heart_result.get('prediction', 'NO')}"
                )
                st.markdown(
                    f"**Risk Level:** {heart_result.get('risk_level', 'Unavailable')}"
                )
                st.markdown(
                    f"**Model Confidence:** {float(heart_result.get('probability', 0.0)):.2%}"
                )
                st.info(
                    heart_result.get(
                        "description",
                        "No heart disease analytics description is available.",
                    )
                )

        st.markdown("<br>", unsafe_allow_html=True)

    elif selected_section == "Readmission Analysis":
        st.markdown(
            """
            <div class="section-header">
                <span class="icon-pink">🩺</span>Readmission Prediction
            </div>
            """,
            unsafe_allow_html=True
        )

        # GUARANTEED PERSISTENCE: When the user clicks "Save Readmission
        # Analysis", all values are stored in saved_readmission_inputs
        # (a plain session_state dict that survives all tab switches).
        # Every time the Readmission tab renders, we unconditionally write
        # those saved values back into the widget keys BEFORE the widgets
        # are created, so Streamlit always shows the saved values.
        # ---------------------------------------------------------------
        _saved = st.session_state.get("saved_readmission_inputs", {})
        _restore_map = [
            ("previous_admissions_input",   "previous_admissions",   0),
            ("length_of_stay_input",        "length_of_stay",        1),
            ("emergency_visits_input",       "emergency_visits",      0),
            ("number_of_medications_input", "number_of_medications", 0),
            ("discharge_disposition_input", "discharge_disposition", "Home / Self Care"),
        ]

        for _wk, _ck, _default in _restore_map:
            # ONLY override if the key is missing (e.g. after a tab switch)
            if _wk not in st.session_state:
                if _saved:
                    st.session_state[_wk] = _saved.get(_ck, _default)
                else:
                    st.session_state[_wk] = st.session_state.get(_ck, _default)

        r_col_left, r_col_right = st.columns(2)

        with r_col_left:
            previous_admissions = st.number_input(
                "1. 🏥 Previous Hospital Admissions (Count)",
                min_value=0,
                max_value=100,
                help="Number of prior hospital admissions.",
                key="previous_admissions_input",
                on_change=_save_previous_admissions,
            )

            emergency_visits = st.number_input(
                "3. 🚨 Emergency Visits (Count)",
                min_value=0,
                max_value=100,
                help="Number of emergency department visits.",
                key="emergency_visits_input",
                on_change=_save_emergency_visits,
            )

            discharge_disposition = st.selectbox(
                "5. 🚪 Discharge Disposition",
                options=[
                    "Home / Self Care",
                    "Home Health Care",
                    "Skilled Nursing Facility (SNF)",
                    "Inpatient Rehabilitation Facility (IRF)",
                    "Left Against Medical Advice (AMA)",
                    "Other / Unknown",
                ],
                key="discharge_disposition_input",
                on_change=_save_discharge_disposition,
            )

        with r_col_right:
            length_of_stay = st.number_input(
                "2. ⏱️ Length of Stay (Days)",
                min_value=0,
                max_value=365,
                help="Duration of inpatient stay in days.",
                key="length_of_stay_input",
                on_change=_save_length_of_stay,
            )

            number_of_medications = st.number_input(
                "4. 💊 Number of Medications",
                min_value=0,
                max_value=100,
                help="Total count of active medications the patient is on.",
                key="number_of_medications_input",
                on_change=_save_number_of_medications,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Save button (full width) between inputs and bottom action row ---
        save_inputs_clicked = st.button("💾 SAVE READMISSION ANALYSIS", use_container_width=True)
        if save_inputs_clicked:
            st.session_state.saved_readmission_inputs = {
                "previous_admissions":   st.session_state.get("previous_admissions_input", 0),
                "length_of_stay":        st.session_state.get("length_of_stay_input", 0),
                "emergency_visits":      st.session_state.get("emergency_visits_input", 0),
                "number_of_medications": st.session_state.get("number_of_medications_input", 0),
                "discharge_disposition": st.session_state.get("discharge_disposition_input", "Home / Self Care"),
            }
            st.success("💾 Readmission inputs saved.")

        col_btn1, col_btn2 = st.columns([1.2, 1])

        with col_btn1:
            analyze_clicked = st.button("🚀 ANALYZE CLINICAL CASE", type="primary", use_container_width=True)

        with col_btn2:
            clear_clicked = st.button("🗑️ CLEAR ALL INPUTS", use_container_width=True)

        if clear_clicked:
            clear_all_session_inputs()
            st.success("All inputs, saved data, predictions, and reports have been cleared.")
            st.rerun()


        if analyze_clicked:
            persist_all_widget_state()
            with st.spinner("Analyzing..."):
                try:
                    # Always read directly from the live widget keys (*_input)
                    # which are guaranteed to hold the user's current values
                    # in the same render cycle as the Analyze button click.
                    # Do NOT fall back to canonical keys first — they may be
                    # stale from a previous run if the user changed a field
                    # without triggering an on_change callback.
                    prev_adm  = st.session_state.get("previous_admissions_input",   st.session_state.get("previous_admissions",   0))
                    los       = st.session_state.get("length_of_stay_input",        st.session_state.get("length_of_stay",        0))
                    em_visits = st.session_state.get("emergency_visits_input",       st.session_state.get("emergency_visits",       0))
                    meds      = st.session_state.get("number_of_medications_input", st.session_state.get("number_of_medications", 0))
                    disp      = st.session_state.get("discharge_disposition_input", st.session_state.get("discharge_disposition", "Home / Self Care"))

                    # Also update canonical keys so the rest of the pipeline
                    # and the display section see the same values.
                    st.session_state["previous_admissions"]   = prev_adm
                    st.session_state["length_of_stay"]        = los
                    st.session_state["emergency_visits"]      = em_visits
                    st.session_state["number_of_medications"] = meds
                    st.session_state["discharge_disposition"] = disp

                    # Debug: confirm exact values sent to agent (visible in terminal)
                    print(f"[READMISSION ANALYZE] prev={prev_adm}, los={los}, ed={em_visits}, meds={meds}, disp={disp}")

                    readmission_data = readmission_predictor.predict(
                        previous_admissions=prev_adm,
                        length_of_stay=los,
                        emergency_visits=em_visits,
                        number_of_medications=meds,
                        discharge_disposition=disp,
                    )
                    print(f"[READMISSION RESULT] {readmission_data.get('probability')}% {readmission_data.get('risk_level')}")
                except Exception as error:
                    readmission_data = None
                    st.error(f"Unable to calculate readmission risk: {error}")

                # Only accept a genuine AI response dict.  Do NOT substitute
                # hard-coded risk numbers — if the agent is unavailable,
                # surface that clearly rather than showing fake data.
                if not isinstance(readmission_data, dict):
                    readmission_data = {
                        "prediction": "UNKNOWN",
                        "probability": 0,
                        "risk_level": "Unknown",
                        "reason": "AI Readmission Agent is currently unavailable. Please check your API key configuration.",
                        "summary": "Could not generate assessment — the AI agent is not reachable at this time.",
                        "recommendations": [
                            "Please try again later or check your API key configuration.",
                        ],
                    }

                st.session_state.readmission_result = readmission_data

                if st.session_state.get("analysis_results"):
                    st.session_state.analysis_results = inject_authoritative_sections(
                        st.session_state.analysis_results,
                        readmission_data,
                        st.session_state.get("clinical_classification_result"),
                    )

                vitals_summary = (
                    f"Blood Pressure: {st.session_state.blood_pressure or 'Not provided'} mmHg\n"
                    f"Heart Rate: {st.session_state.heart_rate} bpm\n"
                    f"Respiratory Rate: {st.session_state.respiratory_rate} breaths/min\n"
                    f"Weight: {st.session_state.patient_weight} kg\n"
                    f"Height: {st.session_state.get('patient_height_feet', 0)} ft "
                    f"{st.session_state.get('patient_height_inches', 0)} in\n"
                    f"Blood Glucose: {st.session_state.blood_glucose} mg/dL\n"
                    f"Temperature: {st.session_state.body_temperature} {st.session_state.temperature_scale}\n"
                    f"Oxygen Saturation: {st.session_state.oxygen_saturation}%\n"
                    f"Waist Circumference: {st.session_state.waist_circumference} cm\n"
                    f"Blood Group: {st.session_state.blood_group}"
                )

                readmission_summary = (
                    f"Prediction: {readmission_data.get('prediction', 'NO')}\n"
                    f"Probability Score: {readmission_data.get('probability', 0)}%\n"
                    f"Risk Level: {readmission_data.get('risk_level', 'Low')}\n"
                    f"Evaluated Factors: Previous Admissions ({prev_adm}), Length of Stay ({los} days), Emergency Visits ({em_visits}), Medications ({meds}), Disposition ({disp})\n"
                    f"Summary: {readmission_data.get('summary', '')}\n"
                    f"Basis: {readmission_data.get('reason', 'Not available')}"
                )

                try:
                    persist_all_widget_state()

                    # --- Retrieval step (MedRAG pattern): retrieve evidence
                    # BEFORE generation, so the LLM can actually ground its
                    # report in it, instead of retrieving it only afterward
                    # for display (which was the previous behavior).
                    combined_clinical_text = build_combined_clinical_text()
                    if rag_service is not None:
                        try:
                            retrieved_evidence = rag_service.retrieve_evidence(
                                combined_clinical_text
                            )
                        except Exception as error:
                            retrieved_evidence = None
                            print(f"[RAG] retrieve_evidence failed: {error}")
                    else:
                        retrieved_evidence = None
                    st.session_state.rag_evidence = retrieved_evidence

                    analysis = generate_diagnostic_report_compat(
                        diagnostic_agent,
                        patient_info={
                            "id": st.session_state.patient_id,
                            "name": st.session_state.patient_name,
                            "age": st.session_state.patient_age,
                            "gender": st.session_state.patient_gender,
                            "last_visit": get_effective_last_visit(),
                            "medical_history": build_effective_medical_history_summary(),
                        },
                        clinical_case=(
                            (
                                f"Chief Complaint: {get_effective_chief_complaint()}\n\n"
                                if get_effective_chief_complaint()
                                else ""
                            )
                            + (
                                f"Symptoms: {build_symptoms_summary()}\n\n"
                                if build_symptoms_summary().strip()
                                else ""
                            )
                            + get_or_generate_clinical_case_and_notes()[0]
                        ),
                        clinical_notes=get_or_generate_clinical_case_and_notes()[1],
                        lab_data=st.session_state.lab_data,
                        image_findings=get_effective_image_findings(),
                        documents=get_effective_document_findings(),
                        vitals=vitals_summary,
                        readmission_summary=readmission_summary,
                        rag_evidence=retrieved_evidence or "",
                    )
                    report_text = extract_report_text(analysis)
                    report_text = ensure_required_diagnostic_headings(report_text)
                    report_text = inject_authoritative_sections(
                        report_text,
                        readmission_data,
                        st.session_state.get("clinical_classification_result"),
                    )
                    report_text = fill_report_patient_details(
                        report_text,
                        patient_name=st.session_state.patient_name,
                        patient_age=st.session_state.patient_age,
                        patient_gender=st.session_state.patient_gender,
                        evaluation_date=datetime.now().strftime("%d-%m-%Y"),
                    )

                    # -----------------------------------------------------
                    # ADDITION ONLY: append Heart Disease Analytics to the
                    # end of the existing Diagnostic Report.
                    # -----------------------------------------------------
                    report_text = (
                        report_text.rstrip()
                        + "\n\n"
                        + build_heart_disease_report_section(
                            st.session_state.get("heart_disease_result")
                        )
                    )

                    if not report_text:
                        raise ValueError(
                            "The DiagnosticAgent returned no usable report text."
                        )

                    st.session_state.analysis_results = report_text
                    st.session_state.show_results = True

                    # combined_clinical_text and rag_evidence were already
                    # computed above, before report generation, so the RAG
                    # step actually grounds the LLM's report (MedRAG pattern:
                    # Retrieval -> Generation) instead of only being
                    # retrieved afterward for display.
                    vitals_for_safety = build_vitals_dict_for_safety_agent()

                    st.session_state.safety_result = safety_agent.screen(
                        combined_clinical_text, vitals=vitals_for_safety
                    )

                    # --------------------------------------------------
                    # Medication Agent — runs after DiagnosticAgent so it
                    # can consume the structured parsed diagnosis sections.
                    # RAG evidence is retrieved here using the confirmed
                    # diagnoses as the query, then passed into recommend().
                    # --------------------------------------------------
                    try:
                        _diag_sections = _extract_diagnostic_sections(report_text)

                        # Build a focused RAG query from the top diagnoses
                        _med_rag_query = " ".join(filter(None, [
                            _diag_sections.get(
                                "Potential Diagnoses Ordered by Likelihood", ""
                            )[:400],
                            get_effective_chief_complaint(),
                            build_symptoms_summary(),
                        ]))

                        _med_rag_evidence = ""
                        if rag_service is not None and _med_rag_query.strip():
                            try:
                                _med_rag_evidence = rag_service.retrieve_evidence(
                                    _med_rag_query
                                )
                            except Exception as _med_rag_err:
                                print(
                                    f"[MedicationAgent RAG] retrieve_evidence "
                                    f"failed: {_med_rag_err}"
                                )

                        st.session_state.medication_result = medication_agent.recommend(
                            structured_diagnosis=_diag_sections,
                            patient_age=st.session_state.patient_age,
                            patient_gender=st.session_state.patient_gender,
                            known_allergies=st.session_state.get(
                                "known_allergies", ""
                            ),
                            current_medications=st.session_state.get(
                                "current_medications", ""
                            ),
                            vitals_summary=vitals_summary,
                            medical_history=build_effective_medical_history_summary(),
                            symptoms=build_symptoms_summary(),
                            chief_complaint=get_effective_chief_complaint(),
                            rag_evidence=_med_rag_evidence,
                            safety_result=st.session_state.get("safety_result"),
                        )
                        print(
                            "[MedicationAgent] Recommendation generated successfully."
                        )
                    except Exception as _med_err:
                        print(f"[MedicationAgent] Error during recommend(): {_med_err}")
                        st.session_state.medication_result = {
                            "status": "unavailable",
                            "otc_medications": "",
                            "medications": [],
                            "self_care_recommendations": "",
                            "critical_safety_finding": "",
                            "immediate_safety_recommendations": "",
                            "why_otc_withheld": "",
                            "dosage_duration": "",
                            "dosage_information": "",
                            "contraindications": "",
                            "drug_interactions": "",
                            "patient_specific_warnings": "",
                            "monitoring_requirements": "",
                            "when_to_escalate": "",
                            "when_to_seek_prescription_care": "",
                            "pharmacist_disclaimer": "",
                            "physician_review_required": True,
                            "emergency_escalation_required": False,
                            "reasoning_summary": "Medication recommendations are currently unavailable.",
                            "rag_evidence_used": "",
                            "reason": "Medication recommendations are currently unavailable.",
                            "error": str(_med_err),
                        }

                    # ADDITION ONLY: append Medication Recommendations to the
                    # end of the report text, following the same pattern used
                    # for Heart Disease Analytics above. This keeps the
                    # MedicationAgent's structured output as the authoritative
                    # source (formatted, not rewritten by any LLM) and makes
                    # it flow into both the Diagnostic Report view and the PDF.
                    st.session_state.analysis_results = (
                        st.session_state.analysis_results.rstrip()
                        + "\n\n"
                        + build_medication_report_section(
                            st.session_state.get("medication_result")
                        )
                    )
                except Exception as error:
                    st.session_state.analysis_results = None
                    st.session_state.show_results = False
                    st.error(f"Unable to generate the diagnostic report: {error}")

        st.markdown("<br>", unsafe_allow_html=True)

        if not st.session_state.get("readmission_result"):
            st.info("Readmission prediction and risk assessment will be generated after clicking 'Analyze Clinical Case / Generate Report'.")
        else:
            rr = st.session_state.readmission_result or {}
            prediction = rr.get("prediction", "NO")
            prob_val = rr.get("probability", 0.03)
            risk_level = rr.get("risk_level", "Low")

            reasoning = rr.get("reason", "AI agent risk assessment based on 6 clinical categories.")

            st.markdown("## 🩺 Readmission Prediction")
            st.markdown("### 30-Day Hospital Readmission Assessment")

            display_score = float(prob_val)
            if display_score <= 1 and display_score > 0:
                display_score *= 100
            display_score = max(0.0, min(display_score, 100.0))

            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Estimated 30-Day Risk", f"{int(round(display_score))}%")
                st.progress(display_score / 100)
            with m_col2:
                st.metric("Risk Level", risk_level)
            with m_col3:
                st.metric("Readmission Flag", prediction)

            risk_badge_map = {
                "High": (st.error, "\U0001F534"),
                "Medium": (st.warning, "\U0001F7E0"),
                "Moderate": (st.warning, "\U0001F7E0"),
                "Low": (st.success, "\U0001F7E2"),
            }
            badge_func, badge_icon = risk_badge_map.get(risk_level, (st.info, "\u26AA"))
            badge_func(f"{badge_icon} {risk_level} Risk Level")

            st.markdown("### 🧠 AI Clinical Reasoning")
            st.info(reasoning)

            recommendations = rr.get("recommendations") or [
                "Continue routine follow-up.",
                "Maintain healthy lifestyle.",
                "Seek medical advice if symptoms worsen.",
            ]

            st.markdown("### ✅ Recommendations")
            rec_html = "".join(f"<li>{item}</li>" for item in recommendations)
            st.markdown(f"<ul>{rec_html}</ul>", unsafe_allow_html=True)

            st.markdown("---")

            summary_text = rr.get(
                "summary",
                f"The patient has an estimated {int(round(display_score))}% risk of readmission within 30 days, indicating a {risk_level} Risk level. The assessment is based on the patient information provided."
            )
            st.markdown(f'<div class="summary-box"><b>Summary</b><br><br>{summary_text}</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

    elif selected_section == "Stroke Risk Assessment":
        st.markdown(
            """
            <div class="section-header">
                <span class="icon-green">🧠</span>Stroke Risk Assessment
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.info(
            "This section adds the supplied StrokeGuard stroke_model.pkl to the "
            "existing Smart Clinic Diagnostic Agent. It does not replace or modify "
            "the existing diagnostic, readmission, medication, safety, or heart-disease agents."
        )

        if stroke_model is None:
            st.error(
                "❌ Stroke model unavailable. Put `stroke_model.pkl` in the same "
                "folder as `app.py` and restart Streamlit."
            )
        else:
            st.markdown("### Patient & Stroke Risk Factors")

            s_col1, s_col2, s_col3 = st.columns(3)

            with s_col1:
                stroke_age = st.number_input(
                    "Age",
                    min_value=1,
                    max_value=120,
                    value=int(st.session_state.get("patient_age", 30) or 30),
                    key="stroke_age_input",
                )
                stroke_gender = st.selectbox(
                    "Gender",
                    ["Male", "Female"],
                    index=(
                        0 if st.session_state.get("patient_gender", "Male") == "Male"
                        else 1
                    ),
                    key="stroke_gender_input",
                )
                stroke_hypertension = st.selectbox(
                    "Hypertension",
                    ["No", "Yes"],
                    index=0 if st.session_state.get("stroke_hypertension", "No") == "No" else 1,
                    key="stroke_hypertension_input",
                )

            with s_col2:
                stroke_heart_disease = st.selectbox(
                    "Heart Disease",
                    ["No", "Yes"],
                    index=0 if st.session_state.get("stroke_heart_disease", "No") == "No" else 1,
                    key="stroke_heart_disease_input",
                )
                stroke_ever_married = st.selectbox(
                    "Ever Married",
                    ["No", "Yes"],
                    index=0 if st.session_state.get("stroke_ever_married", "No") == "No" else 1,
                    key="stroke_ever_married_input",
                )
                stroke_work_type = st.selectbox(
                    "Work Type",
                    [
                        "Private",
                        "Self-employed",
                        "Government Job",
                        "Children",
                        "Never Worked",
                    ],
                    index=[
                        "Private",
                        "Self-employed",
                        "Government Job",
                        "Children",
                        "Never Worked",
                    ].index(
                        st.session_state.get("stroke_work_type", "Private")
                    ),
                    key="stroke_work_type_input",
                )

            with s_col3:
                stroke_residence_type = st.selectbox(
                    "Residence Type",
                    ["Urban", "Rural"],
                    index=0 if st.session_state.get("stroke_residence_type", "Urban") == "Urban" else 1,
                    key="stroke_residence_input",
                )
                stroke_smoking_status = st.selectbox(
                    "Smoking Status",
                    ["never smoked", "formerly smoked", "smokes"],
                    index=[
                        "never smoked",
                        "formerly smoked",
                        "smokes",
                    ].index(
                        st.session_state.get("stroke_smoking_status", "never smoked")
                    ),
                    key="stroke_smoking_input",
                )
                stroke_glucose = st.number_input(
                    "Average Glucose Level",
                    min_value=50.0,
                    max_value=300.0,
                    value=float(st.session_state.get("blood_glucose", 0.0) or 100.0),
                    key="stroke_glucose_input",
                )

            st.markdown("### Body Measurements")

            b_col1, b_col2, b_col3 = st.columns(3)
            with b_col1:
                stroke_weight = st.number_input(
                    "Weight (KG)",
                    min_value=20.0,
                    max_value=250.0,
                    value=float(st.session_state.get("patient_weight", 0.0) or 70.0),
                    key="stroke_weight_input",
                )
            with b_col2:
                stroke_height = st.number_input(
                    "Height (CM)",
                    min_value=50.0,
                    max_value=250.0,
                    value=(
                        float(st.session_state.get("patient_height_feet", 0) or 0) * 30.48
                        + float(st.session_state.get("patient_height_inches", 0) or 0) * 2.54
                    ) if (
                        st.session_state.get("patient_height_feet", 0)
                        or st.session_state.get("patient_height_inches", 0)
                    ) else 170.0,
                    key="stroke_height_input",
                )
            with b_col3:
                st.metric(
                    "Calculated BMI",
                    f"{calculate_stroke_bmi(stroke_weight, stroke_height):.2f}",
                )

            if st.button(
                "🚀 Analyze Stroke Risk",
                type="primary",
                use_container_width=True,
                key="run_stroke_analysis",
            ):
                try:
                    stroke_result = run_stroke_risk_prediction(
                        stroke_model,
                        age=stroke_age,
                        gender=stroke_gender,
                        hypertension=stroke_hypertension,
                        heart_disease=stroke_heart_disease,
                        ever_married=stroke_ever_married,
                        work_type=stroke_work_type,
                        residence_type=stroke_residence_type,
                        avg_glucose=stroke_glucose,
                        weight=stroke_weight,
                        height=stroke_height,
                        smoking_status=stroke_smoking_status,
                    )

                    st.session_state.stroke_result = stroke_result
                    st.session_state.stroke_hypertension = stroke_hypertension
                    st.session_state.stroke_heart_disease = stroke_heart_disease
                    st.session_state.stroke_ever_married = stroke_ever_married
                    st.session_state.stroke_work_type = stroke_work_type
                    st.session_state.stroke_residence_type = stroke_residence_type
                    st.session_state.stroke_smoking_status = stroke_smoking_status
                    st.session_state.stroke_glucose = stroke_glucose
                    st.session_state.stroke_weight = stroke_weight
                    st.session_state.stroke_height = stroke_height

                except Exception as error:
                    st.session_state.stroke_result = None
                    st.error(f"❌ Stroke Risk Analysis Error: {error}")

            stroke_result = st.session_state.get("stroke_result")

            if stroke_result:
                probability = float(stroke_result["probability"])
                risk_level = stroke_result["risk_level"]

                st.divider()

                if stroke_result["prediction"] == 1:
                    st.error("⚠️ Higher Stroke Risk Detected")
                else:
                    st.success("✅ Lower Stroke Risk Detected")

                r_col1, r_col2, r_col3 = st.columns(3)
                r_col1.metric("BMI", f'{stroke_result["bmi"]:.2f}')
                r_col2.metric(
                    "Risk Probability",
                    f"{probability * 100:.2f}%",
                )
                r_col3.metric("Risk Level", risk_level)

                st.subheader("📊 Stroke Risk Meter")
                st.progress(max(0.0, min(probability, 1.0)))
                st.caption(f"Stroke Risk Probability: {probability * 100:.2f}%")

                st.subheader("💡 Health Recommendations")
                for recommendation in stroke_result["recommendations"]:
                    st.write("✔️ " + recommendation)

                with st.expander("🔎 Model Input Values"):
                    st.json(stroke_result["features"])

                st.caption(
                    "Educational/informational use only. This prediction is not a "
                    "medical diagnosis and should be reviewed by a qualified healthcare professional."
                )

            st.markdown("<br>", unsafe_allow_html=True)

    elif selected_section == "Medication Advisor":
        st.markdown(
            """
            <div class="section-header">
                <span class="icon-green">💊</span>Medication Advisor
            </div>
            """,
            unsafe_allow_html=True,
        )

        med = st.session_state.get("medication_result")

        if not st.session_state.get("show_results") or med is None:
            st.info(
                "Run 'Analyze Clinical Case' from the Readmission Analysis tab first. "
                "The Medication Advisor will automatically generate OTC recommendations "
                "once the diagnostic report has been produced."
            )
        elif med.get("error"):
            st.error(
                f"Medication Advisor is currently unavailable: {med['error']}  \n"
                "Please verify your OPENAI_API_KEY or GROQ_API_KEY in the .env file."
            )
        else:
            render_medication_advisor_ui(med)

            # ----------------------------------------------------------------
            # RAG Evidence Used (collapsed expander for transparency)
            # ----------------------------------------------------------------
            rag_used = med.get("rag_evidence_used", "").strip()
            if rag_used:
                with st.expander("📚 Medical Evidence Retrieved (StatPearls)", expanded=False):
                    st.markdown(
                        "The following clinical literature was retrieved from the StatPearls "
                        "knowledge base and used to ground the medication recommendations above."
                    )
                    st.markdown(rag_used)

            st.markdown("<br>", unsafe_allow_html=True)

            # ----------------------------------------------------------------
            # Re-generate button (runs the agent again without a full re-analysis)
            # ----------------------------------------------------------------
            if st.button(
                "🔄 Re-generate Medication Recommendations",
                type="primary",
                key="regen_medication_btn",
                use_container_width=False,
            ):
                with st.spinner("Regenerating medication recommendations..."):
                    try:
                        _current_report = st.session_state.get("analysis_results", "")
                        _regen_sections = _extract_diagnostic_sections(_current_report)

                        _regen_query = " ".join(filter(None, [
                            _regen_sections.get(
                                "Potential Diagnoses Ordered by Likelihood", ""
                            )[:400],
                            get_effective_chief_complaint(),
                        ]))

                        _regen_evidence = ""
                        if rag_service is not None and _regen_query.strip():
                            try:
                                _regen_evidence = rag_service.retrieve_evidence(
                                    _regen_query
                                )
                            except Exception:
                                pass

                        st.session_state.medication_result = medication_agent.recommend(
                            structured_diagnosis=_regen_sections,
                            patient_age=st.session_state.patient_age,
                            patient_gender=st.session_state.patient_gender,
                            known_allergies=st.session_state.get("known_allergies", ""),
                            current_medications=st.session_state.get(
                                "current_medications", ""
                            ),
                            vitals_summary=build_vitals_summary_for_clinical_notes(),
                            medical_history=build_effective_medical_history_summary(),
                            symptoms=build_symptoms_summary(),
                            chief_complaint=get_effective_chief_complaint(),
                            rag_evidence=_regen_evidence,
                            safety_result=st.session_state.get("safety_result"),
                        )

                        # Keep the embedded report/PDF copy of the Medication
                        # Recommendations section in sync with the regenerated
                        # result (same "Medication Recommendations" heading
                        # appended by the main analysis flow above).
                        _new_med_section = build_medication_report_section(
                            st.session_state.get("medication_result")
                        )
                        _current_report_text = st.session_state.get("analysis_results", "") or ""
                        if "### Medication Recommendations" in _current_report_text:
                            st.session_state.analysis_results = re.sub(
                                r"### Medication Recommendations\n.*\Z",
                                _new_med_section,
                                _current_report_text,
                                flags=re.DOTALL,
                            )
                        else:
                            st.session_state.analysis_results = (
                                _current_report_text.rstrip() + "\n\n" + _new_med_section
                            )

                        st.rerun()
                    except Exception as _regen_err:
                        st.error(
                            f"Could not regenerate recommendations: {_regen_err}"
                        )

        st.markdown("<br>", unsafe_allow_html=True)

    elif selected_section == "Diagnostic Report":
        st.markdown(
            """
            <div class="section-header">
                <span class="icon-green">📑</span>Diagnostic Report
            </div>
            """,
            unsafe_allow_html=True
        )

        if not st.session_state.show_results:
            st.info(
                "Generate the diagnostic analysis from the Readmission Analysis section to view the complete report."
            )
        else:
                    report_data = build_report_patient_info()

                    stroke_report = report_data.get("stroke_result")
                    if stroke_report:
                        st.markdown("### 🧠 Stroke Risk Assessment")
                        stroke_probability = float(stroke_report.get("probability", 0.0))
                        stroke_prediction = stroke_report.get("prediction", 0)
                        stroke_level = stroke_report.get("risk_level", "Unavailable")

                        sr_col1, sr_col2, sr_col3 = st.columns(3)
                        sr_col1.metric(
                            "Stroke Risk Probability",
                            f"{stroke_probability * 100:.2f}%",
                        )
                        sr_col2.metric(
                            "Risk Level",
                            stroke_level,
                        )
                        sr_col3.metric(
                            "Prediction",
                            "Higher Risk" if stroke_prediction == 1 else "Lower Risk",
                        )

                        if stroke_prediction == 1:
                            st.warning(
                                "The integrated stroke model flagged a higher predicted risk. "
                                "This is not a diagnosis; clinical review is required."
                            )
                        else:
                            st.success(
                                "The integrated stroke model flagged a lower predicted risk."
                            )

                    if st.session_state.analysis_results:

                        st.markdown("### Patient Information")

                        medical_history_display = (
                            report_data["medical_history"].replace("\n", "<br>")
                            if report_data["medical_history"]
                            else "Not provided"
                        )

                        patient_info_html = f"""
                        <div style="
                            display: grid;
                            grid-template-columns: 1fr 1fr;
                            gap: 10px 30px;
                            padding: 14px 0 18px 0;
                        ">
                            <div><b>Patient ID:</b> {report_data["patient_id"] or "Not provided"}</div>
                            <div><b>Patient Name:</b> {report_data["patient_name"] or "Not provided"}</div>
                            <div><b>Phone Number:</b> {report_data["phone"] or "Not provided"}</div>
                            <div><b>Address:</b> {report_data["address"] or "Not provided"}</div>
                            <div><b>Age:</b> {report_data["patient_age"]}</div>
                            <div><b>Gender:</b> {report_data["patient_gender"]}</div>
                            <div><b>Email:</b> {report_data["email"] or "Not provided"}</div>
                            <div><b>CNIC:</b> {report_data["cnic"] or "Not provided"}</div>
                            <div><b>Last Visit:</b> {report_data["last_visit"] or "Not provided"}</div>
                            <div><b>Member Type:</b> {report_data["member_type"] or "Not provided"}</div>
                            <div><b>Date of Birth:</b> {report_data["date_of_birth"] or "Not provided"}</div>
                            <div><b>Payment Method:</b> {report_data["payment_method"] or "Not provided"}</div>
                            <div style="grid-column: 1 / -1;"><b>Emergency Contact:</b> {report_data["emergency_contact_phone"] or "Not provided"} ({report_data["emergency_contact_relationship"] or "Relationship not provided"})</div>
                        </div>
                        """

                        st.markdown(
                            patient_info_html,
                            unsafe_allow_html=True
                        )

                        st.markdown("### Chief Complaint")
                        st.markdown(
                            report_data["chief_complaint"] or "Not provided"
                        )

                        st.markdown("### Symptoms")
                        st.markdown(
                            report_data["symptoms"] or "Not provided"
                        )

                        st.markdown("### Medical History")
                        st.markdown(
                            medical_history_display,
                            unsafe_allow_html=True,
                        )

                        clinical_intake = report_data["clinical_intake"]
                        st.markdown("### Clinical Case Description")
                        st.markdown(
                            clinical_intake.get("clinical_case_description")
                            or "Not provided"
                        )

                        st.markdown("### Clinical Notes")
                        st.markdown(
                            clinical_intake.get("clinical_notes") or "Not provided"
                        )

                        clinical_exam = report_data["clinical_examination"]
                        if clinical_exam.get("other_symptoms", "").strip():
                            st.markdown("### Other Symptoms / Additional Details")
                            st.markdown(clinical_exam["other_symptoms"])

                        if clinical_exam.get("image_findings", "").strip():
                            st.markdown("### Medical Imaging Findings")
                            st.markdown(clinical_exam["image_findings"])

                        if clinical_exam.get("document_findings", "").strip():
                            st.markdown("### Laboratory & Document Findings")
                            st.markdown(clinical_exam["document_findings"])

                        readmission_result = report_data.get("readmission_result")
                        if readmission_result:
                            st.markdown("### 30-Day Readmission Risk Assessment")

                            basis_text = str(
                                readmission_result.get("reason", "Not available")
                            ).replace(
                                "(trained on 5000 labeled admissions)",
                                ""
                            ).replace(
                                "(trained on 5,000 labeled admissions)",
                                ""
                            ).strip()

                            st.markdown(
                                f"**Prediction:** {readmission_result.get('prediction', 'NO')}  \n"
                                f"**Probability Score:** {readmission_result.get('probability', 0)}/100  \n"
                                f"**Risk Level:** {readmission_result.get('risk_level', 'Low')}  \n"
                                f"**Basis:** {basis_text}"
                            )
                            recommendations = readmission_result.get("recommendations") or []
                            if recommendations:
                                rec_html = "".join(f"<li>{item}</li>" for item in recommendations)
                                st.markdown(f"**Recommendations:**<ul>{rec_html}</ul>", unsafe_allow_html=True)

                        classification_result = report_data.get("clinical_classification_result")
                        if classification_result:
                            predicted_class = classification_result.get("predicted_class", "")
                            confidence = classification_result.get("confidence", 0.0)
                            recommendations = classification_result.get("recommendations") or []
                            st.markdown("### AI-Assisted Clinical Note Classification")
                            st.markdown(
                                f"**Predicted Condition:** {str(predicted_class).title()} "
                                f"(Confidence: {confidence:.2%})"
                            )
                            if recommendations:
                                rec_html = "".join(f"<li>{r}</li>" for r in recommendations)
                                st.markdown(f"**Recommendations:**<ul>{rec_html}</ul>", unsafe_allow_html=True)

                        # ADDITION ONLY: Heart Disease Analytics in Diagnostic Report.
                        heart_result = report_data.get("heart_disease_result")
                        if heart_result:
                            st.markdown("### Heart Disease Analytics")
                            st.markdown(
                                f"**Prediction:** {heart_result.get('prediction', 'NO')}  \\n"
                                f"**Risk Level:** {heart_result.get('risk_level', 'Unavailable')}  \\n"
                                f"**Model Confidence:** "
                                f"{float(heart_result.get('probability', 0.0)):.2%}"
                            )
                            st.markdown(
                                f"**Assessment:** {heart_result.get('description', 'Not available')}"
                            )

                        # ADDITION ONLY: Breast Cancer Ultrasound Classification in Diagnostic Report.
                        breast_cancer_result = report_data.get("breast_cancer_result")
                        if breast_cancer_result:
                            st.markdown("### Breast Cancer Ultrasound Classification")
                            st.markdown(
                                f"**Prediction:** "
                                f"{breast_cancer_result.get('predicted_label', 'Not available')}  \\n"
                                f"**Confidence:** "
                                f"{float(breast_cancer_result.get('confidence', 0.0)):.1%}"
                            )
                            st.markdown(
                                f"**Assessment:** {breast_cancer_result.get('note', 'Not available')}"
                            )

                        safety_result = report_data.get("safety_result")
                        if safety_result:
                            color_map = {
                                "red": ("#dc2626", "#fee2e2", "\U0001F534"),
                                "orange": ("#d97706", "#ffedd5", "\U0001F7E0"),
                                "green": ("#16a34a", "#d1fae5", "\U0001F7E2"),
                            }
                            border_color, bg_color, dot = color_map[safety_result["color"]]

                            flags_html = "".join(
                                f"<li>{f}</li>"
                                for f in (safety_result["critical_flags"] + safety_result["moderate_flags"])
                            )
                            flags_block = f"<ul>{flags_html}</ul>" if flags_html else ""

                            st.markdown(
                                f"""
                                <div style="
                                    background:{bg_color};
                                    border-left:5px solid {border_color};
                                    border-radius:12px;
                                    padding:14px 18px;
                                    margin-bottom:18px;
                                ">
                                    <b>{dot} Safety Check: {safety_result['severity'].upper()}</b><br>
                                    <span style="font-size:0.92rem;">{safety_result['summary']}</span>
                                    {flags_block}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        # ADDITION ONLY: Medication Recommendations in
                        # Diagnostic Report, using the MedicationAgent's own
                        # structured result as the authoritative source.
                        medication_result = report_data.get("medication_result")
                        st.markdown("### 💊 Medication Recommendations")
                        if not medication_result or medication_result.get("error") or medication_result.get("status") == "unavailable":
                            reason = (
                                (medication_result or {}).get("reason")
                                or (medication_result or {}).get("reasoning_summary")
                                or (medication_result or {}).get("error")
                                or "Medication recommendations are currently unavailable."
                            )
                            st.info(reason)
                        else:
                            render_medication_report_ui(medication_result)
                            st.markdown(
                                f"**Physician/Pharmacist Review Required:** "
                                f"{'Yes' if medication_result.get('physician_review_required') else 'No'}  \n"
                                f"**Emergency Escalation Required:** "
                                f"{'Yes' if medication_result.get('emergency_escalation_required') else 'No'}"
                            )

                        st.markdown("### Vitals")

                        vitals = report_data["vitals"]
                        vitals_html = f"""
                        <div style="
                            display: grid;
                            grid-template-columns: repeat(3, 1fr);
                            gap: 12px 24px;
                            padding: 12px 0 20px 0;
                        ">
                            <div><b>Blood Pressure:</b> {vitals.get("blood_pressure") or "Not provided"} mmHg</div>
                            <div><b>Heart Rate:</b> {vitals.get("heart_rate")} bpm</div>
                            <div><b>Respiratory Rate:</b> {vitals.get("respiratory_rate")} breaths/min</div>
                            <div><b>Weight:</b> {vitals.get("weight")} kg</div>
                            <div><b>Height:</b> {vitals.get("height_feet", 0)} ft {vitals.get("height_inches", 0)} in</div>
                            <div><b>Blood Glucose:</b> {vitals.get("blood_glucose")} mg/dL</div>
                            <div><b>Temperature:</b> {vitals.get("temperature")} {vitals.get("temperature_scale")}</div>
                            <div><b>Oxygen Saturation:</b> {vitals.get("oxygen_saturation")}%</div>
                            <div><b>Waist Circumference:</b> {vitals.get("waist_circumference")} cm</div>
                            <div><b>Blood Group:</b> {vitals.get("blood_group")}</div>
                        </div>
                        """

                        st.markdown(
                            vitals_html,
                            unsafe_allow_html=True
                        )

                        formatted_report = standardize_report_headings(
                            st.session_state.analysis_results
                        )
                        st.markdown(
                            formatted_report,
                            unsafe_allow_html=True
                        )

                    st.markdown("---")

                    if PDFReportGenerator is None:
                        st.warning(
                            "PDF export is unavailable because the reportlab package is not installed."
                        )
                    else:
                        pdf_gen = PDFReportGenerator()
                        pdf_bytes = pdf_gen.generate_pdf_report(
                            report_content=fill_report_patient_details(
                                st.session_state.analysis_results or "",
                                patient_name=report_data["patient_name"],
                                patient_age=report_data["patient_age"],
                                patient_gender=report_data["patient_gender"],
                                evaluation_date=datetime.now().strftime("%d-%m-%Y"),
                            ),
                            patient_info=report_data,
                        )
                        if isinstance(pdf_bytes, bytearray):
                            pdf_bytes = bytes(pdf_bytes)

                        st.download_button(
                            label="📥 Download Diagnostic Report as PDF",
                            data=pdf_bytes,
                            file_name=f"Diagnostic_Report_{st.session_state.patient_id or 'Patient'}.pdf",
                            mime="application/pdf",
                            use_container_width=False
                        )

    st.markdown(
        """
        <div class="footer">
            Smart Clinic Diagnostic Agent &copy; 2026 | Powered by AI Diagnostic Intelligence
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()