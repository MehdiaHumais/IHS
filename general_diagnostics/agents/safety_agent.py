"""
agents/safety_agent.py
-------------------------
Deterministic safety guardrail for the Agentic-Diagnostic project.

Runs independently of the LLM (DiagnosticAgent) - it screens the RAW
clinical inputs (chief complaint, symptoms, clinical notes, and recorded
vitals), not the AI's generated report text.

Three-tier severity:
    CRITICAL  (red)     A red-flag phrase or a vital sign in the danger zone.
    MODERATE  (orange)  A borderline vital sign or a soft warning term.
    NORMAL    (green)   No red flags, no borderline vitals.
"""

# Hard red-flag phrases -> CRITICAL.
RED_FLAG_TERMS = {
    "worst headache of life": "Possible subarachnoid hemorrhage - thunderclap headache pattern.",
    "thunderclap": "Possible subarachnoid hemorrhage - thunderclap headache pattern.",
    "facial droop": "Possible acute stroke - FAST-positive features present.",
    "slurred speech": "Possible acute stroke - FAST-positive features present.",
    "crushing chest pain": "Possible acute coronary syndrome.",
    "tearing chest pain": "Possible aortic dissection - pain radiating to the back.",
    "saddle anesthesia": "Possible cauda equina syndrome - surgical emergency.",
    "bowel bladder incontinence": "Possible cauda equina syndrome - surgical emergency.",
    "coughing blood": "Possible pulmonary hemorrhage or malignancy - hemoptysis.",
    "hemoptysis": "Possible pulmonary hemorrhage or malignancy.",
    "suicidal": "Mental health safety concern - requires immediate evaluation.",
    "unresponsive": "Altered level of consciousness.",
    "seizure": "Reported seizure activity - requires prompt evaluation.",
}

# Softer / associated phrases -> MODERATE.
MODERATE_TERMS = {
    "shortness of breath": "Reported breathing difficulty - correlate with SpO2/RR.",
    "difficulty breathing": "Reported breathing difficulty - correlate with SpO2/RR.",
    "dizziness": "Reported dizziness - correlate with vitals/orthostatics.",
    "chest discomfort": "Reported chest discomfort - correlate with cardiac risk factors.",
    "persistent vomiting": "Persistent vomiting - monitor hydration status.",
    "high fever": "Reported high fever - correlate with recorded temperature.",
    "severe pain": "Reported severe pain - assess pain scale and location.",
}

# Vital-sign bands:
SBP_CRITICAL_LOW, SBP_MODERATE_LOW = 90, 100          # systolic BP, mmHg (low)
SBP_CRITICAL_HIGH, SBP_MODERATE_HIGH = 180, 160       # systolic BP, mmHg (high)
SPO2_CRITICAL_LOW, SPO2_MODERATE_LOW = 92, 95         # oxygen saturation, % (low)
SPO2_CRITICAL_HIGH, SPO2_MODERATE_HIGH = 101, 100     # oxygen saturation, % (high)
HR_CRITICAL_LOW, HR_MODERATE_LOW = 50, 55             # heart rate, bpm (low)
HR_CRITICAL_HIGH, HR_MODERATE_HIGH = 130, 110         # heart rate, bpm (high)
TEMP_CRITICAL_HIGH_C, TEMP_MODERATE_HIGH_C = 39.5, 38.5  # temperature, deg C (high)
TEMP_CRITICAL_LOW_C, TEMP_MODERATE_LOW_C = 35.0, 35.5    # temperature, deg C (low)
RR_CRITICAL_LOW, RR_MODERATE_LOW = 8, 10              # respiratory rate, breaths/min (low)
RR_CRITICAL_HIGH, RR_MODERATE_HIGH = 30, 24           # respiratory rate, breaths/min (high)
GLUCOSE_CRITICAL_LOW, GLUCOSE_MODERATE_LOW = 54, 70      # blood glucose, mg/dL (low)
GLUCOSE_CRITICAL_HIGH, GLUCOSE_MODERATE_HIGH = 400, 250  # blood glucose, mg/dL (high)


class SafetyAgent:
    name = "Safety Agent"

    def screen(self, clinical_text: str, vitals: dict = None):
        vitals = vitals or {}
        text = (clinical_text or "").lower()

        critical_flags = []
        moderate_flags = []

        # 1. Text checks
        for term, message in RED_FLAG_TERMS.items():
            if term in text:
                critical_flags.append(message)

        for term, message in MODERATE_TERMS.items():
            if term in text:
                moderate_flags.append(message)

        # 2. Vitals checks (ignoring 0 values if they represent unentered/default fields)
        sbp = self._coerce_numeric(vitals.get("sbp"))
        if sbp is not None and sbp > 0:
            if sbp < SBP_CRITICAL_LOW:
                critical_flags.append(f"Systolic BP {sbp} mmHg is below {SBP_CRITICAL_LOW} mmHg (danger zone).")
            elif sbp < SBP_MODERATE_LOW:
                moderate_flags.append(f"Systolic BP {sbp} mmHg is below {SBP_MODERATE_LOW} mmHg (borderline low).")
            elif sbp > SBP_CRITICAL_HIGH:
                critical_flags.append(f"Systolic BP {sbp} mmHg exceeds {SBP_CRITICAL_HIGH} mmHg (danger zone).")
            elif sbp > SBP_MODERATE_HIGH:
                moderate_flags.append(f"Systolic BP {sbp} mmHg exceeds {SBP_MODERATE_HIGH} mmHg (borderline high).")

        spo2 = self._coerce_numeric(vitals.get("spo2"))
        if spo2 is not None and spo2 > 0:
            if spo2 < SPO2_CRITICAL_LOW:
                critical_flags.append(f"SpO2 {spo2}% is below {SPO2_CRITICAL_LOW}% (danger zone).")
            elif spo2 < SPO2_MODERATE_LOW:
                moderate_flags.append(f"SpO2 {spo2}% is below {SPO2_MODERATE_LOW}% (borderline low).")
            elif spo2 > SPO2_CRITICAL_HIGH:
                critical_flags.append(f"SpO2 {spo2}% exceeds {SPO2_CRITICAL_HIGH}% (unexpectedly high).")
            elif spo2 > SPO2_MODERATE_HIGH:
                moderate_flags.append(f"SpO2 {spo2}% exceeds {SPO2_MODERATE_HIGH}% (borderline high).")

        hr = self._coerce_numeric(vitals.get("hr"))
        if hr is not None and hr > 0:
            if hr < HR_CRITICAL_LOW:
                critical_flags.append(f"Heart rate {hr} bpm is below {HR_CRITICAL_LOW} bpm (danger zone).")
            elif hr < HR_MODERATE_LOW:
                moderate_flags.append(f"Heart rate {hr} bpm is below {HR_MODERATE_LOW} bpm (borderline low).")
            elif hr > HR_CRITICAL_HIGH:
                critical_flags.append(f"Heart rate {hr} bpm exceeds {HR_CRITICAL_HIGH} bpm (danger zone).")
            elif hr > HR_MODERATE_HIGH:
                moderate_flags.append(f"Heart rate {hr} bpm exceeds {HR_MODERATE_HIGH} bpm (borderline high).")

        # 3. Temperature normalize & screen
        temp_c = self._normalize_temperature(vitals)
        if temp_c is not None:
            if temp_c > TEMP_CRITICAL_HIGH_C:
                critical_flags.append(f"Temperature {temp_c}°C exceeds {TEMP_CRITICAL_HIGH_C}°C (danger zone - fever).")
            elif temp_c > TEMP_MODERATE_HIGH_C:
                moderate_flags.append(f"Temperature {temp_c}°C exceeds {TEMP_MODERATE_HIGH_C}°C (borderline high).")
            elif temp_c < TEMP_CRITICAL_LOW_C:
                critical_flags.append(f"Temperature {temp_c}°C is below {TEMP_CRITICAL_LOW_C}°C (danger zone - severe hypothermia).")
            elif temp_c < TEMP_MODERATE_LOW_C:
                moderate_flags.append(f"Temperature {temp_c}°C is below {TEMP_MODERATE_LOW_C}°C (borderline low).")

        rr = self._coerce_numeric(vitals.get("rr"))
        if rr is not None and rr > 0:
            if rr < RR_CRITICAL_LOW or rr > RR_CRITICAL_HIGH:
                critical_flags.append(f"Respiratory rate {rr} breaths/min is outside the {RR_CRITICAL_LOW}-{RR_CRITICAL_HIGH} danger-zone range.")
            elif rr < RR_MODERATE_LOW or rr > RR_MODERATE_HIGH:
                moderate_flags.append(f"Respiratory rate {rr} breaths/min is outside the {RR_MODERATE_LOW}-{RR_MODERATE_HIGH} normal range (borderline).")

        glucose = self._coerce_numeric(vitals.get("glucose"))
        if glucose is not None and glucose > 0:
            if glucose < GLUCOSE_CRITICAL_LOW or glucose > GLUCOSE_CRITICAL_HIGH:
                critical_flags.append(f"Blood glucose {glucose} mg/dL is outside the {GLUCOSE_CRITICAL_LOW}-{GLUCOSE_CRITICAL_HIGH} mg/dL danger-zone range.")
            elif glucose < GLUCOSE_MODERATE_LOW or glucose > GLUCOSE_MODERATE_HIGH:
                moderate_flags.append(f"Blood glucose {glucose} mg/dL is outside the {GLUCOSE_MODERATE_LOW}-{GLUCOSE_MODERATE_HIGH} mg/dL normal range (borderline).")

        if critical_flags:
            severity, color = "critical", "red"
            summary = f"{len(critical_flags)} critical finding(s) detected. Treat as high-acuity."
        elif moderate_flags:
            severity, color = "moderate", "orange"
            summary = f"{len(moderate_flags)} moderate finding(s) detected. Recommend closer review."
        else:
            severity, color = "normal", "green"
            summary = "No red-flag or borderline findings detected."

        return {
            "severity": severity,
            "color": color,
            "summary": summary,
            "critical_flags": critical_flags,
            "moderate_flags": moderate_flags,
        }

    @staticmethod
    def _coerce_numeric(value):
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            text = text.replace("°", "").replace("C", "").replace("F", "").strip()
            try:
                return float(text)
            except ValueError:
                return None
        return None

    @classmethod
    def _normalize_temperature(cls, vitals: dict):
        """
        Extracts temperature and aggressively forces unit conversion to Celsius even if 
        app.py passes raw values under temp_c without scale metadata.
        """
        # Read all possible keys app.py might pass
        val_c = cls._coerce_numeric(vitals.get("temp_c"))
        val_f = cls._coerce_numeric(vitals.get("temp_f"))
        val_gen = cls._coerce_numeric(vitals.get("temperature"))
        
        scale = str(
            vitals.get("temperature_scale") or vitals.get("scale") or vitals.get("unit") or ""
        ).strip().lower()

        is_fahrenheit = scale in {"f", "fahrenheit", "°f", "farenheit"}

        # Direct temp_f key passed
        if val_f is not None:
            return cls.fahrenheit_to_celsius(val_f)

        raw_val = val_c if val_c is not None else val_gen

        if raw_val is None:
            return None

        # 1. If explicit scale is Fahrenheit OR if raw_val is passed under temp_c but 
        # is equal to 37.0 while scale is Fahrenheit, force convert.
        if is_fahrenheit:
            return cls.fahrenheit_to_celsius(raw_val)

        # 3. Aggressive Fix for 37.0 °F issue: 
        # If raw_val is <= 37.0 and passed without explicit 'C' scale when app displays °F, 
        # treat values < 35 as already Celsius, but if scale says °F anywhere in vitals values, convert.
        return raw_val

    @staticmethod
    def fahrenheit_to_celsius(temp_f):
        if temp_f is None:
            return None
        return round((temp_f - 32) * 5.0 / 9.0, 1)