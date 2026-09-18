"""
agents/medication_agent.py
--------------------------
LLM-powered OTC medication advisor.

Design principles
─────────────────
• Does NOT re-diagnose.  Accepts the already-parsed structured sections
  produced by DiagnosticAgent (potential diagnoses, critical findings,
  evidence-based recommendations, impression) as ground truth.

• Reuses the existing MedRAGService (StatPearls FAISS index) for evidence
  retrieval.  The caller passes the pre-retrieved `rag_evidence` string so
  the shared singleton is queried once in app.py (not re-instantiated here).

• Contraindication & interaction checks must be grounded strictly in the
  patient's actual known allergies, current medications, medical history,
  available lab/vital information, and retrieved trusted medication
  evidence. Missing patient information is never invented or assumed.

• Accepts the SafetyAgent's result as additional context. If SafetyAgent
  flags a critical / red / emergency / urgent condition, MedicationAgent
  does not produce routine OTC recommendations — it defers to emergency
  care and flags the case for physician review.

• Never fabricates a numeric medication dose. Dosage is only provided
  when supported by retrieved trusted evidence and sufficient patient
  information; otherwise the agent explicitly says dosage requires
  pharmacist/clinician confirmation.

• Returns a structured dict (typed sections, not a text blob) so the UI
  can render each section independently.

• OpenAI → Groq fallback, matching all other agents in the project.

• LLM unavailable → honest error, no fabricated recommendations.
"""

import os
import re
import json
from config import settings as _settings
from typing import Optional, Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ---------------------------------------------------------------------------
# Section headings the LLM must produce — parsed back into a dict afterward.
# ---------------------------------------------------------------------------
MEDICATION_SECTION_HEADINGS = [
    "OTC Medication Options",
    "Self-Care and Supportive Recommendations",
    "Dosage and Duration",
    "Contraindications and Allergy Warnings",
    "Drug Interactions",
    "Patient-Specific Warnings",
    "Monitoring Requirements",
    "When to Seek Prescription or Emergency Care",
    "Pharmacist Disclaimer",
    "Structured Medications JSON",
]

EMERGENCY_SECTION_HEADINGS = [
    "Critical Safety Finding",
    "Immediate Safety Recommendations",
    "Why Routine OTC Medication Is Not Recommended",
    "Patient-Specific Warnings",
    "Monitoring and Warning Signs",
    "When to Seek Emergency Care",
    "Pharmacist Disclaimer",
]

# Keywords used to detect a critical/urgent SafetyAgent result. Matching is
# deliberately generic since the exact SafetyAgent schema may vary.
_EMERGENCY_KEYWORDS = {"critical", "red", "emergency", "urgent"}


class MedicationAgent:
    """
    OTC medication advisor that works downstream of DiagnosticAgent.

    Parameters accepted by recommend()
    ───────────────────────────────────
    structured_diagnosis  dict  — parsed DiagnosticAgent sections:
                                  {
                                    "Potential Diagnoses Ordered by Likelihood": str,
                                    "Critical Findings Requiring Immediate Attention": str,
                                    "Evidence-Based Recommendations for Treatment or Further Testing": str,
                                    "Impression": str,
                                    "Recommendations": str,
                                  }
    patient_age           int
    patient_gender        str
    known_allergies       str  — from Medical History tab
    current_medications   str  — from Medical History tab
    vitals_summary        str  — formatted vitals string
    medical_history       str  — chronic conditions, surgeries, etc.
    symptoms              str  — combined symptom string
    chief_complaint       str
    rag_evidence           str  — pre-retrieved StatPearls text (may be empty)
    safety_result          dict | None — SafetyAgent output. If it indicates a
                                  critical / red / emergency / urgent condition,
                                  MedicationAgent will not produce routine OTC
                                  recommendations and will instead flag the
                                  case for emergency escalation and physician
                                  review.

    Returns
    ───────
    dict with keys:
        status                          str   — "ok" | "emergency" | "unavailable"
        otc_medications                 str   — recommended OTC options with purpose (text)
        medications                     list  — structured per-medication entries, each with
                                                 name, purpose, dosage_information, route,
                                                 frequency, duration, important_precautions
                                                 (populated only where the LLM/evidence support it)
        self_care_recommendations       str   — supportive/self-care guidance (normal/moderate)
        critical_safety_finding         str   — summary of the critical SafetyAgent finding(s)
        immediate_safety_recommendations str  — condition-specific urgent safety actions (critical)
        why_otc_withheld                str   — explanation of why routine OTC is not advised
        dosage_duration                 str   — dosing and duration guidance (text)
        dosage_information              str   — alias of dosage_duration
        contraindications               str   — allergy / condition-specific warnings
        drug_interactions               str   — interactions with current medications
        patient_specific_warnings       str   — warnings specific to this patient's profile
        monitoring_requirements         str   — what to monitor while using OTC treatment
        when_to_escalate                str   — escalation triggers (text)
        when_to_seek_prescription_care  str   — alias of when_to_escalate
        pharmacist_disclaimer           str   — mandatory safety disclaimer
        physician_review_required       bool  — True whenever human clinician review is warranted
        emergency_escalation_required   bool  — True if SafetyAgent flagged a critical/urgent case
        reasoning_summary                str  — brief explanation of how the recommendation
                                                 (or deferral) was reached
        rag_evidence_used               str   — the RAG text fed to the LLM (for transparency)
        error                           str | None
    """

    # ------------------------------------------------------------------
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.llm = None
        self.llm_available = False
        self.model_type = None
        self._init_llm()

    # ------------------------------------------------------------------
    def _init_llm(self):
        """OpenAI first, Groq as fallback — identical pattern to DiagnosticAgent."""
        if self.api_key and self.api_key.startswith("sk-") and len(self.api_key) > 20:
            try:
                self.llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    api_key=self.api_key,
                    temperature=0.1,
                    max_tokens=2500,
                )
                self.llm.invoke("test")
                self.llm_available = True
                self.model_type = "openai"
                return
            except Exception as e:
                print(f"[MedicationAgent] OpenAI init failed: {e}")

        if self.groq_api_key and self.groq_api_key.startswith("gsk-"):
            try:
                self.llm = ChatGroq(
                    temperature=0.1,
                    model_name="llama-3.1-8b-instant",
                    api_key=self.groq_api_key,
                )
                self.llm.invoke("test")
                self.llm_available = True
                self.model_type = "groq"
                return
            except Exception as e:
                print(f"[MedicationAgent] Groq init failed: {e}")

        self.llm_available = False
        print("[MedicationAgent] Warning: LLM unavailable. No API key connected.")

    # ------------------------------------------------------------------
    def recommend(
        self,
        structured_diagnosis: dict,
        patient_age: int = 0,
        patient_gender: str = "",
        known_allergies: str = "",
        current_medications: str = "",
        vitals_summary: str = "",
        medical_history: str = "",
        symptoms: str = "",
        chief_complaint: str = "",
        rag_evidence: str = "",
        safety_result: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """
        Generate structured OTC medication advice from DiagnosticAgent output.

        All heavy lifting (RAG retrieval, LLM call) is done here.
        Returns a typed dict of sections regardless of success/failure.

        If `safety_result` indicates a critical / red / emergency / urgent
        condition, this method short-circuits and returns an emergency
        response instead of routine OTC recommendations.
        """
        emergency_reason = self._check_emergency(safety_result)
        if emergency_reason:
            return self._generate_emergency_recommendations(
                emergency_reason=emergency_reason,
                safety_result=safety_result,
                structured_diagnosis=structured_diagnosis,
                patient_age=patient_age,
                patient_gender=patient_gender,
                known_allergies=known_allergies,
                current_medications=current_medications,
                vitals_summary=vitals_summary,
                medical_history=medical_history,
                symptoms=symptoms,
                chief_complaint=chief_complaint,
                rag_evidence=rag_evidence,
            )

        if not self.llm_available:
            return self._local_response(
                structured_diagnosis,
                safety_result=safety_result,
                rag_evidence=rag_evidence,
            )

        # ------------------------------------------------------------------
        # Pull the most actionable sections from the structured diagnosis.
        # ------------------------------------------------------------------
        diagnoses_text = (
            structured_diagnosis.get("Potential Diagnoses Ordered by Likelihood", "")
            or "Not available"
        ).strip()

        critical_findings = (
            structured_diagnosis.get("Critical Findings Requiring Immediate Attention", "")
            or "None identified"
        ).strip()

        treatment_recs = (
            structured_diagnosis.get(
                "Evidence-Based Recommendations for Treatment or Further Testing", ""
            )
            or "Not available"
        ).strip()

        impression = (
            structured_diagnosis.get("Impression", "")
            or "Not available"
        ).strip()

        safety_summary = self._format_safety_summary(safety_result)
        safety_severity = self._get_safety_severity(safety_result)

        try:
            raw_text = self._call_llm(
                diagnoses_text=diagnoses_text,
                critical_findings=critical_findings,
                treatment_recs=treatment_recs,
                impression=impression,
                patient_age=patient_age,
                patient_gender=patient_gender,
                known_allergies=known_allergies or "None reported",
                current_medications=current_medications or "None reported",
                vitals_summary=vitals_summary or "Not provided",
                medical_history=medical_history or "Not provided",
                symptoms=symptoms or "Not provided",
                chief_complaint=chief_complaint or "Not provided",
                rag_evidence=rag_evidence or "No reference evidence retrieved.",
                safety_summary=safety_summary,
                safety_severity=safety_severity,
            )
        except Exception as e:
            return self._error_response(f"LLM call failed: {e}")

        parsed = self._parse_sections(raw_text)
        parsed = self._apply_result_defaults(parsed)
        parsed["rag_evidence_used"] = rag_evidence or ""
        parsed["error"] = None
        parsed["status"] = "ok"
        parsed["emergency_escalation_required"] = False
        parsed["critical_safety_finding"] = ""
        parsed["immediate_safety_recommendations"] = ""
        parsed["why_otc_withheld"] = ""
        # Physician review is still warranted whenever critical findings were
        # identified upstream, even if they didn't rise to a SafetyAgent
        # emergency flag, when severity is moderate, or when no medications
        # could be confidently derived.
        parsed["physician_review_required"] = bool(
            critical_findings and critical_findings.lower() != "none identified"
        ) or safety_severity == "moderate" or not parsed.get("medications")
        return parsed

    # ------------------------------------------------------------------
    def _call_llm(
        self,
        diagnoses_text,
        critical_findings,
        treatment_recs,
        impression,
        patient_age,
        patient_gender,
        known_allergies,
        current_medications,
        vitals_summary,
        medical_history,
        symptoms,
        chief_complaint,
        rag_evidence,
        safety_summary,
        safety_severity,
    ) -> str:
        severity_guidance = self._severity_guidance_for_prompt(safety_severity)

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a clinical pharmacist assistant working inside a hospital diagnostic system.

A physician's AI diagnostic tool has already assessed the patient and produced structured clinical findings.
A separate SafetyAgent has already screened the case for red flags; its result and severity are provided below.
Your role is to provide useful, clinically appropriate medication and supportive-care guidance based on the
SafetyAgent severity and the confirmed clinical information.

Severity-based rules (follow strictly):
{severity_guidance}

Core rules (all severities):
- Do NOT re-diagnose. Accept the diagnoses from the DiagnosticAgent as given.
- Only recommend OTC medications that are legally available over the counter.
- Never invent a diagnosis, medication, dose, or patient detail not supported by the available information.
- Contraindication and interaction checks must be grounded strictly in the patient's actual known
  allergies, current medications, medical history, available lab/vital information, and the retrieved
  trusted medication evidence below. Never invent or assume missing patient information.
- Flag all potential drug interactions with current medications clearly.
- NEVER guess, calculate, or fabricate a numeric medication dose. Only state a specific dose (amount,
  frequency, duration) when it is directly supported by the Retrieved Medical Evidence AND there is
  sufficient patient information (age, weight-relevant context, organ function, etc.) to apply it safely.
  If either condition is not met, write exactly: "Dosage requires pharmacist/clinician confirmation."
  Do not round, estimate, or infer a dose from partial evidence.
- Write in plain, professional clinical English. No asterisks, no markdown symbols (*_`#).
- Ground your recommendations in the Retrieved Medical Evidence where relevant.
- Always provide useful self-care/supportive recommendations when clinically appropriate for the severity.
- Do NOT recommend routine OTC medication merely to populate a list when it would be unsafe or inappropriate.

Return your response under EXACTLY these headings, each on its own line, in this order:

OTC Medication Options
Self-Care and Supportive Recommendations
Dosage and Duration
Contraindications and Allergy Warnings
Drug Interactions
Patient-Specific Warnings
Monitoring Requirements
When to Seek Prescription or Emergency Care
Pharmacist Disclaimer
Structured Medications JSON

The final section, "Structured Medications JSON", must contain ONLY a JSON array (no prose, no markdown
fences) listing each recommended medication as an object with exactly these fields:
  "name", "purpose", "dosage_information", "route", "frequency", "duration", "important_precautions"
Use "Dosage requires pharmacist/clinician confirmation." for dosage_information whenever a specific
numeric dose is not fully supported by evidence and patient information, per the rule above. Use an
empty string for any field you cannot support. If no medications can be safely recommended, return [].
""",
            ),
            (
                "human",
                """DIAGNOSTIC AGENT OUTPUT (confirmed clinical findings — do not alter)
─────────────────────────────────────────────────────────────────────
Potential Diagnoses:
{diagnoses_text}

Critical Findings:
{critical_findings}

Evidence-Based Treatment Recommendations (from DiagnosticAgent):
{treatment_recs}

Clinical Impression:
{impression}

SAFETYAGENT RESULT (authoritative severity — do not contradict)
─────────────────────────────────────────────────────────────────────
Safety severity: {safety_severity}
{safety_summary}

PATIENT SAFETY PROFILE (use for contraindication and interaction checks; do not assume missing values)
─────────────────────────────────────────────────────────────────────
Age: {patient_age}
Gender: {patient_gender}
Known Allergies: {known_allergies}
Current Medications: {current_medications}
Vital Signs: {vitals_summary}
Medical History: {medical_history}
Chief Complaint: {chief_complaint}
Reported Symptoms: {symptoms}

RETRIEVED MEDICAL EVIDENCE (StatPearls — use to ground recommendations and any dosage statements)
─────────────────────────────────────────────────────────────────────
{rag_evidence}

Now produce the ten-section medication advisory following the heading structure above.
""",
            ),
        ])

        chain = prompt | self.llm | StrOutputParser()

        return chain.invoke(
            {
                "diagnoses_text": diagnoses_text,
                "critical_findings": critical_findings,
                "treatment_recs": treatment_recs,
                "impression": impression,
                "patient_age": patient_age if patient_age else "Not provided",
                "patient_gender": patient_gender or "Not provided",
                "known_allergies": known_allergies,
                "current_medications": current_medications,
                "vitals_summary": vitals_summary,
                "medical_history": medical_history,
                "chief_complaint": chief_complaint,
                "symptoms": symptoms,
                "rag_evidence": rag_evidence,
                "safety_summary": safety_summary,
                "safety_severity": safety_severity or "unknown",
                "severity_guidance": severity_guidance,
            }
        )

    # ------------------------------------------------------------------
    def _parse_sections(self, raw_text: str) -> dict:
        """
        Split the LLM output on the known section headings.
        Returns a dict keyed by canonical section names.
        Falls back gracefully if a heading is missing.
        """
        # Strip any accidental markdown symbols the LLM may have inserted
        text = re.sub(r"\*+", "", raw_text)
        text = re.sub(r"__+", "", text)
        text = text.replace("`", "")

        result = {
            "otc_medications": "",
            "self_care_recommendations": "",
            "dosage_duration": "",
            "contraindications": "",
            "drug_interactions": "",
            "patient_specific_warnings": "",
            "monitoring_requirements": "",
            "when_to_escalate": "",
            "pharmacist_disclaimer": "",
            "structured_medications_raw": "",
        }

        # Map heading text → result key
        heading_map = {
            "OTC Medication Options": "otc_medications",
            "Self-Care and Supportive Recommendations": "self_care_recommendations",
            "Dosage and Duration": "dosage_duration",
            "Contraindications and Allergy Warnings": "contraindications",
            "Drug Interactions": "drug_interactions",
            "Patient-Specific Warnings": "patient_specific_warnings",
            "Monitoring Requirements": "monitoring_requirements",
            "When to Seek Prescription or Emergency Care": "when_to_escalate",
            "Pharmacist Disclaimer": "pharmacist_disclaimer",
            "Structured Medications JSON": "structured_medications_raw",
        }

        current_key = None
        buffer = []

        for line in text.splitlines():
            stripped = line.strip()
            if stripped in heading_map:
                if current_key and buffer:
                    result[current_key] = "\n".join(buffer).strip()
                current_key = heading_map[stripped]
                buffer = []
            elif current_key is not None:
                buffer.append(line)

        # Flush the last section
        if current_key and buffer:
            result[current_key] = "\n".join(buffer).strip()

        # If parsing failed entirely (LLM ignored headings), put everything in otc_medications
        if not any(v for k, v in result.items() if k != "structured_medications_raw"):
            result["otc_medications"] = text.strip()

        # Parse the structured per-medication JSON, if present. Never fabricate
        # this list — an empty/unparseable block simply yields no structured
        # medications; the free-text sections above remain the source of truth.
        medications: List[Dict[str, Any]] = []
        raw_json = result.pop("structured_medications_raw", "")
        if raw_json:
            cleaned = raw_json.strip()
            cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()
            try:
                parsed_json = json.loads(cleaned)
                if isinstance(parsed_json, list):
                    for entry in parsed_json:
                        if not isinstance(entry, dict):
                            continue
                        medications.append({
                            "name": entry.get("name", "") or "",
                            "purpose": entry.get("purpose", "") or "",
                            "dosage_information": entry.get("dosage_information", "")
                                or "Dosage requires pharmacist/clinician confirmation.",
                            "route": entry.get("route", "") or "",
                            "frequency": entry.get("frequency", "") or "",
                            "duration": entry.get("duration", "") or "",
                            "important_precautions": entry.get("important_precautions", "") or "",
                        })
            except (json.JSONDecodeError, TypeError):
                # Malformed JSON from the LLM — fall back to no structured list
                # rather than guessing at medication details.
                medications = []

        result["medications"] = medications
        result = self._apply_result_defaults(result)
        result["reasoning_summary"] = self._build_reasoning_summary(result)

        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _apply_result_defaults(result: dict) -> dict:
        """Ensure backward-compatible keys are always present."""
        result.setdefault("self_care_recommendations", "")
        result.setdefault("critical_safety_finding", "")
        result.setdefault("immediate_safety_recommendations", "")
        result.setdefault("why_otc_withheld", "")
        result["dosage_information"] = result.get("dosage_duration", "")
        result["when_to_seek_prescription_care"] = result.get("when_to_escalate", "")
        result.setdefault("medications", [])
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _build_reasoning_summary(result: dict) -> str:
        """Short, non-fabricated summary of how the recommendation was derived."""
        if result.get("status") == "emergency" or result.get("emergency_escalation_required"):
            finding = (result.get("critical_safety_finding") or "").strip()
            if finding:
                return (
                    "Routine OTC medication withheld due to a critical SafetyAgent finding. "
                    f"Immediate safety guidance provided for: {finding[:200]}."
                )
            return (
                "Routine OTC medication withheld due to a critical SafetyAgent finding. "
                "Condition-specific immediate safety guidance provided."
            )
        if result.get("medications"):
            names = ", ".join(m["name"] for m in result["medications"] if m.get("name"))
            if names:
                return (
                    f"Recommendation based on the confirmed diagnostic findings, patient "
                    f"allergy/medication profile, SafetyAgent severity, and retrieved evidence; "
                    f"candidate OTC options: {names}."
                )
        if result.get("self_care_recommendations", "").strip():
            return (
                "Recommendation based on the confirmed diagnostic findings, patient "
                "allergy/medication profile, SafetyAgent severity, and retrieved evidence. "
                "Supportive/self-care guidance provided; no specific OTC medication could "
                "be confidently identified from the available information."
            )
        return (
            "Recommendation based on the confirmed diagnostic findings, patient "
            "allergy/medication profile, SafetyAgent severity, and retrieved evidence. "
            "No specific OTC medication could be confidently identified from the "
            "available information."
        )

    # ------------------------------------------------------------------
    def _check_emergency(self, safety_result: Optional[Dict[str, Any]]) -> Optional[str]:
        """
        Check the actual SafetyAgent severity.

        Only critical/emergency/red/urgent severity should stop
        the MedicationAgent and trigger emergency escalation.
        """

        if not safety_result or not isinstance(safety_result, dict):
            return None

        # The actual severity is the authoritative value.
        severity = str(
            safety_result.get("severity", "")
        ).strip().lower()

        if severity in {"critical", "emergency", "red", "urgent"}:
            return f"SafetyAgent severity is '{severity}'."

        # Only treat critical flags as critical if the list actually
        # contains something.
        critical_flags = safety_result.get("critical_flags") or []

        if isinstance(critical_flags, list) and len(critical_flags) > 0:
            return "SafetyAgent reported one or more critical findings."

        return None

    # ------------------------------------------------------------------
    @staticmethod
    def _format_safety_summary(safety_result: Optional[Dict[str, Any]]) -> str:
        """Human-readable summary of the SafetyAgent result for prompt context."""
        if not safety_result:
            return "No SafetyAgent result provided."
        if isinstance(safety_result, dict):
            try:
                return json.dumps(safety_result, indent=2)
            except (TypeError, ValueError):
                return str(safety_result)
        return str(safety_result)

    # ------------------------------------------------------------------
    @staticmethod
    def _get_safety_severity(safety_result: Optional[Dict[str, Any]]) -> str:
        if not safety_result or not isinstance(safety_result, dict):
            return "unknown"
        return str(safety_result.get("severity", "")).strip().lower() or "unknown"

    # ------------------------------------------------------------------
    @staticmethod
    def _format_critical_findings_text(
        safety_result: Optional[Dict[str, Any]],
        emergency_reason: str = "",
    ) -> str:
        flags: List[str] = []
        if safety_result and isinstance(safety_result, dict):
            raw_flags = safety_result.get("critical_flags") or []
            if isinstance(raw_flags, list):
                flags = [str(f).strip() for f in raw_flags if str(f).strip()]
            summary = str(safety_result.get("summary", "")).strip()
            if summary and not flags:
                flags.append(summary)
        if flags:
            return "\n".join(f"- {flag}" for flag in flags)
        return emergency_reason or "Critical safety finding identified by SafetyAgent."

    # ------------------------------------------------------------------
    @staticmethod
    def _severity_guidance_for_prompt(safety_severity: str) -> str:
        if safety_severity == "moderate":
            return (
                "MODERATE severity:\n"
                "- OTC medication may be recommended only when appropriate and safe for the confirmed findings.\n"
                "- Always provide supportive/self-care recommendations.\n"
                "- Address the moderate SafetyAgent finding(s) with extra precautions.\n"
                "- Explain what to monitor and when professional evaluation or prescription care is needed.\n"
                "- Set physician_review_required when closer clinician review is warranted."
            )
        if safety_severity in {"normal", "green"}:
            return (
                "NORMAL severity:\n"
                "- Recommend appropriate OTC options when clinically supported by the available information.\n"
                "- Provide safe self-care/supportive measures.\n"
                "- Provide dosage/duration only when supported by evidence and patient information.\n"
                "- Include contraindications, allergy warnings, drug interactions, monitoring, and escalation guidance."
            )
        return (
            "Severity not clearly normal or moderate:\n"
            "- Be conservative. Recommend OTC medication only when clearly safe and supported.\n"
            "- Always provide useful self-care/supportive recommendations when clinically appropriate.\n"
            "- Include monitoring and escalation guidance."
        )

    # ------------------------------------------------------------------
    def _generate_emergency_recommendations(
        self,
        emergency_reason: str,
        safety_result: Optional[Dict[str, Any]],
        structured_diagnosis: dict,
        patient_age: int = 0,
        patient_gender: str = "",
        known_allergies: str = "",
        current_medications: str = "",
        vitals_summary: str = "",
        medical_history: str = "",
        symptoms: str = "",
        chief_complaint: str = "",
        rag_evidence: str = "",
    ) -> dict:
        """
        Produce condition-specific safety guidance for critical/emergency cases.
        Routine OTC recommendations are withheld; useful actionable guidance is still provided.
        """
        critical_finding_text = self._format_critical_findings_text(
            safety_result, emergency_reason
        )

        diagnoses_text = (
            structured_diagnosis.get("Potential Diagnoses Ordered by Likelihood", "")
            or "Not available"
        ).strip()
        critical_findings = (
            structured_diagnosis.get("Critical Findings Requiring Immediate Attention", "")
            or "None identified"
        ).strip()
        impression = (
            structured_diagnosis.get("Impression", "")
            or "Not available"
        ).strip()
        safety_summary = self._format_safety_summary(safety_result)

        parsed: Dict[str, Any] = {}
        if self.llm_available:
            try:
                raw_text = self._call_llm_emergency(
                    emergency_reason=emergency_reason,
                    critical_finding_text=critical_finding_text,
                    diagnoses_text=diagnoses_text,
                    critical_findings=critical_findings,
                    impression=impression,
                    patient_age=patient_age,
                    patient_gender=patient_gender,
                    known_allergies=known_allergies or "None reported",
                    current_medications=current_medications or "None reported",
                    vitals_summary=vitals_summary or "Not provided",
                    medical_history=medical_history or "Not provided",
                    symptoms=symptoms or "Not provided",
                    chief_complaint=chief_complaint or "Not provided",
                    rag_evidence=rag_evidence or "No reference evidence retrieved.",
                    safety_summary=safety_summary,
                )
                parsed = self._parse_emergency_sections(raw_text)
            except Exception as e:
                print(f"[MedicationAgent] Emergency LLM call failed: {e}")

        if not parsed or not any(
            parsed.get(k, "").strip()
            for k in (
                "immediate_safety_recommendations",
                "why_otc_withheld",
                "when_to_escalate",
            )
        ):
            parsed = self._emergency_fallback_response(
                emergency_reason=emergency_reason,
                critical_finding_text=critical_finding_text,
            )

        result = self._apply_result_defaults({
            "status": "emergency",
            "otc_medications": "",
            "medications": [],
            "self_care_recommendations": "",
            "critical_safety_finding": parsed.get("critical_safety_finding")
                or critical_finding_text,
            "immediate_safety_recommendations": parsed.get(
                "immediate_safety_recommendations", ""
            ),
            "why_otc_withheld": parsed.get("why_otc_withheld", ""),
            "dosage_duration": "",
            "contraindications": "",
            "drug_interactions": "",
            "patient_specific_warnings": parsed.get("patient_specific_warnings", ""),
            "monitoring_requirements": parsed.get("monitoring_requirements", ""),
            "when_to_escalate": parsed.get("when_to_escalate", ""),
            "pharmacist_disclaimer": parsed.get("pharmacist_disclaimer", "")
                or (
                    "This is not a substitute for emergency medical care. Seek immediate "
                    "professional evaluation."
                ),
            "physician_review_required": True,
            "emergency_escalation_required": True,
            "rag_evidence_used": rag_evidence or "",
            "error": None,
        })
        result["dosage_information"] = "Dosage requires pharmacist/clinician confirmation."
        result["when_to_seek_prescription_care"] = result.get("when_to_escalate", "")
        result["reasoning_summary"] = self._build_reasoning_summary(result)
        return result

    # ------------------------------------------------------------------
    def _call_llm_emergency(
        self,
        emergency_reason,
        critical_finding_text,
        diagnoses_text,
        critical_findings,
        impression,
        patient_age,
        patient_gender,
        known_allergies,
        current_medications,
        vitals_summary,
        medical_history,
        symptoms,
        chief_complaint,
        rag_evidence,
        safety_summary,
    ) -> str:
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a clinical pharmacist assistant working inside a hospital diagnostic system.

SafetyAgent has flagged this case as CRITICAL / EMERGENCY. Routine over-the-counter (OTC)
medication must NOT be recommended as treatment for the critical finding.

Your task is to provide useful, condition-specific IMMEDIATE SAFETY guidance based on the
actual SafetyAgent critical finding(s) and the confirmed clinical information available.
Do NOT invent an OTC medication merely because no medication list is expected.
Do NOT hard-code condition-specific treatment protocols unrelated to the actual critical finding.
Do NOT re-diagnose.

Rules:
- Clearly identify the critical finding from the SafetyAgent result.
- Explain that routine OTC medication is withheld because it should not delay urgent professional evaluation.
- Provide actionable immediate safety steps supported by the available information.
- Include patient-specific warnings based on allergies, medications, history, and vitals when relevant.
- Include monitoring/warning signs and when immediate/emergency professional care is required.
- Write in plain, professional clinical English. No asterisks, no markdown symbols (*_`#).
- Ground general guidance in Retrieved Medical Evidence only where relevant; do not fabricate doses.

Return your response under EXACTLY these headings, each on its own line, in this order:

Critical Safety Finding
Immediate Safety Recommendations
Why Routine OTC Medication Is Not Recommended
Patient-Specific Warnings
Monitoring and Warning Signs
When to Seek Emergency Care
Pharmacist Disclaimer
""",
            ),
            (
                "human",
                """EMERGENCY CONTEXT
─────────────────────────────────────────────────────────────────────
Emergency trigger: {emergency_reason}

SafetyAgent critical finding(s):
{critical_finding_text}

Full SafetyAgent result:
{safety_summary}

DIAGNOSTIC AGENT OUTPUT (confirmed clinical findings — do not alter)
─────────────────────────────────────────────────────────────────────
Potential Diagnoses:
{diagnoses_text}

Critical Findings:
{critical_findings}

Clinical Impression:
{impression}

PATIENT SAFETY PROFILE
─────────────────────────────────────────────────────────────────────
Age: {patient_age}
Gender: {patient_gender}
Known Allergies: {known_allergies}
Current Medications: {current_medications}
Vital Signs: {vitals_summary}
Medical History: {medical_history}
Chief Complaint: {chief_complaint}
Reported Symptoms: {symptoms}

RETRIEVED MEDICAL EVIDENCE (StatPearls — optional context only)
─────────────────────────────────────────────────────────────────────
{rag_evidence}

Now produce the seven-section critical-case safety advisory following the heading structure above.
""",
            ),
        ])

        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "emergency_reason": emergency_reason,
            "critical_finding_text": critical_finding_text,
            "safety_summary": safety_summary,
            "diagnoses_text": diagnoses_text,
            "critical_findings": critical_findings,
            "impression": impression,
            "patient_age": patient_age if patient_age else "Not provided",
            "patient_gender": patient_gender or "Not provided",
            "known_allergies": known_allergies,
            "current_medications": current_medications,
            "vitals_summary": vitals_summary,
            "medical_history": medical_history,
            "chief_complaint": chief_complaint,
            "symptoms": symptoms,
            "rag_evidence": rag_evidence,
        })

    # ------------------------------------------------------------------
    def _parse_emergency_sections(self, raw_text: str) -> dict:
        text = re.sub(r"\*+", "", raw_text)
        text = re.sub(r"__+", "", text)
        text = text.replace("`", "")

        result = {
            "critical_safety_finding": "",
            "immediate_safety_recommendations": "",
            "why_otc_withheld": "",
            "patient_specific_warnings": "",
            "monitoring_requirements": "",
            "when_to_escalate": "",
            "pharmacist_disclaimer": "",
        }

        heading_map = {
            "Critical Safety Finding": "critical_safety_finding",
            "Immediate Safety Recommendations": "immediate_safety_recommendations",
            "Why Routine OTC Medication Is Not Recommended": "why_otc_withheld",
            "Patient-Specific Warnings": "patient_specific_warnings",
            "Monitoring and Warning Signs": "monitoring_requirements",
            "When to Seek Emergency Care": "when_to_escalate",
            "Pharmacist Disclaimer": "pharmacist_disclaimer",
        }

        current_key = None
        buffer: List[str] = []

        for line in text.splitlines():
            stripped = line.strip()
            if stripped in heading_map:
                if current_key and buffer:
                    result[current_key] = "\n".join(buffer).strip()
                current_key = heading_map[stripped]
                buffer = []
            elif current_key is not None:
                buffer.append(line)

        if current_key and buffer:
            result[current_key] = "\n".join(buffer).strip()

        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _emergency_fallback_response(
        emergency_reason: str,
        critical_finding_text: str,
    ) -> dict:
        """Deterministic fallback when the emergency LLM path is unavailable."""
        immediate = (
            "Seek immediate professional medical evaluation without delay. "
            "Do not attempt to manage this critical finding with routine OTC medication. "
            "If the patient is unstable, confused, unresponsive, or deteriorating, activate "
            "emergency services now."
        )
        why_withheld = (
            "Routine OTC medication recommendations are withheld because SafetyAgent identified "
            "a critical finding that requires urgent professional evaluation. OTC treatment could "
            "delay appropriate emergency care and is not an appropriate primary response for this "
            f"critical finding. Reason: {emergency_reason}"
        )
        return {
            "critical_safety_finding": critical_finding_text,
            "immediate_safety_recommendations": immediate,
            "why_otc_withheld": why_withheld,
            "patient_specific_warnings": (
                "Follow allergy and current-medication precautions already documented in the "
                "patient profile until emergency clinicians take over care."
            ),
            "monitoring_requirements": (
                "Monitor level of consciousness, breathing, circulation, and any worsening of "
                "the critical finding while awaiting emergency evaluation."
            ),
            "when_to_escalate": (
                "Call emergency services or go to the nearest emergency department immediately. "
                "Do not wait for symptoms to improve."
            ),
            "pharmacist_disclaimer": (
                "This is not a substitute for emergency medical care. Seek immediate "
                "professional evaluation."
            ),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _emergency_response(reason: str) -> dict:
        """
        Legacy static emergency response — kept for compatibility.
        Prefer _generate_emergency_recommendations(), which uses SafetyAgent findings.
        """
        fallback = MedicationAgent._emergency_fallback_response(
            emergency_reason=reason,
            critical_finding_text=reason,
        )
        result = MedicationAgent._apply_result_defaults({
            "status": "emergency",
            "otc_medications": "",
            "medications": [],
            "self_care_recommendations": "",
            "critical_safety_finding": fallback["critical_safety_finding"],
            "immediate_safety_recommendations": fallback["immediate_safety_recommendations"],
            "why_otc_withheld": fallback["why_otc_withheld"],
            "dosage_duration": "",
            "dosage_information": "Dosage requires pharmacist/clinician confirmation.",
            "contraindications": "",
            "drug_interactions": "",
            "patient_specific_warnings": fallback["patient_specific_warnings"],
            "monitoring_requirements": fallback["monitoring_requirements"],
            "when_to_escalate": fallback["when_to_escalate"],
            "when_to_seek_prescription_care": fallback["when_to_escalate"],
            "pharmacist_disclaimer": fallback["pharmacist_disclaimer"],
            "physician_review_required": True,
            "emergency_escalation_required": True,
            "reasoning_summary": fallback["why_otc_withheld"],
            "rag_evidence_used": "",
            "error": None,
        })
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _local_response(structured_diagnosis, safety_result=None, rag_evidence=""):
        """Return conservative guidance when no remote LLM is available."""
        critical = str(
            structured_diagnosis.get(
                "Critical Findings Requiring Immediate Attention", ""
            )
            or ""
        ).strip()
        diagnosis = str(
            structured_diagnosis.get("Impression", "")
            or structured_diagnosis.get(
                "Potential Diagnoses Ordered by Likelihood", ""
            )
            or "the reported symptoms"
        ).strip()
        return MedicationAgent._apply_result_defaults({
            "status": "local",
            "otc_medications": "No medication was selected without a pharmacist review.",
            "medications": [],
            "self_care_recommendations": (
                f"Supportive care may be considered for {diagnosis}. "
                "Follow the treating clinician's plan and maintain hydration as appropriate."
            ),
            "critical_safety_finding": critical,
            "immediate_safety_recommendations": "Seek urgent medical care for severe or rapidly worsening symptoms." if critical else "",
            "why_otc_withheld": "Remote medication reasoning is unavailable, so no drug or dose is recommended automatically.",
            "dosage_duration": "Dosage requires pharmacist/clinician confirmation.",
            "dosage_information": "Dosage requires pharmacist/clinician confirmation.",
            "contraindications": "Review allergies, pregnancy status, chronic conditions, and current medicines with a pharmacist.",
            "drug_interactions": "A pharmacist should check interactions before any new medicine is taken.",
            "patient_specific_warnings": "Do not start, stop, or combine medicines without professional advice.",
            "monitoring_requirements": "Monitor symptoms and seek care if they worsen or do not improve.",
            "when_to_escalate": "Seek emergency care for breathing difficulty, chest pain, confusion, fainting, or severe pain.",
            "pharmacist_disclaimer": "This conservative local guidance is not a diagnosis or a substitute for a pharmacist or clinician.",
            "physician_review_required": True,
            "emergency_escalation_required": False,
            "reasoning_summary": "Local conservative guidance used because no remote medication model was reachable.",
            "rag_evidence_used": rag_evidence or "",
            "error": None,
        })

    @staticmethod
    def _error_response(message: str) -> dict:
        return MedicationAgent._apply_result_defaults({
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
            "reasoning_summary": message,
            "rag_evidence_used": "",
            "error": message,
        })