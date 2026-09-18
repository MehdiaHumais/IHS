import os
import re
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class DiagnosticAgent:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.llm = None
        self.llm_available = False
        self.llm_error_message = ""
        self.model_type = None  # Track which model type is being used

        # Try OpenAI first
        if self.api_key and self.api_key.startswith("sk-") and len(self.api_key) > 20:
            try:
                # Add max_tokens to prevent truncated responses from the LLM
                self.llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=self.api_key, temperature=0.1, max_tokens=3000)
                # A simple test to see if the API key is valid
                self.llm.invoke("test")
                self.llm_available = True
                self.model_type = "openai"
            except Exception as e:
                print(f"OpenAI initialization failed: {e}")
                # Continue to try other options

        # If OpenAI failed or wasn't available, try Groq
        if not self.llm_available and self.groq_api_key and self.groq_api_key.startswith("gsk-"):
            try:
                self.llm = ChatGroq(temperature=0.1, model_name="llama-3.1-8b-instant", api_key=self.groq_api_key)
                # Simple test
                self.llm.invoke("test")
                self.llm_available = True
                self.model_type = "groq"
            except Exception as e:
                print(f"Groq initialization failed: {e}")
                self.llm_error_message = f"LLM service not available. Could not connect to OpenAI or Groq. Please check your API keys and network connection. Error: {e}"

        # If neither worked, provide a detailed error
        if not self.llm_available:
            if not self.api_key or not self.api_key.startswith("sk-"):
                self.llm_error_message = "Invalid or missing OpenAI API key. Please set a valid OPENAI_API_KEY in your .env file."
            elif self.api_key.startswith("sk-proj-") and "incorrect" in self.llm_error_message.lower():
                self.llm_error_message = "Invalid OpenAI API key format. The key you provided appears to be incorrect. Please check your OpenAI API key at https://platform.openai.com/account/api-keys and update your .env file."
            else:
                self.llm_error_message = "Could not connect to the AI model. Both OpenAI and Groq services are unavailable. Please check your API keys and network connection."

            print(f"Warning: {self.llm_error_message}")

    def analyze_case(self, clinical_case, clinical_notes, lab_data, image_findings, documents, document_types=""):
        """
        Analyze clinical case with comprehensive, pre-processed data from all sources.
        The agent now trusts that the input data has been properly parsed upstream.
        """
        if not self.llm_available:
            local_result = self._analyze_locally(clinical_case, clinical_notes, lab_data, image_findings, documents, document_types)
            note = f"LLM unavailable: {self.llm_error_message}"
            local_result["diagnostic_analysis"] = f"{note}\n\n{local_result['diagnostic_analysis']}"
            local_result["formatted_report"] = f"{local_result['formatted_report']}\n\nNOTE: {note}"
            local_result["summary"] = f"{local_result['summary']} Generated locally because no AI model was available."
            return local_result

        try:
            return self._analyze_with_llm(clinical_case, clinical_notes, lab_data, image_findings, documents, document_types)
        except Exception as e:
            return {
                "error": str(e),
                "diagnostic_analysis": "An unexpected error occurred during analysis.",
                "formatted_report": "An unexpected error occurred during analysis.",
                "summary": "Analysis failed due to an unexpected error."
            }



    def _analyze_with_llm(self, clinical_case, clinical_notes, lab_data, image_findings, documents, document_types):
        """Use LLM for detailed clinical analysis"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are an advanced medical diagnostic assistant. Analyze the provided clinical data and generate a comprehensive diagnostic report.
Focus on extracting meaningful insights from clinical notes, lab results, imaging findings, and other documents.
Identify patterns, abnormal values, and potential diagnoses based on the evidence provided.
Pay special attention to numerical values in laboratory results and flag any that are outside normal ranges.
Consider document types to better interpret the context of findings.
"""),
            ("human", """
# Clinical Diagnostic Analysis Request

## Patient Clinical Case
{clinical_case}

## Clinical Notes
{clinical_notes}

## Laboratory Results and Data
{lab_data}

## Medical Imaging Findings
{image_findings}

## Supporting Documents Content
{documents}

## Document Types Processed
{document_types}

## Analysis Instructions
Please analyze all provided information and provide:
1. A comprehensive diagnostic analysis with potential diagnoses ordered by likelihood
2. Critical findings that require immediate attention (flag any abnormal lab values or concerning imaging findings)
3. Evidence-based recommendations for treatment or further testing
4. Risk assessment and urgency level determination
5. A well-formatted medical report suitable for healthcare professionals

Be sure to integrate information from all document types and identify relationships between findings.
""")        ])

        chain = prompt | self.llm | StrOutputParser()

        try:
            analysis_result = chain.invoke({
                "clinical_case": clinical_case,
                "clinical_notes": clinical_notes,
                "lab_data": lab_data,
                "image_findings": image_findings,
                "documents": documents,
                "document_types": document_types
            })

            # Generate summary from the analysis
            summary_prompt = ChatPromptTemplate.from_messages([
                ("system", "As a medical professional, summarize the key diagnostic findings in 2-3 sentences focusing on the most critical and actionable points."),
                ("human", analysis_result)
            ])

            summary_chain = summary_prompt | self.llm | StrOutputParser()
            summary = summary_chain.invoke({})

            return {
                "diagnostic_analysis": analysis_result,
                "formatted_report": analysis_result,
                "summary": summary
            }
        except Exception as e:
            # Fallback to local analysis if LLM fails
            return self._analyze_locally(clinical_case, clinical_notes, lab_data, image_findings, documents, document_types)

    def _analyze_locally(self, clinical_case, clinical_notes, lab_data, image_findings, documents, document_types):
        """Local analysis when LLM is not available"""
        # Perform keyword analysis to identify important medical terms
        critical_keywords = [
            "abnormal", "elevated", "decreased", "positive", "negative",
            "critical", "urgent", "emergency", "danger", "risk", "high", "low",
            "diabetes", "hypertension", "cardiac", "respiratory", "infection",
            "tumor", "cancer", "stroke", "heart attack", "bleeding", "mass",
            "lesion", "fracture", "aneurysm", "embolism", "infarct", "edema"
        ]

        # Find critical terms in all inputs
        all_text = clinical_case + " " + clinical_notes + " " + lab_data + " " + image_findings + " " + documents

        critical_finds = []
        for keyword in critical_keywords:
            if keyword.lower() in all_text.lower():
                # Avoid duplicate entries
                kw_cap = keyword.capitalize()
                if kw_cap not in critical_finds:
                    critical_finds.append(kw_cap)

        # Extract numerical lab values if present
        import re
        lab_numbers = re.findall(r'\d+\.\d+', lab_data)

        # Determine urgency based on keywords
        high_risk_keywords = ["emergency", "critical", "urgent", "danger", "immediate"]
        urgency_level = "Medium"
        for keyword in high_risk_keywords:
            if keyword.lower() in all_text.lower():
                urgency_level = "High"
                break

        # Count processed document types
        doc_count = len([dt for dt in document_types.split("\n") if dt.strip()]) if document_types.strip() else 0

        # Generate analysis
        analysis = f"""
COMPREHENSIVE DIAGNOSTIC ANALYSIS

CLINICAL CASE OVERVIEW: {clinical_case[:200] if clinical_case else 'Not specified'}{'...' if len(clinical_case) > 200 else ''}

CRITICAL FINDINGS IDENTIFIED:
{chr(10).join(['• ' + find for find in critical_finds[:15]]) if critical_finds else '• No critical keywords detected in available data'}

LABORATORY DATA ANALYSIS:
{lab_data if lab_data.startswith("LABORATORY DATA:") or lab_data.startswith("STRUCTURED") else f'- Relevant values identified: {len(lab_numbers)} numerical values detected'}
- Abnormal value flags: {self._identify_abnormal_values(lab_data)}

IMAGING FINDINGS SYNTHESIS:
{image_findings.split("IMAGE FINDINGS:")[1] if "IMAGE FINDINGS:" in image_findings else image_findings[:500] + '...' if len(image_findings) > 500 else image_findings or 'No imaging data'}

SUPPORTING DOCUMENTS REVIEW:
- Total document types processed: {doc_count}
- Document content integrated: {len(documents) > 100 and 'Yes' or 'Limited due to content availability'}

CLINICAL NOTES SYNTHESIS:
{clinical_notes[:300] + '...' if len(clinical_notes) > 300 else clinical_notes or 'No additional clinical notes'}

RECOMMENDATIONS:
• Prioritize review of flagged critical findings
• Correlate imaging findings with laboratory results for comprehensive assessment
• Consider specialist consultation based on identified abnormalities
• Schedule follow-up tests as indicated by abnormal values
• Monitor high-risk indicators closely

URGENCY ASSESSMENT: {urgency_level}
"""

        # Create a formatted report
        report = f"""MEDICAL DIAGNOSTIC REPORT
Generated: {self._get_current_datetime()}

==============================
PATIENT CLINICAL PRESENTATION
==============================
{clinical_case or 'No clinical case description provided'}

=======================
CLINICAL NOTES SUMMARY
=======================
{clinical_notes or 'No clinical notes provided'}

========================
LABORATORY FINDINGS
========================
{lab_data}

========================
IMAGING RESULTS
========================
{image_findings}

========================
SUPPORTING DOCUMENTS
========================
Document Types Processed: {doc_count}
{documents or 'No supporting documents provided'}

========================
CLINICAL ASSESSMENT
========================
{analysis}

========================
RECOMMENDATIONS
========================
• Review all identified critical findings with appropriate specialists
• Address any abnormal laboratory values per standard protocols
• Consider additional diagnostic testing based on clinical presentation
• Schedule follow-up appointments as clinically indicated
• Monitor patient for any changes in condition
"""

        summary = f"Comprehensive analysis completed. {len(critical_finds)} critical findings identified across documents. Urgency: {urgency_level}. {doc_count} document types processed."

        return {
            "diagnostic_analysis": analysis,
            "formatted_report": report,
            "summary": summary
        }

    def _identify_abnormal_values(self, lab_data):
        """Identify potentially abnormal lab values"""
        # Basic pattern matching for lab values
        import re

        # Look for common lab tests and their potential abnormal values
        abnormal_flags = []

        # Glucose - normal is typically 70-140 mg/dL
        glucose_matches = re.findall(r'(glucose|blood sugar|random glucose|fasting glucose)[\s:=]*([\d.]+)', lab_data, re.IGNORECASE)
        for match in glucose_matches:
            try:
                val = float(match[1])
                if val > 200 or val < 50:
                    abnormal_flags.append(f"HIGH GLUCOSE: {val} mg/dL")
            except:
                pass

        # Hemoglobin - normal ranges vary but roughly 12-16 g/dL for women, 13.5-17.5 for men
        hgb_matches = re.findall(r'(hemoglobin|hgb)[\s:=]*([\d.]+)', lab_data, re.IGNORECASE)
        for match in hgb_matches:
            try:
                val = float(match[1])
                if val < 11 or val > 18:
                    abnormal_flags.append(f"ABNORMAL HEMOGLOBIN: {val} g/dL")
            except:
                pass

        # Creatinine - normal is usually 0.6-1.2 mg/dL
        cre_matches = re.findall(r'(creatinine|cr)[\s:=]*([\d.]+)', lab_data, re.IGNORECASE)
        for match in cre_matches:
            try:
                val = float(match[1])
                if val > 2.0:
                    abnormal_flags.append(f"ELEVATED CREATININE: {val} mg/dL (kidney function concern)")
            except:
                pass

        # Bilirubin - normal is usually <1.2 mg/dL
        bilirubin_matches = re.findall(r'(bilirubin)[\s:=]*([\d.]+)', lab_data, re.IGNORECASE)
        for match in bilirubin_matches:
            try:
                val = float(match[1])
                if val > 2.0:
                    abnormal_flags.append(f"ELEVATED BILIRUBIN: {val} mg/dL (liver function concern)")
            except:
                pass

        if abnormal_flags:
            return "; ".join(abnormal_flags)
        else:
            return "No abnormal values detected in basic screening"

    def _get_current_datetime(self):
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")