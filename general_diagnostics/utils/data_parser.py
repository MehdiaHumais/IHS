"""
Medical Data Parser Module
Handles parsing and processing of medical data from various sources
"""
import re
import pandas as pd
from datetime import datetime
import json


class MedicalDataParser:
    def __init__(self):
        self.medical_keywords = [
            'lab', 'test', 'result', 'value', 'panel', 'level', 'count', 'ratio', 'index',
            'range', 'normal', 'abnormal', 'high', 'low', 'flag', 'units', 'ref',
            'glucose', 'cholesterol', 'triglycerides', 'hdl', 'ldl', 'tsh', 't3', 't4',
            'wbc', 'rbc', 'hgb', 'hct', 'platelet', 'creatinine', 'bun', 'bilirubin',
            'alt', 'ast', 'alk', 'phosphatase', 'sodium', 'potassium', 'chloride', 'co2',
            'crp', 'esr', 'hba1c', 'troponin', 'ck', 'ldh', 'psa', 'cea', 'ca125',
            'blood pressure', 'bp', 'hr', 'heart rate', 'temperature', 'temp', 'rr', 'respiratory rate'
        ]
    
    def parse_medical_text(self, text):
        """
        Parse medical text to extract structured data
        """
        result = {
            'patient_info': [],
            'lab_results': [],
            'vital_signs': [],
            'imaging_findings': [],
            'medications': [],
            'diagnoses': [],
            'procedures': [],
            'summary': ''
        }
        
        # Extract lab results
        result['lab_results'] = self._extract_lab_results(text)
        
        # Extract vital signs
        result['vital_signs'] = self._extract_vital_signs(text)
        
        # Extract patient info
        result['patient_info'] = self._extract_patient_info(text)
        
        # Extract imaging findings
        result['imaging_findings'] = self._extract_imaging_findings(text)
        
        # Extract medications
        result['medications'] = self._extract_medications(text)
        
        # Extract diagnoses
        result['diagnoses'] = self._extract_diagnoses(text)
        
        # Extract procedures
        result['procedures'] = self._extract_procedures(text)
        
        # Create summary
        result['summary'] = self._create_summary(text)
        
        return result
    
    def _extract_lab_results(self, text):
        """
        Extract laboratory results from text
        """
        lab_results = []
        
        # Pattern to match lab test names and values
        patterns = [
            r'([A-Za-z\s]+?)\s*[:=]\s*(\d+\.?\d*)\s*([A-Za-z/]+)?',  # Name: Value Unit
            r'([A-Za-z\s]+?)\s+(\d+\.?\d*)\s*([A-Za-z/]+)?',  # Name Value Unit
            r'([\w\s]+?)\s*(\d+\.?\d*)\s*([<>=]\s*[\d\.]+[\s\w/]+)?',  # Name Value Range
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                test_name = match[0].strip()
                value = match[1].strip()
                unit = match[2].strip() if len(match) > 2 and match[2] else ''
                
                # Filter for medical tests only
                if any(keyword.lower() in test_name.lower() for keyword in self.medical_keywords[-15:]):  # Last 15 keywords are lab-related
                    lab_results.append({
                        'test_name': test_name,
                        'value': value,
                        'unit': unit,
                        'reference_range': self._get_reference_range(test_name)
                    })
        
        return lab_results
    
    def _extract_vital_signs(self, text):
        """
        Extract vital signs from text
        """
        vital_signs = []
        
        # Patterns for vital signs
        patterns = {
            'blood_pressure': r'bp[:\s]+(\d+)/(\d+)',
            'heart_rate': r'(hr|heart rate)[:\s]+(\d+)',
            'temperature': r'(temp|temperature)[:\s]+(\d+\.?\d+)\s*([CF]?)',
            'respiratory_rate': r'(rr|respiratory rate)[:\s]+(\d+)',
            'oxygen_saturation': r'(o2 sat|oxygen saturation)[:\s]+(\d+\.?\d+)%'
        }
        
        for vital_type, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if vital_type == 'blood_pressure':
                    vital_signs.append({
                        'type': 'blood_pressure',
                        'systolic': match[0],
                        'diastolic': match[1],
                        'unit': 'mmHg'
                    })
                elif vital_type == 'temperature':
                    vital_signs.append({
                        'type': 'temperature',
                        'value': match[0],
                        'unit': match[1] if len(match) > 1 else 'C'
                    })
                else:
                    vital_signs.append({
                        'type': vital_type,
                        'value': match[-1] if isinstance(match, tuple) else match,
                        'unit': 'bpm' if 'rate' in vital_type else ''
                    })
        
        return vital_signs
    
    def _extract_patient_info(self, text):
        """
        Extract patient information
        """
        patient_info = []
        
        # Patterns for patient info
        patterns = {
            'name': r'(?:patient|name)[:\s]+([A-Z][a-z]+\s[A-Z][a-z]+)',
            'age': r'age[:\s]+(\d+)',
            'dob': r'(?:date of birth|dob)[:\s]+([\d/]+)',
            'mrn': r'(?:mrn|medical record number)[:\s]+([\w\d-]+)',
            'gender': r'(?:gender|sex)[:\s]+(\w+)'
        }
        
        for info_type, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                patient_info.append({
                    'type': info_type,
                    'value': match
                })
        
        return patient_info
    
    def _extract_imaging_findings(self, text):
        """
        Extract imaging findings
        """
        imaging_findings = []
        
        # Look for imaging-related keywords and context
        imaging_sections = re.split(r'(chest x-ray|ct scan|mri|ultrasound|echo|ekg)', text, flags=re.IGNORECASE)
        
        for section in imaging_sections:
            if any(keyword in section.lower() for keyword in ['x-ray', 'ct', 'mri', 'ultrasound', 'echo', 'ekg']):
                findings = re.findall(r'(findings|impression|conclusion)[:\s]*([^.]*\.)', section, re.IGNORECASE)
                for finding in findings:
                    imaging_findings.append({
                        'imaging_type': section.split()[0] if section.split() else 'imaging',
                        'finding': finding[1].strip()
                    })
        
        return imaging_findings
    
    def _extract_medications(self, text):
        """
        Extract medications
        """
        medications = []
        
        # Look for medication sections
        medication_patterns = [
            r'(?:medications|meds)[:\s]+([^.]+?)(?:\n|\.|$)',
            r'(?:current medications)[:\s]*([^.]+?)(?:\n|\.|$)',
            r'([A-Za-z]+\s*\d+mg|\d+\s*mg\s*[A-Za-z]+)',  # Generic medication pattern
        ]
        
        for pattern in medication_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                med_list = [med.strip() for med in match.split(',')]
                for med in med_list:
                    if re.search(r'[A-Za-z]+\s+\d+mg|\d+\s+mg\s+[A-Za-z]+', med, re.IGNORECASE):
                        medications.append({
                            'name': med.strip()
                        })
        
        return medications
    
    def _extract_diagnoses(self, text):
        """
        Extract diagnoses
        """
        diagnoses = []
        
        # Patterns for diagnoses
        patterns = [
            r'(?:diagnosis|dx|primary dx)[:\s]*([^.]+?)(?:\n|\.|$)',
            r'(?:primary diagnosis)[:\s]*([^.]+?)(?:\n|\.|$)',
            r'(?:secondary diagnosis)[:\s]*([^.]+?)(?:\n|\.|$)',
            r'(?:impression)[:\s]*([^.]+?)(?:\n|\.|$)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                diagnosis_list = [diag.strip() for diag in match.split(',')]
                for diag in diagnosis_list:
                    if diag:
                        diagnoses.append({
                            'diagnosis': diag
                        })
        
        return diagnoses
    
    def _extract_procedures(self, text):
        """
        Extract procedures
        """
        procedures = []
        
        # Look for procedure-related keywords
        procedure_patterns = [
            r'(?:procedure|surgery|surgical procedure)[:\s]*([^.]+?)(?:\n|\.|$)',
            r'(?:performed|done)[:\s]*([^.]+?)(?:\n|\.|$)',
        ]
        
        for pattern in procedure_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                procedures.append({
                    'procedure': match.strip()
                })
        
        return procedures
    
    def _get_reference_range(self, test_name):
        """
        Get reference range for a common lab test
        """
        ranges = {
            'glucose': '70-140 mg/dL',
            'wbc': '4.0-11.0 x10^3/μL',
            'rbc': '4.2-5.4 x10^6/μL',
            'hgb': '12.0-16.0 g/dL',
            'hct': '36-46%',
            'platelet': '150-450 x10^3/μL',
            'creatinine': '0.6-1.2 mg/dL',
            'bun': '7-20 mg/dL',
            'bilirubin': '0.1-1.2 mg/dL',
            'alt': '7-56 U/L',
            'ast': '10-40 U/L',
            'sodium': '135-145 mEq/L',
            'potassium': '3.5-5.0 mEq/L',
            'chloride': '98-107 mEq/L',
            'co2': '23-29 mEq/L',
            'tsh': '0.4-4.0 mIU/L',
            't3': '100-200 ng/dL',
            't4': '4.5-12.0 μg/dL'
        }
        
        test_lower = test_name.lower()
        for key, value in ranges.items():
            if key in test_lower:
                return value
        
        return 'Normal range varies'
    
    def _create_summary(self, text):
        """
        Create a brief summary of the medical text
        """
        # Extract key sentences with medical keywords
        sentences = text.split('.')
        medical_sentences = []
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in self.medical_keywords[:20]):  # First 20 keywords
                medical_sentences.append(sentence.strip())
        
        # Return up to 5 key medical sentences
        return '. '.join(medical_sentences[:5]) + '.'

    def parse_lab_data(self, data):
        """
        Parse structured lab data from various formats
        """
        if isinstance(data, pd.DataFrame):
            return self._parse_dataframe_lab_data(data)
        elif isinstance(data, dict):
            return self._parse_dict_lab_data(data)
        elif isinstance(data, str):
            return self._parse_text_lab_data(data)
        else:
            return []
    
    def _parse_dataframe_lab_data(self, df):
        """
        Parse lab data from pandas DataFrame
        """
        lab_results = []
        
        # Check if this looks like lab data
        for col in df.columns:
            if any(keyword in str(col).lower() for keyword in self.medical_keywords[-15:]):
                for _, row in df.iterrows():
                    if pd.notna(row[col]):
                        value = row[col]
                        lab_results.append({
                            'test_name': str(col),
                            'value': str(value),
                            'unit': '',
                            'reference_range': self._get_reference_range(str(col))
                        })
        
        return lab_results
    
    def _parse_dict_lab_data(self, data):
        """
        Parse lab data from dictionary
        """
        lab_results = []
        
        for key, value in data.items():
            if any(keyword in key.lower() for keyword in self.medical_keywords[-15:]):
                lab_results.append({
                    'test_name': key,
                    'value': str(value),
                    'unit': '',
                    'reference_range': self._get_reference_range(key)
                })
        
        return lab_results
    
    def _parse_text_lab_data(self, text):
        """
        Parse lab data from text (already handled by main parser)
        """
        return self._extract_lab_results(text)