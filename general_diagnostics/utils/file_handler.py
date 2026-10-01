# utils/file_handler.py
import os
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from PIL import Image
import base64
import io
import tempfile
import sqlite3
from datetime import datetime
import hashlib
import json
import csv
import xlrd  # For older Excel files
import shutil

# --- OCR and PDF-to-Image Conversion ---
# It's recommended to have these libraries installed for full functionality
try:
    import pytesseract
    from pdf2image import convert_from_path
except ImportError:
    pytesseract = None
    convert_from_path = None

def is_tesseract_available():
    """Check if Tesseract OCR is installed and available in the system's PATH"""
    if pytesseract is None:
        return False
    return shutil.which("tesseract") is not None

class FileHandler:
    def __init__(self, upload_dir="uploads", db_path="medical_data.db"):
        self.upload_dir = upload_dir
        self.db_path = db_path
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(os.path.join(upload_dir, "documents"), exist_ok=True)
        os.makedirs(os.path.join(upload_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(upload_dir, "excel"), exist_ok=True)
        os.makedirs(os.path.join(upload_dir, "processed"), exist_ok=True)
        self.init_database()
        self.tesseract_available = is_tesseract_available()
        if not self.tesseract_available:
            print("WARNING: Tesseract OCR is not installed or not in your PATH. Image and scanned PDF text extraction will be disabled.")

    def init_database(self):
        """Initialize SQLite database for file metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS uploaded_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size INTEGER,
                file_type TEXT,
                document_type TEXT DEFAULT 'Unknown',
                patient_id TEXT DEFAULT 'unknown',
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed BOOLEAN DEFAULT FALSE,
                extracted_text TEXT,
                metadata TEXT,
                checksum TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patient_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT UNIQUE,
                name TEXT,
                age INTEGER,
                gender TEXT,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                clinical_case TEXT,
                clinical_notes TEXT,
                lab_data TEXT,
                image_findings TEXT,
                documents TEXT,
                document_types TEXT,
                analysis_result TEXT,
                report TEXT,
                summary TEXT
            )
        ''')

        conn.commit()
        conn.close()
    
    def calculate_checksum(self, file_content):
        """Calculate MD5 checksum for file integrity"""
        return hashlib.md5(file_content).hexdigest()
    
    def save_uploaded_file(self, uploaded_file, folder="documents", patient_id="unknown"):
        """Save uploaded file and return the path with metadata"""
        if uploaded_file is not None:
            # Calculate checksum
            file_content = uploaded_file.getvalue()
            checksum = self.calculate_checksum(file_content)
            
            # Check if file already exists
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM uploaded_files WHERE checksum = ?", (checksum,))
            existing_file = cursor.fetchone()
            
            if existing_file:
                conn.close()
                return None, "File already exists"
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_extension = uploaded_file.name.split('.')[-1].lower()
            unique_filename = f"{timestamp}_{checksum[:8]}.{file_extension}"
            
            file_path = os.path.join(self.upload_dir, folder, unique_filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, "wb") as f:
                f.write(file_content)
            
            # Extract text content
            extracted_text = self.extract_text_from_file(file_path)
            
            # Store metadata
            metadata = {
                "original_filename": uploaded_file.name,
                "content_type": uploaded_file.type,
                "size_mb": round(uploaded_file.size / (1024 * 1024), 2)
            }
            
            # Insert into database
            cursor.execute('''
                INSERT INTO uploaded_files 
                (filename, original_filename, file_path, file_size, file_type, 
                 patient_id, extracted_text, metadata, checksum)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                unique_filename,
                uploaded_file.name,
                file_path,
                uploaded_file.size,
                file_extension,
                patient_id,
                extracted_text,
                json.dumps(metadata),
                checksum
            ))
            
            conn.commit()
            conn.close()
            
            return file_path, "File saved successfully"
        return None, "No file provided"
    
    def read_pdf(self, file_path):
        """Read text from PDF file, with OCR fallback for scanned images."""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            # If text is minimal, it might be a scanned PDF. Try OCR.
            if len(text.strip()) < 100:
                text += self._ocr_pdf(file_path)
        except Exception as e:
            print(f"Error reading PDF with standard method: {str(e)}. Attempting OCR.")
            text = self._ocr_pdf(file_path)
        return text

    def _ocr_pdf(self, file_path):
        """Helper function to perform OCR on a PDF file."""
        if not self.tesseract_available or convert_from_path is None:
            return "\n[OCR UNAVAILABLE: Cannot process scanned PDF. Please install Tesseract and pdf2image.]"

        text = ""
        try:
            images = convert_from_path(file_path)
            for i, image in enumerate(images):
                text += f"\n--- OCR of Page {i+1} ---\n"
                text += pytesseract.image_to_string(image)
        except Exception as e:
            return f"\n[OCR FAILED on PDF: {str(e)}]"
        return text

    def read_docx(self, file_path):
        """Read text from DOCX file"""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            # Also extract tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join([cell.text for cell in row.cells])
                    text += row_text + "\n"
            # Extract from textboxes and headers/footers if available
            for section in doc.sections:
                header = section.header
                for paragraph in header.paragraphs:
                    text += f"[HEADER] {paragraph.text}\n"
                footer = section.footer
                for paragraph in footer.paragraphs:
                    text += f"[FOOTER] {paragraph.text}\n"
            return text
        except Exception as e:
            print(f"Error reading DOCX: {str(e)}")
            return "Could not read DOCX file"
    
    def read_excel(self, file_path):
        """Read data from Excel file with proper formatting"""
        try:
            # Try pandas first (for .xlsx files)
            df = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')  # Read all sheets
            text = ""
            for sheet_name, sheet_df in df.items():
                text += f"\n--- Sheet: {sheet_name} ---\n"
                # Convert to string with better formatting
                text += sheet_df.to_string(index=False) + "\n"

                # Try to identify medical lab data with more comprehensive checks
                columns = [str(col).lower() for col in sheet_df.columns if pd.notna(col)]
                medical_keywords = ['test', 'result', 'value', 'lab', 'panel', 'level', 'count', 'ratio', 'index',
                                   'range', 'normal', 'abnormal', 'high', 'low', 'flag', 'units', 'ref',
                                   'glucose', 'cholesterol', 'triglycerides', 'hdl', 'ldl', 'tsh', 't3', 't4',
                                   'wbc', 'rbc', 'hgb', 'hct', 'platelet', 'creatinine', 'bun', 'bilirubin',
                                   'alt', 'ast', 'alk', 'phosphatase', 'sodium', 'potassium', 'chloride', 'co2',
                                   'crp', 'esr', 'hba1c', 'troponin', 'ck', 'ldh', 'psa', 'cea', 'ca125']

                medical_cols = [col for col in columns if any(keyword in col for keyword in medical_keywords)]
                if medical_cols or any('lab' in col or 'test' in col or 'result' in col for col in columns):
                    text += "\n[LABORATORY DATA IDENTIFIED]\n"
                    for idx, row in sheet_df.iterrows():
                        for col, val in row.items():
                            if pd.notna(val):
                                str_val = str(val)
                                col_lower = str(col).lower()
                                # Check if this is a medical lab value
                                if any(keyword in col_lower for keyword in medical_keywords) or self._is_numeric_value(str_val):
                                    normal_range = ""
                                    # Try to find normal values in comments or related cells
                                    text += f"[Lab test: {str(col)} = {str_val}{normal_range}]\n"
            return text
        except Exception:
            # Fallback for older Excel formats using xlrd
            try:
                workbook = xlrd.open_workbook(file_path)
                text = ""
                for sheet_idx, sheet in enumerate(workbook.sheets()):
                    sheet_name = sheet.name or f"Sheet{sheet_idx+1}"
                    text += f"\n--- Sheet: {sheet_name} ---\n"
                    for row in range(min(sheet.nrows, 100)):  # Limit to first 100 rows
                        row_values = [str(sheet.cell_value(row, col)) for col in range(sheet.ncols)]
                        text += " | ".join(row_values) + "\n"
                return text
            except Exception as e:
                return f"Could not read Excel file: {str(e)}"
    
    def _is_numeric_value(self, value):
        """Check if a string value is numeric"""
        try:
            float(value.replace(',', '').replace('<', '').replace('>', '').replace('=', ''))
            return True
        except ValueError:
            return False

    def read_csv(self, file_path):
        """Read data from CSV file"""
        try:
            df = pd.read_csv(file_path)
            text = df.to_string(index=False)

            # Try to identify medical lab data with more comprehensive checks
            columns = [str(col).lower() for col in df.columns if pd.notna(col)]
            medical_keywords = ['test', 'result', 'value', 'lab', 'panel', 'level', 'count', 'ratio', 'index',
                               'range', 'normal', 'abnormal', 'high', 'low', 'flag', 'units', 'ref',
                               'glucose', 'cholesterol', 'triglycerides', 'hdl', 'ldl', 'tsh', 't3', 't4',
                               'wbc', 'rbc', 'hgb', 'hct', 'platelet', 'creatinine', 'bun', 'bilirubin',
                               'alt', 'ast', 'alk', 'phosphatase', 'sodium', 'potassium', 'chloride', 'co2',
                               'crp', 'esr', 'hba1c', 'troponin', 'ck', 'ldh', 'psa', 'cea', 'ca125']

            medical_cols = [col for col in columns if any(keyword in col for keyword in medical_keywords)]
            if medical_cols or any('lab' in col or 'test' in col or 'result' in col for col in columns):
                text += "\n[LABORATORY DATA IDENTIFIED]\n"
                for idx, row in df.iterrows():
                    for col, val in row.items():
                        if pd.notna(val):
                            str_val = str(val)
                            col_lower = str(col).lower()
                            # Check if this is a medical lab value
                            if any(keyword in col_lower for keyword in medical_keywords) or self._is_numeric_value(str_val):
                                normal_range = ""
                                text += f"[Lab test: {str(col)} = {str_val}{normal_range}]\n"
            return text
        except Exception as e:
            # Fallback to manual CSV reading
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = ""
                    reader = csv.reader(f)
                    for row in reader:
                        text += " | ".join(row) + "\n"
                return text
            except Exception as e2:
                return f"Could not read CSV file: {str(e2)}"
    
    def read_image(self, file_path):
        """Extract text from image using OCR and medical image analysis (if available)"""
        result = f"Image file: {os.path.basename(file_path)}\n"
        try:
            # Handle DICOM files separately with detailed analysis
            if file_path.lower().endswith(('.dcm', '.dicom')):
                try:
                    from .image_processor import MedicalImageProcessor
                    processor = MedicalImageProcessor()
                    return processor.analyze_image_for_report(file_path)
                except Exception as e:
                    return f"Failed to process DICOM file: {str(e)}"

            # For other images, perform medical image analysis
            try:
                from .image_processor import MedicalImageProcessor
                processor = MedicalImageProcessor()
                medical_analysis = processor.analyze_image_for_report(file_path)
                result += f"\n{medical_analysis}\n"
            except Exception as e:
                print(f"Medical image processing error: {str(e)}")

            # Then, try OCR for text extraction with medical context
            if not self.tesseract_available:
                result += "\n[OCR DISABLED: Tesseract OCR is not installed or not in your PATH.]\n"
                # Still provide basic image properties
                img = Image.open(file_path)
                result += f"Image Properties:\n"
                result += f"- Dimensions: {img.size[0]} x {img.size[1]} pixels\n"
                result += f"- Color Mode: {img.mode}\n"
                result += f"- Format: {img.format}\n"
                return result

            try:
                img = Image.open(file_path)
                ocr_text = pytesseract.image_to_string(img)

                # Enhance image for better OCR if possible
                if not ocr_text.strip() or len(ocr_text.strip()) < 20:
                    # Try to enhance the image for better OCR
                    try:
                        import cv2
                        import numpy as np
                        # Convert PIL to OpenCV format
                        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                        # Enhance image
                        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                        # Apply adaptive thresholding
                        enhanced = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
                        # Convert back to PIL for pytesseract
                        enhanced_pil = Image.fromarray(enhanced)
                        ocr_text = pytesseract.image_to_string(enhanced_pil)
                    except:
                        pass  # If enhancement fails, continue with original OCR

                if ocr_text.strip():
                    result += f"\nOCR TEXT FOUND IN IMAGE:\n{ocr_text}\n"
                    # Check for medical content in OCR text
                    medical_keywords = [
                        'lab', 'test', 'result', 'patient', 'date', 'doctor', 'radiology', 'imaging',
                        'x-ray', 'ct', 'mri', 'ultrasound', 'radiologist', 'findings', 'impression',
                        'normal', 'abnormal', 'positive', 'negative', 'diagnosis', 'recommendation',
                        'glucose', 'cholesterol', 'creatinine', 'bilirubin', 'alt', 'ast', 'wbc', 'rbc',
                        'hgb', 'hct', 'platelet', 'tsh', 't3', 't4', 'sodium', 'potassium', 'chloride'
                    ]
                    found_keywords = [kw for kw in medical_keywords if kw.lower() in ocr_text.lower()]
                    if found_keywords:
                        result += f"[MEDICAL KEYWORDS IDENTIFIED: {', '.join(set(found_keywords))}]\n"

                    # Extract potential medical values
                    import re
                    medical_values = re.findall(r'([a-zA-Z\s]+)\s*[:=]\s*([\d.]+[^\s]*)', ocr_text, re.IGNORECASE)
                    if medical_values:
                        result += "\n[POSSIBLE MEDICAL VALUES IDENTIFIED:]\n"
                        for test_name, value in medical_values:
                            result += f"- {test_name.strip()}: {value}\n"
                else:
                    # Provide image properties when no text is found
                    result += f"Image Properties:\n"
                    result += f"- Dimensions: {img.size[0]} x {img.size[1]} pixels\n"
                    result += f"- Color Mode: {img.mode}\n"
                    result += f"- Format: {img.format}\n"
                    result += "- No readable text found via OCR.\n"
            except Exception as e:
                result += f"OCR processing error: {str(e)}\n"
                # Provide image properties even if OCR fails
                try:
                    img = Image.open(file_path)
                    result += f"Image Properties:\n"
                    result += f"- Dimensions: {img.size[0]} x {img.size[1]} pixels\n"
                    result += f"- Color Mode: {img.mode}\n"
                    result += f"- Format: {img.format}\n"
                except:
                    pass

            return result
        except Exception as e:
            return f"Could not process image file: {os.path.basename(file_path)} (Error: {str(e)})"
    
    def extract_text_from_file(self, file_path):
        """Extract text from various file types with comprehensive error handling"""
        if file_path is None:
            return ""

        try:
            ext = os.path.splitext(file_path.lower())[1]

            if ext == '.pdf':
                return self.read_pdf(file_path)
            elif ext in ['.doc', '.docx']:
                return self.read_docx(file_path)
            elif ext in ['.xls', '.xlsx']:
                return self.read_excel(file_path)
            elif ext == '.csv':
                return self.read_csv(file_path)
            elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.dcm', '.dicom']:
                return self.read_image(file_path)
            elif ext in ['.txt', '.rtf', '.md']:
                # For simple text files
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return f.read()
                except UnicodeDecodeError:
                    # Try with different encoding
                    try:
                        with open(file_path, 'r', encoding='latin-1') as f:
                            return f.read()
                    except:
                        return f"Could not decode text file: {os.path.basename(file_path)}"
            else:
                # For other file types, try to read as text if possible
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return f.read()
                except:
                    return f"File type not supported for text extraction: {os.path.basename(file_path)}"
        except Exception as e:
            return f"Error processing file {os.path.basename(file_path)}: {str(e)}"
    
    def get_patient_files(self, patient_id):
        """Get all files for a specific patient"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM uploaded_files WHERE patient_id = ? ORDER BY upload_date DESC",
            (patient_id,)
        )
        files = cursor.fetchall()
        conn.close()
        return files
    
    def get_file_by_id(self, file_id):
        """Get file by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM uploaded_files WHERE id = ?", (file_id,))
        file_info = cursor.fetchone()
        conn.close()
        return file_info

    def get_file_info_by_name(self, original_filename, patient_id):
        """Get file information from the database by original filename and patient ID."""
        conn = sqlite3.connect(self.db_path)
        # Make the connection return dictionaries
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM uploaded_files WHERE original_filename = ? AND patient_id = ? ORDER BY id DESC LIMIT 1",
            (original_filename, patient_id)
        )
        file_info = cursor.fetchone()
        conn.close()
        # Convert the sqlite3.Row object to a standard dictionary if found
        return dict(file_info) if file_info else None
    
    def update_document_type(self, file_id, document_type):
        """Update document type for a file"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE uploaded_files SET document_type = ? WHERE id = ?",
            (document_type, file_id)
        )
        conn.commit()
        conn.close()
    
    def mark_as_processed(self, file_id):
        """Mark file as processed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE uploaded_files SET processed = TRUE WHERE id = ?",
            (file_id,)
        )
        conn.commit()
        conn.close()
    
    def save_patient_data(self, patient_id, name, age, gender):
        """Save or update patient data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO patient_data 
            (patient_id, name, age, gender, updated_date)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (patient_id, name, age, gender))
        
        conn.commit()
        conn.close()
    
    def save_analysis_result(self, patient_id, clinical_case, clinical_notes, 
                           lab_data, image_findings, documents, document_types,
                           analysis_result, report, summary):
        """Save analysis result to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO analysis_history 
            (patient_id, clinical_case, clinical_notes, lab_data, image_findings,
             documents, document_types, analysis_result, report, summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            patient_id, clinical_case, clinical_notes, lab_data, image_findings,
            documents, document_types, analysis_result, report, summary
        ))
        
        conn.commit()
        conn.close()
    
    def get_analysis_history(self, patient_id):
        """Get analysis history for a patient"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM analysis_history WHERE patient_id = ? ORDER BY analysis_date DESC",
            (patient_id,)
        )
        history = cursor.fetchall()
        conn.close()
        return history
    
    def get_all_extracted_text(self, patient_id):
        """Get all extracted text from PROCESSED files for a patient"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Corrected Bug: Was fetching from processed = FALSE, now fetches from processed = TRUE
        cursor.execute(
            "SELECT extracted_text, document_type, original_filename FROM uploaded_files WHERE patient_id = ? AND processed = TRUE",
            (patient_id,)
        )
        files = cursor.fetchall()
        conn.close()
        
        all_text = ""
        for file_record in files:
            extracted_text, doc_type, filename = file_record
            if extracted_text:
                all_text += f"\n\n--- {filename} ({doc_type or 'Unknown'}) ---\n{extracted_text}"
        
        return all_text

class PatientManager:
    def __init__(self, db_path="medical_data.db"):
        self.db_path = db_path
    
    def create_patient(self, patient_id, name, age, gender):
        """Create a new patient record"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO patient_data 
            (patient_id, name, age, gender)
            VALUES (?, ?, ?, ?)
        ''', (patient_id, name, age, gender))
        
        conn.commit()
        conn.close()
        return patient_id
    
    def get_patient_info(self, patient_id):
        """Get patient information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM patient_data WHERE patient_id = ?",
            (patient_id,)
        )
        patient_info = cursor.fetchone()
        conn.close()
        return patient_info