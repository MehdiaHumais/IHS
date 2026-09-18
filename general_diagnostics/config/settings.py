import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BASE_DIR.parent

for env_file in (
    PROJECT_DIR / ".env",
    BASE_DIR / ".env",
    PROJECT_DIR / "clinical_chatbot" / ".env",
):
    if env_file.is_file():
        load_dotenv(env_file, override=False)

class Settings:

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # File upload settings
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS = {
        'images': ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp'],
        'documents': ['pdf', 'doc', 'docx', 'xls', 'xlsx', 'csv', 'txt', 'rtf', 'md'],
        'medical': ['dcm', 'dicom']
    }

    # Supported medical image types
    MEDICAL_IMAGE_TYPES = [
        'X-Ray', 'CT Scan', 'MRI', 'Ultrasound',
        'PET Scan', 'Mammography', 'DICOM',
        'Endoscopy', 'Pathology Slides'
    ]

    # Report sections
    REPORT_SECTIONS = [
        'Patient Information',
        'Clinical Findings',
        'Laboratory Results',
        'Imaging Results',
        'Diagnosis',
        'Recommendations',
        'Follow-up Plan'
    ]

    # LLM Settings
    LLM_MODEL_PREFERENCES = {
        'primary': 'openai',  # Options: 'openai', 'groq'
        'fallback': 'groq'
    }

    # Default models
    DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"
    DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"

settings = Settings()