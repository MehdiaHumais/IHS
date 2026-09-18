# Medical Diagnostic Agent

A comprehensive AI-powered clinical diagnostic agent that analyzes patient data, laboratory results, medical images, and clinical documents to generate diagnostic reports.

## Features

- Support for various file types: PDF, DOCX, XLSX, CSV, images (JPG, PNG), DICOM
- Multi-model AI support (OpenAI and Groq)
- Medical data parsing and structured analysis
- Detailed diagnostic reports with recommendations
- Patient data management and history tracking

## Prerequisites

- Python 3.8 or higher
- OpenAI API key (optional but recommended)
- Groq API key (as fallback option)

## Installation

1. **Clone or download the project**

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - On Windows: `venv\Scripts\activate`
   - On macOS/Linux: `source venv/bin/activate`

4. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Install additional dependencies for OCR (optional but recommended):**
   ```bash
   # For Windows, you may need to install Tesseract manually from:
   # https://github.com/UB-Mannheim/tesseract/wiki
   
   # Then install Python wrapper:
   pip install pytesseract pdf2image
   ```

## Configuration

1. **Create a `.env` file** in the project root directory with your API keys:

   ```
   OPENAI_API_KEY=your_openai_api_key_here
   GROQ_API_KEY=your_groq_api_key_here
   ```

2. **API Key Options:**
   - **OpenAI API Key**: Get from [OpenAI Dashboard](https://platform.openai.com/api-keys)
   - **Groq API Key**: Get from [Groq Cloud](https://console.groq.com/keys)
   
   If you don't have either key, the system will attempt to analyze data locally with reduced capabilities.

## Usage

1. **Run the application:**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Enter patient information** in the Patient Information section

4. **Add clinical data:**
   - Clinical case description
   - Clinical notes/messages
   - Upload medical documents (PDF, DOCX)
   - Upload lab results (XLSX, CSV)
   - Upload medical images (JPG, PNG, DICOM)

5. **Click "Analyze Clinical Case"** to generate the diagnostic report

## File Types Supported

- **Documents**: PDF, DOCX, DOC
- **Spreadsheets**: XLSX, XLS, CSV
- **Images**: JPG, JPEG, PNG, GIF, BMP, TIFF, WEBP
- **Medical Images**: DICOM, medical reports as images
- **Text Files**: TXT, RTF, MD

## Troubleshooting

### Common Issues

1. **"Incorrect API key provided" Error**:
   - Verify your API keys in the `.env` file
   - Ensure the keys are valid and not expired
   - Check for any extra spaces or characters in the keys

2. **File Upload Issues**:
   - Ensure files are within the 50MB size limit
   - Verify file extensions are supported
   - Check that files are not corrupted

3. **Image/Document Processing Issues**:
   - Install Tesseract OCR for better text extraction from images
   - Some image types may require additional processing libraries

### OCR Setup (Windows)

If you plan to process images or scanned PDFs:

1. Download and install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
2. Add Tesseract to your system PATH
3. Restart your terminal/command prompt

## Data Security

- All patient data is stored locally in `medical_data.db`
- API keys are stored only in the `.env` file and not in the code
- No data is transmitted to external servers except for AI model processing

## Technologies Used

- Streamlit (UI)
- LangChain (AI orchestration)
- OpenAI/Groq (AI models)
- Pandas (data processing)
- PyPDF2 (PDF processing)
- OpenCV (image processing)
- SQLite (data storage)

## Note

This application is designed for educational and research purposes. The diagnostic reports generated should be reviewed by qualified healthcare professionals before any clinical decisions are made.