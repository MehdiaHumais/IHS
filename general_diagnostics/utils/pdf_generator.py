from fpdf import FPDF
from datetime import datetime
import os
import textwrap

class PDFReportGenerator:
    def __init__(self):
        pass

    def generate_pdf_report(self, report_content, patient_info=None):
        """
        Generate a PDF report from the diagnostic analysis

        Args:
            report_content (str): The formatted diagnostic report content
            patient_info (dict): Optional patient information dictionary

        Returns:
            bytes: PDF file content as bytes
        """
        try:
            # Create a new PDF document
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=12)
            pdf.add_page()

            # Set margins and calculate effective width
            pdf.set_margins(12, 12, 12)
            effective_width = pdf.w - 2 * pdf.l_margin  # Page width minus margins

            # Professional header with better clarity
            pdf.set_font("Arial", "B", 18)
            pdf.set_fill_color(200, 220, 255)  # Light blue background for header
            pdf.cell(effective_width, 12, "MEDICAL DIAGNOSTIC REPORT", 0, 1, 'C', True)
            pdf.ln(4)  # Space after header

            # Add patient info in a structured format with better clarity
            if patient_info:
                pdf.set_font("Arial", "B", 12)
                pdf.set_fill_color(230, 230, 230)  # Light gray background for section headers
                pdf.cell(effective_width, 8, "PATIENT INFORMATION", 0, 1, 'L', True)
                pdf.ln(3)

                pdf.set_font("Arial", "", 11)  # Slightly larger font for better readability

                # Format patient info in a table-like structure with better spacing
                for key, value in patient_info.items():
                    safe_key = str(key).replace('_', ' ').title()
                    safe_val = str(value)
                    text = f"{safe_key}: {safe_val}"

                    # Ensure text is compatible with PDF
                    try:
                        safe_text = text.encode('ascii', 'ignore').decode('ascii')
                    except Exception:
                        safe_text = repr(text).encode('ascii', 'ignore').decode('ascii')

                    # Use cell for better structure
                    pdf.set_fill_color(245, 245, 245)  # Very light gray for each row
                    pdf.cell(5, 6, "", 0, 0)  # Small indentation
                    pdf.cell(effective_width - 5, 6, safe_text, 0, 1, 'L', True)  # With background fill
                    pdf.set_fill_color(255, 255, 255)  # Reset to white

                pdf.ln(4)  # Space after patient info

            # Add generated timestamp with better clarity
            pdf.set_font("Arial", "I", 9)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            timestamp_text = f"Generated: {timestamp}"
            try:
                safe_timestamp = timestamp_text.encode('ascii', 'ignore').decode('ascii')
            except Exception:
                safe_timestamp = repr(timestamp_text).encode('ascii', 'ignore').decode('ascii')

            # Right-align the timestamp
            pdf.cell(effective_width, 5, safe_timestamp, 0, 1, 'R')
            pdf.ln(4)  # Space before content

            # Clean the report content first
            clean_content = self._clean_report_content(report_content)

            # Parse the medical report into structured sections
            sections = self._parse_medical_report(clean_content)

            # Process each section with enhanced clarity
            for section_title, section_content in sections:
                if section_title:
                    # Process section title as header with better clarity
                    header_text = section_title.strip()

                    if header_text and len(header_text) < 100:  # Make sure it's a real header, not just symbols
                        pdf.ln(3)  # Space before header
                        pdf.set_font("Arial", "B", 13)  # Larger, bolder section headers
                        pdf.set_fill_color(220, 230, 255)  # Light blue background for section headers
                        # Ensure header text fits within page width
                        line_width = pdf.w - 2 * pdf.l_margin  # Page width minus margins
                        display_header = header_text[:int(line_width / 1.2)] if len(header_text) > int(line_width / 1.2) else header_text  # Truncate if too long
                        pdf.cell(line_width, 7, display_header, 0, 1, 'L', True)
                        pdf.set_font("Arial", "", 11)  # Reset font for content
                        pdf.set_fill_color(255, 255, 255)  # Reset to white
                        pdf.ln(2)  # Small space after header

                if section_content:
                    # Process section content with better clarity
                    content_lines = section_content.split('\n')

                    for line_idx, line in enumerate(content_lines):
                        safe_line = line.strip()  # Remove leading and trailing whitespace

                        # Handle empty lines to create clear paragraph breaks
                        if not safe_line:
                            # Add more space for paragraph breaks
                            pdf.ln(3)
                            continue

                        # Check if it's a list item (starts with dash, number, or bullet)
                        is_list_item = safe_line.startswith(('-', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '* ', '- ', '• '))

                        if is_list_item:
                            # Add indentation for list items with better clarity
                            line_width = pdf.w - 2 * pdf.l_margin  # Page width minus margins
                            indent_space = 10  # More pronounced indentation

                            # Use multi_cell for proper text wrapping within available width
                            pdf.set_font("Arial", "", 11)  # Consistent font
                            pdf.cell(indent_space, 6, "", 0, 0)  # Indentation space
                            # Calculate available width for the actual text
                            available_width = line_width - indent_space

                            # Use multi_cell for proper text wrapping
                            pdf.multi_cell(w=available_width, h=6, txt=safe_line, border=0, align='L')
                            # Add small space after list items for clarity
                            pdf.ln(1)
                        else:
                            # Add the line with proper formatting for better clarity
                            line_width = pdf.w - 2 * pdf.l_margin  # Page width minus margins

                            # Use multi_cell for consistent text wrapping
                            pdf.set_font("Arial", "", 11)  # Slightly larger font for better readability

                            # Handle very long lines by breaking them properly
                            if len(safe_line) > 120:  # If line is very long
                                # Break into multiple lines with proper word wrapping
                                pdf.multi_cell(w=line_width, h=6, txt=safe_line, border=0, align='L')
                            else:
                                # For normal lines, use single line if it fits
                                text_width = pdf.get_string_width(safe_line)
                                if text_width <= line_width:
                                    pdf.cell(line_width, 6, safe_line, 0, 1, 'L')
                                else:
                                    # If too wide, use multi_cell
                                    pdf.multi_cell(w=line_width, h=6, txt=safe_line, border=0, align='L')

                        # Add consistent small spacing between content lines
                        pdf.ln(1)

        except Exception as e:
            print(f"Error generating PDF: {e}")
            # Return an error indicator or handle gracefully
            pdf = FPDF()
            pdf.add_page()
            pdf.set_margins(12, 12, 12)
            effective_width = pdf.w - 2 * pdf.l_margin  # Page width minus margins
            pdf.set_font("Arial", "B", 14)
            pdf.set_fill_color(255, 200, 200)  # Light red background for error
            pdf.cell(effective_width, 10, "ERROR GENERATING REPORT", 0, 1, 'C', True)
            pdf.set_font("Arial", "", 12)
            pdf.set_fill_color(255, 255, 255)  # Reset to white
            pdf.ln(10)
            # Also wrap the error message itself
            pdf.multi_cell(effective_width, 8, f"Error: {str(e)}", 0, 'L')

        # fpdf2 returns bytearray; Streamlit download_button requires bytes.
        return bytes(pdf.output())

    def _clean_report_content(self, content):
        """Clean the report content by removing formatting symbols and encoding properly"""
        if not content:
            return ""

        # Remove markdown formatting symbols
        clean_content = (content
                        .replace('#', '')
                        .replace('*', '- ')
                        .replace('•', '- ')
                        .replace('→', ' -> ')
                        .replace('≥', ' >= ')
                        .replace('≤', ' <= ')
                        .replace('α', 'a')
                        .replace('β', 'b')
                        .replace('γ', 'g')
                        .replace('–', '-')
                        .replace('—', '-')
                        .replace('"', '"')
                        .replace('"', '"')
                        .replace("'", "'")
                        .replace("'", "'")
                        .strip())

        # Encode to ASCII to remove problematic characters
        try:
            clean_content = clean_content.encode('ascii', 'ignore').decode('ascii')
        except:
            clean_content = repr(clean_content).encode('ascii', 'ignore').decode('ascii')

        return clean_content

    def _parse_medical_report(self, content):
        """Parse the medical report content into structured sections"""
        if not content:
            return []

        lines = content.split('\n')
        sections = []
        current_section_title = ""
        current_section_content = []

        # Common medical section headers keywords
        section_keywords = [
            'CLINICAL', 'PATIENT', 'DIAGNOSIS', 'TREATMENT', 'ASSESSMENT',
            'RECOMMENDATION', 'LABORATORY', 'IMAGING', 'FINDINGS',
            'HISTORY', 'EXAMINATION', 'SUMMARY', 'PLAN', 'PROGNOSIS',
            'CLINICAL NOTES', 'LABORATORY RESULTS', 'IMAGING FINDINGS',
            'SYMPTOMS', 'MEDICATION', 'PRESCRIPTION', 'VITALS', 'OBSERVATIONS',
            'DIAGNOSTIC', 'TESTS', 'RESULTS', 'CONDITION', 'THERAPY'
        ]

        for line in lines:
            clean_line = line.strip()

            # Check if this line is a section header - improved detection
            is_header = any(keyword.upper() in clean_line.upper() for keyword in section_keywords) and len(clean_line) < 80 and clean_line.isupper()

            # Also check for markdown-style headers
            if clean_line.startswith(('##', '###')) or (clean_line and all(c in ['=', '-', '#'] for c in clean_line.strip()) and len(clean_line.strip()) > 3):
                is_header = True
                # Extract the actual header text from markdown-style
                if clean_line.startswith('##'):
                    clean_line = clean_line[2:].strip()
                elif clean_line.startswith('###'):
                    clean_line = clean_line[3:].strip()
                elif clean_line.startswith(('===', '---')):
                    # This was likely an underline header, we need to get the previous line
                    pass  # Handle this case by looking at previous content

            if is_header and clean_line and not all(c in ['=', '-', '#'] for c in clean_line.strip()):
                # Save previous section if it exists
                if current_section_title and current_section_content:
                    sections.append((current_section_title, '\n'.join(current_section_content)))

                # Start new section
                current_section_title = clean_line
                current_section_content = []
            else:
                # Add to current section content
                current_section_content.append(line)  # Keep original line to preserve structure

        # Add the last section if it exists
        if current_section_title and current_section_content:
            sections.append((current_section_title, '\n'.join(current_section_content)))
        elif current_section_content and not any(sections):  # If no sections were identified, put everything in one section
            sections.append(("", '\n'.join(content.split('\n'))))

        return sections

    def save_pdf_to_file(self, report_content, patient_info=None, file_path=None):
        """
        Save the PDF report to a file

        Args:
            report_content (str): The formatted diagnostic report content
            patient_info (dict): Optional patient information dictionary
            file_path (str): Path to save the PDF file (optional)

        Returns:
            str: Path to saved PDF file
        """
        if not file_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"diagnostic_report_{timestamp}.pdf"
            file_path = os.path.join("static", "reports", filename)

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

        pdf_bytes = self.generate_pdf_report(report_content, patient_info)

        with open(file_path, 'wb') as f:
            f.write(pdf_bytes)

        return file_path