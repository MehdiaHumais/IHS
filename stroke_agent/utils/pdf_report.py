from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


# ==========================
# GENERATE PDF REPORT
# ==========================

def generate_pdf(report_data):

    file_name = str(Path(__file__).resolve().parents[1] / "Stroke_Risk_Report.pdf")

    pdf = SimpleDocTemplate(
        file_name,
        pagesize=letter
    )


    styles = getSampleStyleSheet()

    content = []


    title = Paragraph(
        "Brain Stroke Diagnosis - Stroke Risk Report",
        styles["Title"]
    )

    content.append(title)

    content.append(
        Spacer(1, 20)
    )


    for key, value in report_data.items():

        text = f"{key}: {value}"

        content.append(
            Paragraph(
                text,
                styles["Normal"]
            )
        )

        content.append(
            Spacer(1, 10)
        )


    content.append(
        Paragraph(
            """
            Disclaimer: This report is for educational
            purposes only and does not replace professional
            medical advice.
            """,
            styles["Italic"]
        )
    )


    pdf.build(content)


    return file_name