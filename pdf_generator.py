from fpdf import FPDF

def generate_pdf_report(text: str) -> bytes:
    """
    Generate a PDF report from plain text and return as bytes.

    Args:
        text (str): The report text.

    Returns:
        bytes: The PDF content as bytes.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    # Split and add lines
    for line in text.split('\n'):
        pdf.multi_cell(0, 10, line)

    # Return as bytes
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    return pdf_bytes
