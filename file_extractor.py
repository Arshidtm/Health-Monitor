import io
from typing import Optional
from PIL import Image
import pytesseract
import PyPDF2
import docx
import tempfile
import fitz  # PyMuPDF for advanced PDF handling including images inside PDFs

def extract_text_from_pdf(file) -> str:
    """
    Extract text from PDF file.
    If PDF is image-based, perform OCR on each page.
    """
    text = ""
    try:
        # Try extracting text normally with PyPDF2
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        if text.strip():
            return text.strip()
    except Exception:
        pass  # fallback to OCR if normal extraction fails

    # If normal extraction fails or no text found, try OCR on images
    try:
        # Use PyMuPDF to open file (support images inside PDF)
        file.seek(0)
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_text = pytesseract.image_to_string(img)
            text += ocr_text + "\n"
        return text.strip()
    except Exception as e:
        return ""


def extract_text_from_docx(file) -> str:
    """
    Extract text from DOCX file.
    """
    try:
        doc = docx.Document(io.BytesIO(file.read()))
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception:
        return ""


def extract_text_from_txt(file) -> str:
    """
    Extract text from plain text file.
    """
    try:
        return file.getvalue().decode("utf-8")
    except Exception:
        return ""


def extract_text_from_image(file) -> str:
    """
    Extract text from image file using OCR.
    """
    try:
        image = Image.open(file)
        text = pytesseract.image_to_string(image)
        return text
    except Exception:
        return ""


def extract_text_from_file(file, filename: Optional[str] = None) -> str:
    """
    Unified extractor for various file types including PDF, DOCX, TXT, and images.

    Args:
        file: Uploaded file-like object (BytesIO)
        filename: Optional, used to guess file type by extension

    Returns:
        Extracted text as string, empty string if extraction fails
    """
    if filename is None:
        filename = getattr(file, 'name', '')

    filename = filename.lower()

    if filename.endswith(".pdf"):
        return extract_text_from_pdf(file)
    elif filename.endswith(".docx"):
        return extract_text_from_docx(file)
    elif filename.endswith(".txt"):
        return extract_text_from_txt(file)
    elif filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
        return extract_text_from_image(file)
    else:
        # Try to guess from mime type or fallback
        try:
            # Assume text for unknown file
            return extract_text_from_txt(file)
        except Exception:
            return ""
