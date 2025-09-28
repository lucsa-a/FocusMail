import PyPDF2

def extract_text_from_pdf(file) -> str:
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + " "
    return text

def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text.strip()