import re

import PyPDF2
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

nltk.data.path.append("./nltk_data") 

stop_words = set(stopwords.words('portuguese'))
stemmer = SnowballStemmer('portuguese')


def extract_text_from_pdf(file) -> str:
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + " "
    return text


def preprocess_text(texto: str) -> str:
    texto = texto.lower()

    texto = re.sub(r'\S+@\S+', '', texto)
    texto = re.sub(r'http\S+', '', texto)
    texto = re.sub(r'\d+', '', texto)

    texto = re.sub(r'[^\w\s]', '', texto)

    palavras = texto.split()
    palavras_processadas = [stemmer.stem(p) for p in palavras if p not in stop_words]

    return " ".join(palavras_processadas)
