from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2

app = FastAPI(title="Email Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_text_from_pdf(file) -> str:
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + " "
    return text

def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text.strip()

@app.post("/processar_email")
async def processar_email(
    arquivo: Optional[UploadFile] = File(None),
    texto: Optional[str] = Form(None),
):
    if arquivo:
        if arquivo.filename.lower().endswith(".txt"):
            contents = await arquivo.read()
            texto_extraido = contents.decode("utf-8", errors="ignore")
        elif arquivo.filename.lower().endswith(".pdf"):
            texto_extraido = extract_text_from_pdf(arquivo.file)
        else:
            return JSONResponse(
                {"erro": "Formato não suportado (.txt ou .pdf apenas)."},
                status_code=400
            )
    elif texto:
        texto_extraido = texto
    else:
        return JSONResponse({"erro": "Nenhum conteúdo fornecido."}, status_code=400)

    texto_limpo = clean_text(texto_extraido)

    categoria = "Produtivo" if "relatório" in texto_limpo.lower() or "reunião" in texto_limpo.lower() else "Improdutivo"
    resposta = "Obrigado pelo envio, iremos avaliar." if categoria == "Produtivo" else "Agradecemos a mensagem, sem necessidade de ação."

    return {
        "categoria": categoria,
        "resposta_sugerida": resposta,
        "texto_extraido": texto_limpo[:500]
    }
