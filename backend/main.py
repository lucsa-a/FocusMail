from typing import List, Optional
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

def classify_email(texto: str) -> str:
    return "Produtivo" if "relatório" in texto.lower() or "reunião" in texto.lower() else "Improdutivo"

def generate_response(category: str) -> str:
    return "Obrigado pelo envio, iremos avaliar." if category == "Produtivo" else "Agradecemos a mensagem, sem necessidade de ação."

@app.post("/processar_email")
async def processar_email(
    arquivos: Optional[List[UploadFile]] = File(None),
    textos: Optional[List[str]] = Form(None)
):
    resultados = []

    if arquivos:
        for arquivo in arquivos:
            if arquivo.filename.lower().endswith(".txt"):
                contents = await arquivo.read()
                texto_extraido = contents.decode("utf-8", errors="ignore")
            elif arquivo.filename.lower().endswith(".pdf"):
                texto_extraido = extract_text_from_pdf(arquivo.file)
            else:
                resultados.append({
                    "filename": arquivo.filename,
                    "erro": "Formato não suportado (.txt ou .pdf apenas)."
                })
                continue

            texto_limpo = clean_text(texto_extraido)
            categoria = classify_email(texto_limpo)
            resposta = generate_response(categoria)

            resultados.append({
                "filename": arquivo.filename,
                "categoria": categoria,
                "resposta_sugerida": resposta,
                "texto_extraido": texto_limpo[:500]
            })
            
    if textos:
        for idx, texto in enumerate(textos):
            texto_limpo = clean_text(texto)
            categoria = classify_email(texto_limpo)
            resposta = generate_response(categoria)
            resultados.append({
                "filename": f"texto_{idx+1}",
                "categoria": categoria,
                "resposta_sugerida": resposta,
                "texto_extraido": texto_limpo[:500]
            })

    if not resultados:
        return JSONResponse({"erro": "Nenhum conteúdo fornecido."}, status_code=400)

    return {"resultados": resultados}