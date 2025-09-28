from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Optional
import torch

from utils import extract_text_from_pdf, clean_text

MODEL_PATH = "../model/model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
label_map = model.config.id2label

app = FastAPI(title="Email Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def classify_text(texto: str):
    encodings = tokenizer(texto, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**encodings)
    pred = torch.argmax(outputs.logits, dim=-1).item()
    return label_map[pred]


def generate_response(category: str) -> str:
    if category.lower() == "produtivo":
        return "Obrigado pelo envio, iremos avaliar."
    else:
        return "Agradecemos a mensagem, sem necessidade de ação."


@app.post("/processar_email")
async def processar_email(
    arquivos: Optional[List[UploadFile]] = File(None),
    textos: Optional[List[str]] = Form(None)
):
    resultados = []

    if arquivos:
        for arquivo in arquivos:
            contents = await arquivo.read()
            if not contents:
                resultados.append({
                    "filename": arquivo.filename,
                    "erro": "Arquivo vazio"
                })
                continue

            arquivo.file.seek(0)

            if arquivo.filename.lower().endswith(".txt"):
                texto_extraido = contents.decode("utf-8", errors="ignore")
            elif arquivo.filename.lower().endswith(".pdf"):
                texto_extraido = extract_text_from_pdf(arquivo.file)
            else:
                resultados.append({
                    "filename": arquivo.filename,
                    "erro": "Formato não suportado"
                })
                continue

            texto_limpo = clean_text(texto_extraido)
            categoria = classify_text(texto_limpo)
            resposta = generate_response(categoria)

            resultados.append({
                "filename": arquivo.filename,
                "categoria": categoria,
                "resposta_sugerida": resposta,
                "texto_extraido": texto_limpo[:300]
            })

    if textos:
        for idx, texto in enumerate(textos):
            if not texto.strip():
                resultados.append({
                    "filename": f"texto_{idx+1}",
                    "erro": "Texto vazio"
                })
                continue

            texto_limpo = clean_text(texto)
            categoria = classify_text(texto_limpo)
            resposta = generate_response(categoria)

            resultados.append({
                "filename": f"texto_{idx+1}",
                "categoria": categoria,
                "resposta_sugerida": resposta,
                "texto_extraido": texto_limpo[:300]
            })

    return {"resultados": resultados}