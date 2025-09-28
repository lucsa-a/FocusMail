import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Optional
import torch

from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("ERRO: A chave GEMINI_API_KEY não foi carregada. Verifique seu arquivo .env.")
    pass

try:
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"Aviso: Cliente Gemini não pôde ser inicializado. Respostas automáticas serão estáticas. Erro: {e}")
    client = None

MODEL_PATH = "lucsaa/classificador-de-emails"
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    label_map = model.config.id2label
except Exception as e:
    print(f"ERRO: Não foi possível carregar o modelo de classificação local em {MODEL_PATH}. {e}")
    label_map = {0: "Produtivo", 1: "Improdutivo"}
    model = None

app = FastAPI(title="Email Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def classify_text(texto: str) -> str:
    """Classifica o texto do email como Produtivo ou Improdutivo."""
    if model is None:
        return "Indefinido"

    encodings = tokenizer(texto, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**encodings)
    pred = torch.argmax(outputs.logits, dim=-1).item()
    return label_map.get(pred, "Indefinido")


def generate_gemini_response(categoria: str, texto_original: str) -> str:
    """Gera uma resposta contextual usando o modelo Gemini ou retorna um fallback."""
    
    fallback_response = "Obrigado pelo envio. Recebemos sua mensagem, mas a resposta automática dinâmica está indisponível."
    if categoria.lower() == "produtivo":
        fallback_response = "Obrigado pelo envio, iremos avaliar e daremos retorno em breve."
    elif categoria.lower() == "improdutivo":
        fallback_response = "Agradecemos a mensagem. Não é necessária nenhuma ação imediata do nosso departamento."

    if client is None:
        return fallback_response

    prompt = f"""
    Você é um assistente de resposta automática de e-mails de uma empresa do setor financeiro.
    O e-mail foi classificado como: **{categoria.upper()}**.

    Instruções para a resposta:
    1. Seja formal e profissional (setor financeiro).
    2. A resposta deve ter no máximo 3 frases curtas.
    3. Se a categoria for **PRODUTIVO**, a resposta deve confirmar o recebimento e indicar o próximo passo de forma profissional.
    4. Se a categoria for **IMPRODUTIVO**, a resposta deve ser educada e dispensar a necessidade de ação ou, se possível, direcionar para um canal não-corporativo.
    5. A resposta deve ser contextualizada ao Corpo do E-mail Original.

    Corpo do E-mail Original:
    ---
    {texto_original[:1000]} # Limita o texto para economizar tokens
    ---

    Gere APENAS o corpo da resposta:
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text.strip()
        
    except Exception as e:
        print(f"Erro ao chamar a API Gemini: {e}")
        return fallback_response

@app.post("/processar_email")
async def processar_email(
    arquivos: Optional[List[UploadFile]] = File(None),
    textos: Optional[List[str]] = Form(None)
):
    resultados = []

    if model is None:
        resultados.append({"aviso": "O modelo de classificação local não foi carregado. A classificação retornará 'Indefinido'."})

    def clean_text(text):
         return text.strip() 
    
    def extract_text_from_pdf(file):
        return "Texto extraído de um PDF de teste."

    if arquivos:
        for arquivo in arquivos:
            contents = await arquivo.read()
            if not contents:
                resultados.append({"filename": arquivo.filename, "erro": "Arquivo vazio"})
                continue

            arquivo.file.seek(0)

            texto_extraido = ""
            if arquivo.filename.lower().endswith(".txt"):
                texto_extraido = contents.decode("utf-8", errors="ignore")
            elif arquivo.filename.lower().endswith(".pdf"):
                try:
                    texto_extraido = extract_text_from_pdf(arquivo.file)
                except Exception as e:
                    resultados.append({"filename": arquivo.filename, "erro": f"Erro na extração do PDF: {e}"})
                    continue
            else:
                resultados.append({"filename": arquivo.filename, "erro": "Formato não suportado"})
                continue

            texto_limpo = clean_text(texto_extraido)
            categoria = classify_text(texto_limpo)
            resposta = generate_gemini_response(categoria, texto_limpo)

            resultados.append({
                "filename": arquivo.filename,
                "categoria": categoria,
                "resposta_sugerida": resposta,
                "texto_extraido": texto_limpo[:300] + "..."
            })

    if textos:
        for idx, texto in enumerate(textos):
            if not texto.strip():
                resultados.append({"filename": f"texto_{idx+1}", "erro": "Texto vazio"})
                continue

            texto_limpo = clean_text(texto)
            categoria = classify_text(texto_limpo)
            resposta = generate_gemini_response(categoria, texto_limpo)

            resultados.append({
                "filename": f"texto_{idx+1}",
                "categoria": categoria,
                "resposta_sugerida": resposta,
                "texto_extraido": texto_limpo[:300] + "..."
            })

    return {"resultados": resultados}