import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from utils import extract_text_from_pdf, preprocess_text
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Optional
from google import genai
import torch

load_dotenv()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))

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
    print(f"ERRO: Não foi possível carregar o modelo em {MODEL_PATH}. {e}")
    label_map = {0: "Produtivo", 1: "Improdutivo"}
    model = None

app = FastAPI(title="Email Classifier API")
app.mount("/static", StaticFiles(directory="../frontend/styles"), name="static")
templates = Jinja2Templates(directory="../frontend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def classify_text(texto: str) -> str:
    if model is None:
        return "Indefinido"

    texto = preprocess_text(texto)

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


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/processar_email_html")
async def processar_email_html(
        request: Request,
        arquivos: Optional[List[UploadFile]] = File(None),
        textos: Optional[List[str]] = Form(None)
):
    resultados = []

    if arquivos:
        for arquivo in arquivos:
            contents = await arquivo.read()
            if not contents:
                resultados.append({"filename": arquivo.filename, "erro": "Arquivo vazio"})
                continue

            arquivo.file.seek(0)

            if arquivo.filename.lower().endswith(".txt"):
                texto = contents.decode("utf-8", errors="ignore")
            elif arquivo.filename.lower().endswith(".pdf"):
                try:
                    texto = extract_text_from_pdf(arquivo.file)
                except Exception as e:
                    resultados.append({"filename": arquivo.filename, "erro": f"Erro na extração do PDF: {e}"})
                    continue
            else:
                resultados.append({"filename": arquivo.filename, "erro": "Formato não suportado"})
                continue

            if not texto.strip():
                resultados.append({"filename": arquivo.filename, "erro": "Não foi possível encontrar texto no arquivo"})
                continue

            categoria = classify_text(texto)
            resposta = generate_gemini_response(categoria, texto)
            resultados.append({
                "filename": arquivo.filename,
                "categoria": categoria,
                "resposta_sugerida": resposta,
                "texto_extraido": texto if len(texto) <= 300 else texto[:300] + "..."
            })

    if textos:
        for idx, texto in enumerate(textos):
            if texto.strip() == "":
                resultados.append(
                    {"filename": f"texto_{idx + 1}", "erro": "Não foi possível encontrar texto no arquivo"})
                continue
            categoria = classify_text(texto)
            resposta = generate_gemini_response(categoria, texto)
            resultados.append({
                "filename": f"texto_{idx + 1}",
                "categoria": categoria,
                "resposta_sugerida": resposta,
                "texto_extraido": texto if len(texto) <= 300 else texto[:300] + "..."
            })

    total = len([r for r in resultados if "erro" not in r])
    produtivos = len([r for r in resultados if r.get("categoria", "").lower() == "produtivo"])
    improdutivos = len([r for r in resultados if r.get("categoria", "").lower() == "improdutivo"])

    porcentagem_produtivos = int((produtivos / total) * 100) if total > 0 else 0
    porcentagem_improdutivos = 100 - porcentagem_produtivos if total > 0 else 0

    return templates.TemplateResponse(
        "resultado.html",
        {
            "request": request,
            "resultados": resultados,
            "porcentagem_produtivos": porcentagem_produtivos,
            "porcentagem_improdutivos": porcentagem_improdutivos
        }
    )
