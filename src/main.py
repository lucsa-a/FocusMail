import os
from typing import List, Optional
from dotenv import load_dotenv
from gradio_client import Client
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from google import genai

from src.utils import extract_text_from_pdf, preprocess_text

load_dotenv()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))

# --- Gemini setup ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("ERRO: A chave GEMINI_API_KEY não foi carregada. Verifique seu arquivo .env.")

try:
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"Aviso: Cliente Gemini não pôde ser inicializado. Respostas automáticas serão estáticas. Erro: {e}")
    client = None

# --- Hugging Face Space / Model API setup ---
try:
    hf_client = Client("lucsaa/FocusMail", hf_token=os.getenv("HF_API_TOKEN"))
except Exception as e:
    print(f"Aviso: Cliente Hugging Face não pôde ser inicializado. Erro: {e}")
    hf_client = None

# --- FastAPI setup ---
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

# --- Funções ---
def classify_text(texto: str) -> str:
    """Classifica o texto usando o Gradio Client."""
    if not texto.strip():
        return "Indefinido"

    texto = preprocess_text(texto)
    
    if hf_client is None:
        return "Indefinido"

    try:
        resultado = hf_client.predict(
            texto=texto,
            api_name="/predict"
        )
        return resultado.capitalize()
    except Exception as e:
        print(f"Erro ao chamar o Hugging Face Space (Gradio Client): {e}")
        return "Indefinido"

def generate_gemini_response(categoria: str, texto_original: str) -> str:
    """Gera uma resposta contextual usando o modelo Gemini ou retorna fallback."""
    fallback_response = "Obrigado pelo envio. Resposta automática dinâmica indisponível."
    if categoria.lower() == "produtivo":
        fallback_response = "Obrigado pelo envio, iremos avaliar e daremos retorno em breve."
    elif categoria.lower() == "improdutivo":
        fallback_response = "Agradecemos a mensagem. Não é necessária nenhuma ação imediata do nosso departamento."

    if client is None:
        return fallback_response

    prompt = f"""
    Você é um assistente de resposta automática de e-mails de uma empresa do setor financeiro.
    O e-mail foi classificado como: **{categoria.upper()}**.

    Instruções:
    1. Seja formal e profissional.
    2. Máx. 3 frases curtas.
    3. Para PRODUTIVO: confirme recebimento e indique próximo passo.
    4. Para IMPRODUTIVO: educado, dispensando ação.
    5. Contextualize ao Corpo do E-mail Original.

    Corpo do E-mail Original:
    ---
    {texto_original[:1000]}
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

# --- Endpoints ---
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
                resultados.append({"filename": f"texto_{idx + 1}", "erro": "Não foi possível encontrar texto no arquivo"})
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