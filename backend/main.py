from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Email Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/processar_email")
async def processar_email(
    arquivo: Optional[UploadFile] = File(None),
    texto: Optional[str] = Form(None),
):
    if arquivo:
        contents = await arquivo.read()
        if arquivo.filename.lower().endswith(".txt"):
            texto_extraido = contents.decode("utf-8", errors="ignore")
        else:
            texto_extraido = "[conteúdo de arquivo — parser PDF a implementar]"
    elif texto:
        texto_extraido = texto
    else:
        return JSONResponse({"erro": "Nenhum conteúdo fornecido."}, status_code=400)

    categoria = "Produtivo" if "relatório" in texto_extraido.lower() or "reunião" in texto_extraido.lower() else "Improdutivo"
    resposta = "Obrigado pelo envio, iremos avaliar." if categoria == "Produtivo" else "Agradecemos a mensagem, sem necessidade de ação."

    return {"categoria": categoria, "resposta_sugerida": resposta}
