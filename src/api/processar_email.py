from typing import List, Optional

from backend.main import generate_gemini_response, classify_text
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://focus-mail-two.vercel.app",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class EmailInput(BaseModel):
    textos: Optional[List[str]] = None


@app.post("/processar_email")
async def processar_email(data: EmailInput):
    resultados = []
    if data.textos:
        for idx, texto in enumerate(data.textos):
            categoria = classify_text(texto)
            resposta = generate_gemini_response(categoria, texto)
            resultados.append({
                "filename": f"texto_{idx + 1}",
                "categoria": categoria,
                "resposta_sugerida": resposta,
            })
    return {"resultados": resultados}


handler = Mangum(app)
