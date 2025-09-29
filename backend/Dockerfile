# === Etapa 1: Base com Python 3.11 slim ===
FROM python:3.11-slim

# Definindo diretório de trabalho
WORKDIR /app

# Evita prompts interativos e mensagens de apt
ENV DEBIAN_FRONTEND=noninteractive

# Atualiza e instala dependências do sistema (para PDFs, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpoppler-cpp-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copia requirements
COPY requirements.txt .

# Instala dependências Python
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copia todo o código
COPY . .

# Expõe porta padrão do FastAPI
EXPOSE 8000

# Comando de start com Gunicorn + Uvicorn
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:10000}"]