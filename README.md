<p align="center">
  <a href="https://focus-mail-two.vercel.app"><img width="411" height="141" alt="Logo FocusMail" src="https://github.com/user-attachments/assets/881c1ffc-9ff4-4028-a4a1-ce222b7d0414" /></a>
</p>

FocusMail é uma aplicação de IA que classifica e sugera respostas para e-mails recebidos por empresas do setor financeiro.  
O sistema utiliza **FastAPI**, integração com **Hugging Face Spaces** e **Google Gemini** para classificar mensagens como **Produtivas** ou **Improdutivas**, além de sugerir respostas automáticas baseadas na classificação do e-mail.

---

## 🚀 Funcionalidades
- Inserção de corpo de texto de e-mail através da caixa de texto.
- Upload de arquivos `.pdf` ou `.txt` contendo e-mails.
- Classificação automática em **Produtivo** ou **Improdutivo**.
- Geração de respostas automáticas com **Gemini API**.
- Interface web simples para testar a aplicação.
- Deploy local (FastAPI/Uvicorn), em **Vercel**.

---

## 🤖 Modelos de IA

- Classificação de e-mails usando um [modelo treinado e hospedado no **Hugging Face Spaces**](https://huggingface.co/spaces/lucsaa/FocusMail).
- Respostas automáticas geradas com **Google Gemini**, ajustadas com prompts específicos para cada categoria de e-mail.
- Pré-processamento de texto com **NLTK** (remoção de stopwords, stemming, limpeza de caracteres especiais).

---

## 🔧 Guia de Instalação

### 1. Instalar Python

O projeto requer Python 3.8 ou superior: [https://www.python.org/downloads/](https://www.python.org/downloads/)

### 2. Instalar as dependências

Com o Python instalado, instale as dependências do projeto

```bash
pip install -r requirements.txt
# ou
pip3 install -r requirements.txt
```

---

## ▶️ Executando a Aplicação

### Local

```bash
uvicorn src.main:app --reload
```

Acesse em: ```http://localhost:8000```

### Remoto

Acesse em: [https://focus-mail-two.vercel.app](https://focus-mail-two.vercel.app)

---

## 💻 Tecnologias Utilizadas

<!-- Linguagem -->
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

<!-- Frameworks e APIs -->
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Gradio](https://img.shields.io/badge/Gradio-FF6C37?style=for-the-badge&logo=gradio&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)

<!-- Bibliotecas Python -->
![PyPDF2](https://img.shields.io/badge/PyPDF2-4B8BBE?style=for-the-badge&logo=python&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-FF6F61?style=for-the-badge&logo=python&logoColor=white)
![python-dotenv](https://img.shields.io/badge/dotenv-2D6A4F?style=for-the-badge&logo=python&logoColor=white)
![Jinja2](https://img.shields.io/badge/Jinja2-B41717?style=for-the-badge&logo=jinja&logoColor=white)

<!-- Frontend -->
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)

<!-- Middleware e Deploy -->
![CORS](https://img.shields.io/badge/CORS-4B0082?style=for-the-badge)
![Mangum](https://img.shields.io/badge/Mangum-2C3E50?style=for-the-badge)
![Vercel](https://img.shields.io/badge/Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)
