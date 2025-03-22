FROM python:3.10-slim

# Instale as dependências do sistema
RUN apt-get update && apt-get install -y \
    libsqlite3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    default-jre \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instale as bibliotecas Python necessárias
RUN pip install PyMuPDF tqdm nltk pandas tabula-py pdfminer.six spacy pycorrector textblob sumy torch

# Instale kenlm
RUN pip install https://github.com/kpu/kenlm/archive/master.zip

# Baixe o modelo de linguagem do spaCy
RUN python -m spacy download pt_core_news_sm

# Defina o diretório de trabalho
WORKDIR /app

# Baixe os recursos necessários do nltk
RUN python -m nltk.downloader punkt punkt_tab

# Comando para executar o script
CMD ["python", "teste.py"]


# docker run --rm -v "$(pwd)":/app --memory=8g teste