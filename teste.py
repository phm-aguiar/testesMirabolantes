import fitz  # PyMuPDF
import re
import spacy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from textblob import TextBlob
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

# Verifica se o pycorrector está instalado e funciona
try:
    import pycorrector
    PY_CORRECTOR_AVAILABLE = True
except ImportError:
    PY_CORRECTOR_AVAILABLE = False
    print("Aviso: pycorrector não está instalado. A correção de texto será ignorada.")

# Funções de extração de texto
def extract_with_pymupdf(pdf_path):
    """Extrai texto de um PDF usando PyMuPDF."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Funções de limpeza e processamento de texto
def clean_text(text):
    """Limpa o texto removendo caracteres especiais e espaços em branco desnecessários."""
    # Remove múltiplos espaços em branco
    text = re.sub(r'\s+', ' ', text)
    # Remove caracteres especiais (exceto letras, números e pontuação básica)
    text = re.sub(r'[^\w\s.,;!?]', '', text)
    # Normaliza espaços em branco
    text = text.strip()
    return text

def remove_repeticoes(text):
    """Remove trechos repetitivos como cabeçalhos, rodapés e sumários."""
    lines = text.split("\n")
    seen = set()
    cleaned_lines = []

    for line in lines:
        normalized_line = re.sub(r'\s+', ' ', line.strip())  # Normaliza espaços
        if normalized_line and normalized_line not in seen:
            seen.add(normalized_line)
            cleaned_lines.append(line.strip())

    return "\n".join(cleaned_lines)

def format_text(text):
    """Formata o texto para melhor legibilidade."""
    text = re.sub(r'\s+', ' ', text)  # Remove espaços extras
    text = re.sub(r'\n+', '\n', text)  # Remove quebras de linha excessivas
    text = text.strip()
    return text

def split_into_sections(text, section_titles):
    """Divide o texto em seções com base em títulos predefinidos."""
    sections = {}
    for title in section_titles:
        pattern = re.compile(rf'{title}(.*?)(?=\n[A-Z])', re.DOTALL)
        match = pattern.search(text)
        if match:
            sections[title] = match.group(1).strip()
    return sections

def extract_entities(text):
    """Extraia entidades nomeadas do texto usando spaCy."""
    try:
        nlp = spacy.load("pt_core_news_sm")
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities
    except OSError:
        print("Aviso: Modelo de linguagem 'pt_core_news_sm' do spaCy não está instalado.")
        return []

def summarize_text(text, sentences_count=5):
    """Resume o texto em um número específico de frases."""
    parser = PlaintextParser.from_string(text, Tokenizer("portuguese"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return " ".join([str(sentence) for sentence in summary])

def correct_text(text):
    """Corrige erros de OCR no texto."""
    if PY_CORRECTOR_AVAILABLE:
        corrector = pycorrector.Corrector()
        corrected_text = corrector.correct(text)
        return corrected_text
    else:
        return text

def analyze_sentiment(text):
    """Analisa o sentimento do texto usando TextBlob."""
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return sentiment

def count_tokens(text):
    """Conta o número de tokens no texto usando spaCy."""
    try:
        nlp = spacy.load("pt_core_news_sm")
        doc = nlp(text)
        return len(doc)
    except OSError:
        print("Aviso: Modelo de linguagem 'pt_core_news_sm' do spaCy não está instalado.")
        return 0

# Funções de pré-processamento
def remove_repetitive_text(text):
    lines = text.split("\n")
    seen = set()
    cleaned_lines = []

    for line in lines:
        normalized_line = re.sub(r'\s+', ' ', line.strip())
        if normalized_line and normalized_line not in seen:
            seen.add(normalized_line)
            cleaned_lines.append(line.strip())

    return "\n".join(cleaned_lines)

def remove_glossary(text):
    glossary_pattern = re.compile(r'GLOSSÁRIO.*?(?=\n[A-Z])', re.DOTALL)
    return glossary_pattern.sub('', text)

def remove_tables(text):
    table_pattern = re.compile(r'(\b\d+\b\s*){3,}')
    return table_pattern.sub('', text)

def preprocess_text(text):
    text = remove_repetitive_text(text)
    text = remove_glossary(text)
    text = remove_tables(text)
    return text

# Função para resumir com LLM
def summarize_with_llm(text, max_tokens=500):
    model_name = "huggyllama/llama-7b"  # Substitua pelo modelo desejado
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs.input_ids, max_length=max_tokens, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Caminho do PDF (substitua pelo caminho correto no container)
pdf_path = "janeiro-tgar11.pdf"

# Fluxo principal
print("Extraindo texto do PDF com PyMuPDF...")
text_pymupdf = extract_with_pymupdf(pdf_path)

print("Limpando o texto extraído...")
cleaned_text = clean_text(text_pymupdf)
cleaned_text = remove_repeticoes(cleaned_text)
cleaned_text = format_text(cleaned_text)

print("Pré-processando o texto...")
cleaned_text = preprocess_text(cleaned_text)

print("Segmentando o texto em seções...")
section_titles = ["OBJETIVO DO FUNDO", "CARTA DO GESTOR", "RESUMO DO MÊS"]
sections = split_into_sections(cleaned_text, section_titles)

print("Extraindo entidades nomeadas...")
entities = extract_entities(cleaned_text)

print("Sumarizando o texto...")
summary = summarize_text(cleaned_text)

print("Corrigindo erros de OCR (se aplicável)...")
corrected_text = correct_text(cleaned_text)

print("Analisando o sentimento do texto...")
sentiment = analyze_sentiment(cleaned_text)

# Contar tokens
token_count = count_tokens(cleaned_text)

# Resumo com LLM
llm_summary = summarize_with_llm(cleaned_text)
print("Resumo gerado pela LLM:\n", llm_summary)

# Criar contexto estruturado
context = {
    "cleaned_text": cleaned_text,
    "sections": sections,
    "entities": entities,
    "summary": summary,
    "corrected_text": corrected_text,
    "sentiment": sentiment,
    "token_count": token_count,  # Adiciona o número de tokens ao contexto
    "llm_summary": llm_summary  # Adiciona o resumo gerado pela LLM
}

# Salvar o contexto em um arquivo JSON
with open("context.json", "w", encoding="utf-8") as f:
    json.dump(context, f, ensure_ascii=False, indent=4)

print(f"Processamento concluído! Contexto salvo em 'context.json'.")
print(f"Número de tokens no texto: {token_count}")



