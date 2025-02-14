import os
from fastapi import FastAPI, UploadFile, File
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import UnstructuredFileLoader
import requests
from PyPDF2 import PdfReader
import pandas as pd
from dotenv import load_dotenv
import torch
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables from .env
load_dotenv()

# Access your Gemini API key securely
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI()

# Initialize the FinBERT model
finbert_model = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(finbert_model)
model = AutoModelForSequenceClassification.from_pretrained(finbert_model)

# FAISS for RAG
vectorstore = None

UPLOAD_FOLDER = "data/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Helper function to extract text from PDF/TXT/CSV
def extract_text(file_path):
    if file_path.endswith('.pdf'):
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
        return text
    elif file_path.endswith('.txt'):
        with open(file_path, 'r') as f:
            return f.read()
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        return df.to_string()
    return ''

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """ Save file & create vector embeddings for RAG. """
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    # Extract text
    document_text = extract_text(file_path)

    # Create vector database with HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    global vectorstore
    vectorstore = FAISS.from_texts([document_text], embeddings)

    return {"message": "File uploaded & indexed for search."}

@app.post("/gpt-only/")
async def gpt_response(question: str):
    """ GPT-only response using Gemini API. """
    if not question:
        return {"error": "Question parameter is missing"}
    
    url = "https://gemini-api-url.com"  # Replace with actual Gemini API endpoint
    headers = {"Authorization": f"Bearer {GEMINI_API_KEY}"}
    payload = {"question": question}
    response = requests.post(url, json=payload, headers=headers)
    return {"response": response.json().get("answer")}

@app.post("/enhanced-response/")
async def enhanced_response(question: str):
    """ FinBERT + RAG + GPT-4 response. """

    # Log incoming request
    logging.debug(f"Received question: {question}")

    if not question:
        return {"error": "Question is required."}

    # Retrieve relevant financial text using FAISS RAG
    retrieved_docs = vectorstore.similarity_search(question, k=3)
    retrieved_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Log retrieved documents text
    logging.debug(f"Retrieved documents text: {retrieved_text}")

    # FinBERT Sentiment & Risk Detection
    inputs = tokenizer(retrieved_text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    sentiment = torch.argmax(outputs.logits).item()
    labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
    finbert_analysis = labels[sentiment]

    # Log FinBERT analysis
    logging.debug(f"FinBERT analysis: {finbert_analysis}")

    # Use Gemini (or Hugging Face GPT) to explain the result
    url = "https://gemini-api-url.com"  # Replace with actual Gemini API endpoint
    headers = {"Authorization": f"Bearer {GEMINI_API_KEY}"}
    final_prompt = f"Analyze this financial document:\n\n{retrieved_text}\n\nFinBERT detected risk: {finbert_analysis}. Provide an in-depth analysis."
    payload = {"question": final_prompt}
    response = requests.post(url, json=payload, headers=headers)

    # Log response from Gemini API
    logging.debug(f"Gemini response: {response.json()}")

    return {
        "response": response.json().get("answer"),
        "finbert_analysis": finbert_analysis
    }
