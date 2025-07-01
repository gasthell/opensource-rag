import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import fitz  # PyMuPDF
import pandas as pd
import docx
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

SOURCE_DIRECTORY = "gradio/data"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

EMBEDDING_MODEL = None
FAISS_INDEX = None
CHUNK_STORE = []

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_xlsx(file_path):
    df = pd.read_excel(file_path, engine='openpyxl')
    return df.to_string()

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def extract_text_from_txt_or_csv(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def split_text_into_chunks(text, source_name):
    chunks = []
    paragraphs = text.split('\n\n')
    for para in paragraphs:
        if len(para) > CHUNK_SIZE:
            for i in range(0, len(para), CHUNK_SIZE - CHUNK_OVERLAP):
                chunk_text = para[i:i + CHUNK_SIZE]
                chunks.append({"source": source_name, "content": chunk_text})
        elif para.strip():
            chunks.append({"source": source_name, "content": para})
    return chunks

def setup_rag_pipeline():
    global EMBEDDING_MODEL, FAISS_INDEX, CHUNK_STORE
    
    print("Setting up RAG pipeline from scratch...")
    
    if not os.path.isdir(SOURCE_DIRECTORY):
        raise FileNotFoundError(f"Source directory '{SOURCE_DIRECTORY}' not found.")

    all_texts = []
    for filename in os.listdir(SOURCE_DIRECTORY):
        file_path = os.path.join(SOURCE_DIRECTORY, filename)
        if not os.path.isfile(file_path): continue

        print(f"Processing {filename}...")
        text = ""
        try:
            if filename.lower().endswith(".pdf"):
                text = extract_text_from_pdf(file_path)
            elif filename.lower().endswith(".xlsx"):
                text = extract_text_from_xlsx(file_path)
            elif filename.lower().endswith(".docx"):
                text = extract_text_from_docx(file_path)
            elif filename.lower().endswith((".txt", ".csv")):
                text = extract_text_from_txt_or_csv(file_path)
            else:
                print(f"Skipping unsupported file type: {filename}")
                continue
            
            CHUNK_STORE.extend(split_text_into_chunks(text, filename))
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    if not CHUNK_STORE:
        raise ValueError("No text could be extracted from the documents. The pipeline cannot be built.")

    print(f"Total chunks created: {len(CHUNK_STORE)}")

    print("Loading embedding model...")
    EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Creating embeddings for all chunks...")
    chunk_contents = [chunk['content'] for chunk in CHUNK_STORE]
    embeddings = EMBEDDING_MODEL.encode(chunk_contents, convert_to_tensor=False, show_progress_bar=True)
    
    embedding_dimension = embeddings.shape[1]
    FAISS_INDEX = faiss.IndexFlatL2(embedding_dimension)
    FAISS_INDEX.add(np.array(embeddings))
    
    print("RAG pipeline is ready.")

    return EMBEDDING_MODEL, FAISS_INDEX, CHUNK_STORE