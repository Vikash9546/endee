
import os
import time
import tempfile
import fitz  # PyMuPDF
import streamlit as st
from sentence_transformers import SentenceTransformer
from endee import Endee, Precision
from dotenv import load_dotenv
import PIL.Image
import io
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

INDEX_NAME = "knowledge_base"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
DIMENSION = 384

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def get_endee():
    client = Endee()
    remote_url = os.environ.get("NDD_URL")
    if remote_url:
        client.set_base_url(remote_url)
    return client

def ensure_index(client):
    try:
        return client.get_index(name=INDEX_NAME)
    except Exception as e:
        raw_error = str(e).lower()
        if "not found" in raw_error or "404" in raw_error:
            client.create_index(name=INDEX_NAME, dimension=DIMENSION, space_type="cosine", precision=Precision.FLOAT32, sparse_model="None")
            return client.get_index(name=INDEX_NAME)
        st.error(f"Endee Connection Error: {e}")
        st.stop()

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return chunks

def extract_text(filepath, filename):
    if filename.lower().endswith(".pdf"):
        doc = fitz.open(filepath)
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text
    else:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

def vision_ocr_pdf(filepath):
    """Uses Gemini Vision to read handwritten notes with model fallback for robustness."""
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        return ""
    
    from google import genai
    gen_client = genai.Client()
    doc = fitz.open(filepath)
    
    def process_page(page_num):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        img_data = pix.tobytes("png")
        img = PIL.Image.open(io.BytesIO(img_data))
        
        models_to_try = ["gemini-3-flash-preview", "gemini-2.0-flash"]
        for model_name in models_to_try:
            try:
                prompt = "Extract all text from this handwritten note. Return ONLY raw text."
                response = gen_client.models.generate_content(model=model_name, contents=[prompt, img])
                if response.text:
                    return response.text
            except:
                continue
        return ""

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_page, range(len(doc))))
            
    doc.close()
    return "\n\n".join([r for r in results if r])

def delete_by_filename(client, filename):
    try:
        idx = ensure_index(client)
        if idx:
            idx.delete_with_filter([{"source": {"$eq": filename}}])
            return True
    except: pass
    return False

def get_indexed_files(client):
    try:
        idx = ensure_index(client)
        if idx:
            # Query with a dummy vector to find unique sources
            results = idx.query(vector=[0.0]*DIMENSION, top_k=100)
            sources = set(r["meta"]["source"] for r in results if "meta" in r and "source" in r["meta"])
            return sorted(list(sources))
    except: pass
    return []

@st.cache_data(show_spinner=False, ttl=3600)
def get_llm_response(prompt_text):
    current_key = os.environ.get("GEMINI_API_KEY")
    if not current_key: return None
    from google import genai
    gen_client = genai.Client(api_key=current_key)
    models_to_try = ["gemini-3-flash-preview", "gemini-flash-latest", "gemini-2.0-flash", "gemini-1.5-flash"]
    for model_name in models_to_try:
        try:
            resp = gen_client.models.generate_content(model=model_name, contents=prompt_text)
            if resp.text: return resp.text
        except Exception as e:
            if "429" in str(e):
                time.sleep(2)
            continue
    return None
