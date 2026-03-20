
import streamlit as st
import os
import time
import tempfile
import json
import pandas as pd
import numpy as np
from endee import Precision
from sentence_transformers import SentenceTransformer
from logic import (load_model, get_endee, ensure_index, chunk_text, extract_text, 
                  vision_ocr_pdf, delete_by_filename, get_indexed_files, get_llm_response)
from ui import apply_custom_styles, render_sidebar, get_base64

# --- FIX: Monkey-patch Endee Client Bug ---
import endee.index
def patched_is_hybrid(self):
    # original bug: return self.sparse_model != "None" 
    # where self.sparse_model is None (type None) if not from server
    # Correct logic should check if sparse_model is a non-empty, non-"None" string
    return bool(self.sparse_model and self.sparse_model.lower() != "none")

endee.index.Index.is_hybrid = property(patched_is_hybrid)
# ------------------------------------------

# Page config
st.set_page_config(page_title="Curator AI | Knowledge Engine", page_icon="⚡", layout="wide")
apply_custom_styles()

# Statistics Persistence
STATS_FILE = "assignment/stats.json"

def load_stats():
    DEFAULT_STATS = {"total_queries": 1482, "topics": {"Market Analysis": 45, "Revenue Growth": 32}, "query_history": [10, 15, 8, 12, 20, 18, 25, 30]}
    if not os.path.exists(STATS_FILE) or os.path.getsize(STATS_FILE) == 0:
        return DEFAULT_STATS
    try:
        with open(STATS_FILE, "r") as f: return json.load(f)
    except (json.JSONDecodeError, ValueError):
        return DEFAULT_STATS

def save_stats(stats):
    with open(STATS_FILE, "w") as f: json.dump(stats, f)

if "stats" not in st.session_state:
    st.session_state.stats = load_stats()

# Initialize session state
if "messages" not in st.session_state: st.session_state.messages = []
if "current_page" not in st.session_state: st.session_state.current_page = "Dashboard"
if "deleted_files" not in st.session_state: st.session_state.deleted_files = set()

model = load_model()
client = get_endee()

# Sidebar
render_sidebar()

# Page Content
page = st.session_state.current_page

# --- Dashboard ---
if page == "Dashboard":
    st.markdown('<div class="hero-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">AI Knowledge Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Navigate your intellectual property with absolute precision. Our neural engine synthesizes your documents into an accessible, searchable, and intelligent private database.</p>', unsafe_allow_html=True)
    
    # Feature Cards
    st.markdown("""
        <div class="card-container">
            <div class="feature-card">
                <div class="icon-box" style="background:#E0F2FE;"><i class="fas fa-brain" style="color:#0EA5E9;"></i></div>
                <h4 class="card-title">Neural Synthesis</h4>
                <p class="card-desc">Beyond simple search. Understand complex relationships across thousands of documents with recursive vector mapping.</p>
            </div>
            <div class="feature-card">
                <div class="icon-box" style="background:#F5F3FF;"><i class="fas fa-shield-alt" style="color:#8B5CF6;"></i></div>
                <h4 class="card-title">Private Edge</h4>
                <p class="card-desc">Your data never leaves your infrastructure. Enterprise-grade encryption at rest and in transit with zero-knowledge architecture.</p>
            </div>
            <div class="feature-card">
                <div class="icon-box" style="background:#E0F2FE;"><i class="fas fa-bolt" style="color:#0EA5E9;"></i></div>
                <h4 class="card-title">Real-time Index</h4>
                <p class="card-desc">Instant updates. As soon as a file is uploaded, it is vectorized and available for natural language querying within seconds.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Quote
    st.markdown("""
        <div class="quote-section">
            <p class="quote-text">"The goal is not to store information, but to generate intelligence. Your library is a sleeping giant; let us wake it."</p>
            <p class="quote-author">— CHIEF DATA CURATOR</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Chat Area
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if prompt := st.chat_input("Ask anything about your knowledge base..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        with st.spinner("🧠 Retrieving Context..."):
            try:
                kb_index = ensure_index(client)
                query_vec = model.encode([prompt])[0].tolist()
                kb_results = kb_index.query(vector=query_vec, top_k=3)
            except: kb_results = []

        contexts = [f"[Source: {m.get('meta', {}).get('source', 'Unknown')}] {m.get('meta', {}).get('text', '')}" for m in kb_results]
        context_block = "\n\n---\n\n".join(contexts) if contexts else "No relevant documents."
        chat_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-5:]])
        llm_prompt = f"Context: {context_block}\n\nHistory: {chat_history}\n\nUser: {prompt}\n\nAssistant:"
        
        with st.spinner("Reasoning..."):
            response = get_llm_response(llm_prompt)
            # Update stats
            st.session_state.stats["total_queries"] += 1
            st.session_state.stats["query_history"] = st.session_state.stats["query_history"][-19:] + [st.session_state.stats["query_history"][-1] + 1]
            save_stats(st.session_state.stats)
        
        if not response and kb_results: response = "Quota issue. Best match: " + kb_results[0].get('meta', {}).get('text', '')
        elif not response: response = "I couldn't find an answer."
        
        with st.chat_message("assistant"): st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# --- Uploads ---
elif page == "Uploads":
    st.title("📁 Upload Documents")
    st.markdown("Ingest PDFs, Markdown, or Text files to build your knowledge base.")
    
    uploaded_files = st.file_uploader("Drop your files here", type=["pdf", "md", "txt"], accept_multiple_files=True)
    
    if st.button("🚀 Ingest into Endee", disabled=not uploaded_files, use_container_width=True):
        idx = ensure_index(client)
        prog = st.progress(0, text="Processing...")
        for i, val in enumerate(uploaded_files):
            ext = os.path.splitext(val.name)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(val.read())
                path = tmp.name
            
            text = extract_text(path, val.name)
            if not text.strip() and ext == ".pdf":
                with st.status(f"🔍 OCR for {val.name}"): text = vision_ocr_pdf(path)
            
            if text.strip():
                chunks = chunk_text(text)
                vectors = model.encode(chunks)
                payloads = [{
                    "id": f"text::{val.name}::{j}",
                    "vector": v.tolist(),
                    "meta": {"text": c, "source": val.name, "type": "text"},
                    "filter": {"source": val.name}
                } for j, (c, v) in enumerate(zip(chunks, vectors))]
                idx.upsert(payloads)
                st.success(f"Ingested {val.name}")
            else: st.warning(f"Skipped {val.name}")
            os.unlink(path)
            prog.progress((i+1)/len(uploaded_files))
        st.balloons()

# --- Library ---
elif page == "Library":
    st.title("📚 Knowledge Library")
    st.markdown("Manage your indexed documents and AI resources. These files power your assistant's contextual intelligence.")
    
    files = [f for f in get_indexed_files(client) if f not in st.session_state.deleted_files]
    
    if not files:
        st.info("🌑 Memory is empty. Upload files to get started.")
    else:
        # Table Header
        h1, h2, h3, h4, h5 = st.columns([3, 1, 1, 1, 0.5])
        h1.markdown("**DOCUMENT NAME**")
        h2.markdown("**STATUS**")
        h3.markdown("**SIZE**")
        h4.markdown("**UPLOADED**")
        h5.markdown("**ACTIONS**")
        st.divider()

        for f in files:
            c1, c2, c3, c4, c5 = st.columns([3, 1, 1, 1, 0.5])
            c1.markdown(f"📄 **{f}**")
            c2.markdown('<span class="badge-indexed">• INDEXED</span>', unsafe_allow_html=True)
            c3.write("2.4 MB") # Placeholder
            c4.write("Oct 12, 2023") # Placeholder
            if c5.button("🗑️", key=f"del_{f}"):
                if delete_by_filename(client, f):
                    st.session_state.deleted_files.add(f)
                    st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Wipe Knowledge Base", use_container_width=True, type="secondary"):
            try:
                client.delete_index("knowledge_base")
                st.success("Knowledge Base Wiped. It will be recreated on next upload.")
                st.rerun()
            except Exception as e:
                st.error(f"Wipe failed: {e}")

        if st.button("🔄 Force Recreate Index (Fix Mode)", use_container_width=True):
            try:
                client.delete_index("knowledge_base")
            except:
                pass 
            
            client.create_index(name="knowledge_base", dimension=384, space_type="cosine", precision=Precision.FLOAT32, sparse_model="None")
            st.success("Index Recreated as Dense-Only. Try uploading again.")
            st.rerun()

# --- Analytics ---
elif page == "Analytics":
    st.title("📊 Knowledge Analytics")
    st.markdown("Real-time performance metrics and cognitive usage insights.")
    
    real_files = len(get_indexed_files(client))
    stats = st.session_state.stats
    
    # CSS defined in ui.py already
    st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-card">
                <p class="stat-label">Total Documents</p>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <p class="stat-value">{real_files}</p>
                    <span style="color:#10B981; font-weight:600; font-size:0.875rem;">+12% vs LW</span>
                </div>
            </div>
            <div class="stat-card">
                <p class="stat-label">Total Queries</p>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <p class="stat-value">{stats['total_queries']:,}</p>
                    <span style="color:#10B981; font-weight:600; font-size:0.875rem;">+24% vs LW</span>
                </div>
            </div>
            <div class="stat-card">
                <p class="stat-label">Most Queried Topic</p>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <p class="stat-value" style="font-size:1.5rem;">Market Analysis</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📈 Intelligent Retrieval Velocity")
    import pandas as pd
    import numpy as np
    
    chart_data = pd.DataFrame(
        stats["query_history"],
        columns=['Queries']
    )
    st.area_chart(chart_data, color="#6366F1")
    
    st.markdown("### 🔍 Topic Distribution")
    topic_data = pd.DataFrame({
        'Topic': list(stats["topics"].keys()),
        'Frequency': list(stats["topics"].values())
    })
    st.bar_chart(topic_data.set_index('Topic'), color="#3B82F6")
    
    if st.button("🔄 Refresh Analytics", use_container_width=True):
        st.rerun()

# --- Settings ---
elif page == "Settings":
    st.title("⚙️ Settings")
    st.markdown("Refine your assistant's intellect and presence.")
    
    with st.expander("👤 Profile", expanded=True):
        st.text_input("Full Name", "Sarah Mitchell")
        st.text_input("Email Address", "sarah.m@ai.curator")
        st.button("Update Profile")
        
    with st.expander("🤖 AI Configuration"):
        st.selectbox("Model Selection", ["Claude 3.5 Sonnet", "Gemini 2.0 Flash", "GPT-4o"])
        st.slider("Temperature", 0.0, 1.0, 0.7)
        st.number_input("Max Tokens", 4096)
        
    with st.expander("🔐 Security & Privacy"):
        st.text_input("API Key Management", "••••••••••••••••••••", type="password")
        st.checkbox("Keep chat history indefinitely", value=True)
        st.button("Rotate Key")


st.markdown("<br><br><br>", unsafe_allow_html=True)
