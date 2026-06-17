import sys
sys.modules['spacy'] = None

import streamlit as st
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import anthropic
import os
import time
from dotenv import load_dotenv

# Load optional .env file variables
load_dotenv()

# Set page configuration with premium UI setup
st.set_page_config(
    page_title="RAG Document Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS to make all custom styled text 100% visible and visually premium
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');
    
    /* Global font override */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header gradient styling */
    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-background-fill-color: transparent;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
        line-height: 1.2;
    }
    
    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        font-size: 1.15rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
    
    /* Workflow card design */
    .step-card {
        background-color: #1e293b;
        color: #f8fafc;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        transition: transform 0.2s, border-color 0.2s;
    }
    .step-card:hover {
        transform: translateY(-2px);
        border-color: #8b5cf6;
    }
    .step-num {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 1.3rem;
        color: #a78bfa;
        margin-bottom: 4px;
    }
    .step-title {
        font-weight: 600;
        font-size: 1rem;
        color: #f1f5f9;
        margin-bottom: 8px;
    }
    .step-desc {
        font-size: 0.85rem;
        color: #94a3b8;
        line-height: 1.4;
    }

    /* Source Citation Cards - Specifying BOTH background and text explicitly to guarantee readability */
    .citation-card {
        background-color: #0f172a;
        color: #f1f5f9;
        border-left: 5px solid #3b82f6;
        border-right: 1px solid #1e293b;
        border-top: 1px solid #1e293b;
        border-bottom: 1px solid #1e293b;
        border-radius: 6px;
        padding: 16px;
        margin-top: 12px;
        margin-bottom: 12px;
        box-shadow: inset 0 2px 4px 0 rgb(0 0 0 / 0.06);
    }
    .citation-header {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        color: #38bdf8;
        font-size: 0.95rem;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
    }
    .citation-body {
        font-family: 'Inter', sans-serif;
        color: #cbd5e1;
        font-size: 0.875rem;
        line-height: 1.5;
        white-space: pre-wrap;
    }
    .citation-match-badge {
        background-color: #1e3a8a;
        color: #93c5fd;
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 500;
    }
    
    /* Metrics Container Custom Styles */
    .metric-box {
        background-color: #1e293b;
        color: #f8fafc;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE INIT -----------------
if "processed_pdf" not in st.session_state:
    st.session_state.processed_pdf = False
if "db" not in st.session_state:
    st.session_state.db = None
if "filename" not in st.session_state:
    st.session_state.filename = ""
if "total_pages" not in st.session_state:
    st.session_state.total_pages = 0
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0
if "elapsed_time" not in st.session_state:
    st.session_state.elapsed_time = 0.0
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------------- SIDEBAR CONFIG -----------------
with st.sidebar:
    st.markdown("## ⚙️ Configuration Dashboard")
    
    # 1. API Configuration
    st.markdown("### 🔑 API Access")
    env_api_key = os.getenv("ANTHROPIC_API_KEY", "")
    
    api_key_input = st.text_input(
        "Anthropic API Key",
        value=env_api_key if env_api_key else "",
        type="password",
        placeholder="sk-ant-...",
        help="Input your Anthropic Claude API Key. If set in .env, it will autofill."
    )
    
    if api_key_input:
        st.success("API Key Provided", icon="✅")
    else:
        st.warning("API Key Required to ask questions", icon="⚠️")
        
    st.markdown("---")
    
    # 2. Model configuration
    st.markdown("### 🧠 LLM Settings")
    selected_model = st.selectbox(
        "Claude Model Selection",
        options=["claude-3-5-sonnet-latest", "claude-3-5-haiku-latest", "claude-3-opus-20240229"],
        index=0,
        help="Select the Claude model for processing document answers."
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        help="Lower values are more factual, higher values are more creative."
    )
    
    st.markdown("---")
    
    # 3. RAG Splitting Settings
    st.markdown("### ✂️ Chunking & Retrieval Options")
    chunk_size = st.slider("Chunk Size (characters)", min_value=200, max_value=2000, value=800, step=50)
    chunk_overlap = st.slider("Chunk Overlap (characters)", min_value=0, max_value=500, value=150, step=25)
    top_k = st.slider("Retrieve Top-K Chunks", min_value=1, max_value=10, value=4, step=1)
    
    # Clear session state button
    st.markdown("---")
    if st.button("🔄 Clear System Cache & History", use_container_width=True):
        st.session_state.processed_pdf = False
        st.session_state.db = None
        st.session_state.filename = ""
        st.session_state.total_pages = 0
        st.session_state.total_chunks = 0
        st.session_state.elapsed_time = 0.0
        st.session_state.chat_history = []
        st.toast("System cache cleared successfully!", icon="🧹")
        st.rerun()

# ----------------- MAIN APP HEADER -----------------
st.markdown('<h1 class="hero-title">RAG Document Q&A System</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">An advanced Retrieval-Augmented Generation dashboard powered by Streamlit, FAISS, and Claude API.</p>', unsafe_allow_html=True)

# ----------------- CACHED EMBEDDINGS MODEL -----------------
@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ----------------- CORE DATA PROCESSING FUNCTIONS -----------------
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    documents = []
    total_pages = len(pdf_reader.pages)
    
    for page_idx, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text()
        if page_text and page_text.strip():
            doc = Document(
                page_content=page_text,
                metadata={
                    "source": uploaded_file.name,
                    "page": page_idx + 1,
                    "total_pages": total_pages
                }
            )
            documents.append(doc)
            
    return documents, total_pages

def build_vector_store(documents, chunk_size, chunk_overlap):
    # Split text chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = splitter.split_documents(documents)
    
    # Load embeddings and build FAISS store
    embeddings_model = get_embeddings_model()
    db = FAISS.from_documents(split_docs, embeddings_model)
    return db, len(split_docs)

# ----------------- APP BODY -----------------
# Render app introductory steps if no document is processed
if not st.session_state.processed_pdf:
    st.markdown("### How it Works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="step-card">
            <div class="step-num">01</div>
            <div class="step-title">Upload a PDF</div>
            <div class="step-desc">Provide any PDF document. PyPDF2 reads and splits it into semantic chunks preserving original page numbers.</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="step-card">
            <div class="step-num">02</div>
            <div class="step-title">Vector Indexing</div>
            <div class="step-desc">Chunks are converted to dense vector embeddings using sentence-transformers and indexed in a FAISS vector database.</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="step-card">
            <div class="step-num">03</div>
            <div class="step-title">Ask & Cite</div>
            <div class="step-desc">Claude reads retrieved context matching your question to generate grounded answers with precise page citations.</div>
        </div>
        """, unsafe_allow_html=True)

# ----------------- PDF UPLOAD -----------------
st.markdown("### 📤 Upload Document")
uploaded_file = st.file_uploader("Upload a PDF document to begin", type=["pdf"])

if uploaded_file:
    # If the file has changed or hasn't been processed yet
    if st.session_state.filename != uploaded_file.name:
        with st.status("🚀 Initializing RAG Pipeline...", expanded=True) as status:
            start_time = time.time()
            
            # Step 1: Extract Text
            status.update(label="📄 Extracting text from PDF pages...", state="running")
            documents, total_pages = extract_text_from_pdf(uploaded_file)
            
            if not documents:
                status.update(label="❌ Failed to extract text from PDF. It may be scanned or empty.", state="error")
                st.error("No text could be extracted from this PDF. Please verify it contains digital text, not just scanned image layers.")
            else:
                # Step 2: Indexing
                status.update(label="🧠 Splitting text and generating FAISS vector index...", state="running")
                db, total_chunks = build_vector_store(documents, chunk_size, chunk_overlap)
                
                # Save to session state
                st.session_state.db = db
                st.session_state.filename = uploaded_file.name
                st.session_state.total_pages = total_pages
                st.session_state.total_chunks = total_chunks
                st.session_state.processed_pdf = True
                st.session_state.elapsed_time = time.time() - start_time
                st.session_state.chat_history = [] # clear chat for new file
                
                status.update(label="✅ Vector database build complete!", state="complete")
                st.toast(f"Successfully processed {uploaded_file.name}!", icon="🎯")

# ----------------- LOGS & STATS DASHBOARD -----------------
if st.session_state.processed_pdf:
    st.markdown("### 📊 Document Intelligence Metrics")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Document Name</div>
            <div class="metric-value" style="font-size: 1.15rem; padding: 10px 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                {st.session_state.filename}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Total Pages</div>
            <div class="metric-value">{st.session_state.total_pages}</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Vector Chunks</div>
            <div class="metric-value">{st.session_state.total_chunks}</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Indexing Time</div>
            <div class="metric-value">{st.session_state.elapsed_time:.2f}s</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")

    # ----------------- QA LOGIC & CHAT INTERFACE -----------------
    st.markdown("### 💬 Grounded Q&A Interface")
    
    # Check for API Key
    if not api_key_input:
        st.info("💡 Please provide your Anthropic API Key in the sidebar to activate the Q&A engine.")
    else:
        # Show Chat History
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat["question"])
            with st.chat_message("assistant"):
                st.write(chat["answer"])
                with st.expander("🔍 Citations & Grounded Context"):
                    for idx, doc in enumerate(chat["sources"]):
                        page = doc.metadata.get("page", "Unknown")
                        text = doc.page_content
                        st.markdown(f"""
                        <div class="citation-card">
                            <div class="citation-header">
                                <span>📍 Chunk {idx + 1} (Page {page})</span>
                                <span class="citation-match-badge">Retrieved Source</span>
                            </div>
                            <div class="citation-body">"{text}"</div>
                        </div>
                        """, unsafe_allow_html=True)

        # Handle New Query Input
        user_query = st.chat_input("Ask a question about the uploaded document...")
        
        if user_query:
            # 1. Show user question
            with st.chat_message("user"):
                st.write(user_query)
            
            # 2. Retrieve relevant chunks
            with st.spinner("Searching document database..."):
                retrieved_docs = st.session_state.db.similarity_search(user_query, k=top_k)
                
            # 3. Call Claude API
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner(f"Generating grounded answer using {selected_model}..."):
                    try:
                        # Construct context string
                        context_str = ""
                        for idx, doc in enumerate(retrieved_docs):
                            page_num = doc.metadata.get("page", "Unknown")
                            context_str += f"[Chunk {idx+1}] (Page {page_num}):\n{doc.page_content}\n\n"
                        
                        # Set prompt instructions
                        system_prompt = (
                            "You are a precise, helpful RAG (Retrieval-Augmented Generation) assistant.\n"
                            "Your task is to answer the query based strictly on the provided context chunks extracted from a PDF document.\n\n"
                            "Instructions:\n"
                            "1. Base your answer ONLY on the provided context chunks. Do not assume or extrapolate from external knowledge unless it is directly referenced in the context.\n"
                            "2. If the answer cannot be found in the provided context, state: 'I cannot find the answer to this question in the uploaded document.' Do not try to make up an answer.\n"
                            "3. Be objective, concise, and structured in your explanation.\n"
                            "4. When writing your answer, cite the page numbers of the facts you mention using square brackets, e.g., [Page X] or [Pages X, Y], based on the page metadata in the context."
                        )
                        
                        user_content = f"Context Chunks:\n{context_str}\n\nQuery: {user_query}"
                        
                        # Initialize Anthropic Client
                        client = anthropic.Anthropic(api_key=api_key_input)
                        
                        response = client.messages.create(
                            model=selected_model,
                            max_tokens=1500,
                            temperature=temperature,
                            system=system_prompt,
                            messages=[
                                {"role": "user", "content": user_content}
                            ]
                        )
                        
                        answer_text = response.content[0].text
                        message_placeholder.markdown(answer_text)
                        
                        # Render citation sources
                        with st.expander("🔍 Citations & Grounded Context", expanded=True):
                            for idx, doc in enumerate(retrieved_docs):
                                page = doc.metadata.get("page", "Unknown")
                                text = doc.page_content
                                st.markdown(f"""
                                <div class="citation-card">
                                    <div class="citation-header">
                                        <span>📍 Chunk {idx + 1} (Page {page})</span>
                                        <span class="citation-match-badge">Retrieved Source</span>
                                    </div>
                                    <div class="citation-body">"{text}"</div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Save to history
                        st.session_state.chat_history.append({
                            "question": user_query,
                            "answer": answer_text,
                            "sources": retrieved_docs
                        })
                        
                    except Exception as e:
                        st.error(f"Error calling Anthropic API: {str(e)}")
