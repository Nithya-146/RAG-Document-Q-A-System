# 📚 RAG Document Q&A System

An advanced Retrieval-Augmented Generation (RAG) dashboard that allows you to upload PDF documents, index their contents into a vector database, and ask questions using Claude. The application generates highly precise, grounded answers accompanied by page-level source citations.

---

## 🌟 Key Features

*   **🔒 Secure Configuration**: API Key is read directly from your local `.env` file or can be safely entered via the interactive sidebar.
*   **⚙️ Dashboard Controls**: Tune hyperparameters in real-time, including:
    *   **LLM Model**: Switch between Claude 3.5 Sonnet, Claude 3.5 Haiku, and Claude 3 Opus.
    *   **Generation Parameters**: Adjust temperature levels for creativity/factual precision.
    *   **Retrieval Customization**: Set text chunk size, chunk overlap, and retrieved Top-K source documents.
*   **📊 Document Intelligence Metrics**: Instantly visualizes vector storage statistics, including total pages, total text chunks created, and vector indexing time.
*   **💬 Grounded Chat Interface**: Displays chronological chat history, along with expandable sections showing the exact text chunks retrieved from the FAISS database and their page citations.

---

## 🛠️ Architecture & Tech Stack

1.  **Frontend Interface**: [Streamlit](https://streamlit.io/) (with a custom dark theme)
2.  **PDF Parser**: [PyPDF2](https://pypdf-ec.github.io/PyPDF2/) (extracts digital text layer with page metadata)
3.  **Embeddings Model**: `all-MiniLM-L6-v2` via [HuggingFace Embeddings / sentence-transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
4.  **Vector Store**: [FAISS](https://github.com/facebookresearch/faiss) (locally runs similarity searches)
5.  **Language Model**: [Anthropic Claude API](https://www.anthropic.com/api) (`claude-3-5-sonnet-latest`)

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.10+ installed on your system.

### 2. Clone the Repository
```bash
git clone https://github.com/Nithya-146/RAG-Document-Q-A-System.git
cd RAG-Document-Q-A-System
```

### 3. Install Dependencies
Install the required packages using pip:
```bash
pip install -r requirements.txt
```

### 4. Configuration
Create a `.env` file in the root directory and add your Anthropic Claude API Key:
```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```
*(Note: If you do not create a `.env` file, you can still type your key directly into the application UI at runtime).*

### 5. Run the Application
Start the Streamlit development server:
```bash
python -m streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser to start analyzing documents.