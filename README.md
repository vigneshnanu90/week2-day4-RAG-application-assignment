# 📚 Multi-PDF RAG App with Sources (`rag_app.py`)

A single-file **Multi-PDF Retrieval-Augmented Generation (RAG)** application built with **Python**, **LangChain**, **FAISS**, **OpenAI**, **Streamlit**, and an embedded **React 18 Chatbox UI**.

Users can upload an entire folder's worth of PDF documents in a single batch, index all chunks into a unified **FAISS** vector database (`st.session_state.store`), ask questions across every uploaded document, verify exact **file name and page number** sources (`Sources: file.pdf - page X`), and maintain full session state even when the browser page is refreshed (`F5`).

## ✨ Features & Assignment Requirements Covered

* **Single-File Architecture (`rag_app.py`):** All backend RAG pipelines, state persistence logic, and the embedded React 18 Chatbox interface reside in one file launched via `streamlit run rag_app.py`.
* **Batch Multi-PDF Upload:** Uses `st.file_uploader(..., type="pdf", accept_multiple_files=True)` so users can upload multiple PDFs at once rather than one at a time.
* **Unified FAISS Vector Database:** Extracts pages via `PyPDFLoader` (converting 0-indexed page metadata to human-readable 1-based page numbers), splits text with `RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)`, and embeds all chunks into a single `FAISS` vector store using `OpenAIEmbeddings("text-embedding-3-small")`.
* **Indexed Chunk Reporting:** Immediately reports all uploaded filenames and the exact total number of indexed chunks (`Uploaded: ... | Indexed N chunks.`) upon completion.
* **Source & Page Citations on Every Answer:** Retrieves context across all uploaded PDFs using Maximal Marginal Relevance (`MMR`) and displays deduplicated file and page citations underneath every answer (`Sources: policy.pdf - page 1 | handbook.pdf - page 2`) with expandable page excerpts.
* **Cross-Document Disagreement Detection ("These Two Disagree"):** When multiple uploaded PDFs contain conflicting figures, rules, or claims, the assistant explicitly contrasts both positions side by side with their respective `[File Name - page X]` citations rather than silently picking one.
* **Persistent `st.session_state` Across Browser Refreshes (`F5`):** Combines a `@st.cache_resource` server-memory vault with local disk snapshots (`.faiss_session_cache/faiss_index` and `session_state.json`) plus React `localStorage` backup so refreshing the browser page never loses the indexed FAISS store, chunk count, or chat history.
* **Environment-Based API Key Security:** Reads `OPENAI_API_KEY` automatically from environment variables or a local `.env` file via `python-dotenv`.

## 🧠 Design Decisions ("Think It Through")

| Design Question | How `rag_app.py` Solves It |
| :--- | :--- |
| **1. What happens when two uploaded PDFs disagree?** | Standard top-k similarity search often pulls redundant chunks from a single verbose PDF. `rag_app.py` uses **Maximal Marginal Relevance (`search_type="mmr"`, `k=4`, `fetch_k=16`)** to diversify retrieval across different files, paired with a conflict-aware system prompt that instructs `gpt-4o-mini` to explicitly report when two documents disagree and cite each file and page number. |
| **2. If an answer cites 5 different pages, is chunk size doing its job?** | Citing 5+ pages for a single focused question indicates either overly small chunks (fragmenting a single thought) or an excessive `k` value. `rag_app.py` uses a balanced `chunk_size=900` (`chunk_overlap=150`, `k=4`) and deduplicates `(source, page)` metadata pairs so answers cite only the 1–2 exact pages that contain the facts. |

## 🛠️ Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Chatbox Frontend** | React 18 (`react`, `react-dom`, JSX via Babel) embedded in Streamlit |
| **App Framework** | Streamlit (`st.session_state`, `@st.cache_resource`) |
| **RAG Orchestration** | LangChain (`langchain-core`, `langchain-community`, `langchain-text-splitters`) |
| **Vector Store** | FAISS (`faiss-cpu`) with local disk & session persistence |
| **Embeddings & LLM** | OpenAI `text-embedding-3-small` & `gpt-4o-mini` (`langchain-openai`) |
| **PDF Loader** | `PyPDFLoader` (`pypdf`) |

## 📁 Project Structure

```text
multi-rag/
├── rag_app.py                 # Single-file Streamlit + React 18 + LangChain + FAISS app
├── .env                       # Local environment file storing OPENAI_API_KEY
├── .faiss_session_cache/      # Auto-generated local persistence cache for browser refreshes
└── README.md                  # Project documentation
```

## 🚀 Local Setup & Execution (Windows `cmd` / macOS / Linux)

### 1. Create and Activate a Virtual Environment

**Windows (Command Prompt `cmd`):**

```cmd
mkdir multi-rag && cd multi-rag
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**

```bash
mkdir multi-rag && cd multi-rag
python3 -m venv venv && source venv/bin/activate
```

### 2. Install Dependencies

```cmd
pip install streamlit langchain langchain-openai langchain-community langchain-core langchain-text-splitters pypdf faiss-cpu python-dotenv
```

### 3. Set Your OpenAI API Key

Create a `.env` file in the project folder:

```env
OPENAI_API_KEY=your-openai-api-key-here
```

Or export it directly in your terminal:

* **Windows (`cmd`):** `set OPENAI_API_KEY=your-openai-api-key-here`
* **macOS / Linux:** `export OPENAI_API_KEY="your-openai-api-key-here"`

### 4. Run the Application

```cmd
streamlit run rag_app.py
```

## 🖥️ Example Output

```text
Uploaded: 01_People_and_Leave_Policy.pdf, 02_Hybrid_Work_and_Equipment.pdf, 03_Expenses_and_Travel_Policy.pdf
Indexed 12 chunks.

Q: How many annual leave days do full-time employees get, and what is the home-office equipment allowance?
A: Full-time employees receive 18 paid annual leave days per calendar year (prorated for new joiners and rounded up to the next half day), with up to 5 unused days carrying over until March 31. For home-office equipment, employees may claim up to $250 once every two years by submitting an itemized receipt within 30 days with manager approval.
   Sources: 01_People_and_Leave_Policy.pdf - page 1 | 02_Hybrid_Work_and_Equipment.pdf - page 2 | 03_Expenses_and_Travel_Policy.pdf - page 2
```
