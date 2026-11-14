import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load OPENAI_API_KEY from .env or system environment variables
load_dotenv()

PERSIST_DIR = Path(".faiss_session_cache")
FAISS_INDEX_DIR = PERSIST_DIR / "faiss_index"
STATE_JSON_PATH = PERSIST_DIR / "session_state.json"

SYSTEM_PROMPT = """You are an accurate, source-grounded Multi-PDF RAG Assistant.
Answer the user's question using ONLY the retrieved document chunks below.

Rules:
1. Grounding: Use only facts directly stated in the provided chunks.
2. Handling Disagreements Across PDFs ("These Two Disagree"):
   - If two or more uploaded PDFs (or different pages) contain conflicting numbers, rules, dates, or claims, DO NOT quietly pick one.
   - Explicitly state that the documents disagree, and explain what each document states along with its file name and page number (e.g., `fileA.pdf - page 2` vs. `fileB.pdf - page 1`).
3. Missing Information:
   - If the uploaded PDFs do not contain the answer, say clearly that the answer is not found in the uploaded documents.
4. Keep your answer concise, direct, and easy to verify.

Retrieved Chunks:
{context}
"""


@st.cache_resource
def get_server_session_vault() -> Dict[str, Any]:
    """Server-side memory vault that survives browser page refreshes (F5)."""
    return {
        "store": None,
        "uploaded_names": [],
        "chunk_count": 0,
        "file_signature": None,
        "messages": [],
    }


def save_state_to_disk_and_vault() -> None:
    """Persist st.session_state (including FAISS store) so page refreshes never lose state."""
    vault = get_server_session_vault()
    vault["store"] = st.session_state.get("store")
    vault["uploaded_names"] = st.session_state.get("uploaded_names", [])
    vault["chunk_count"] = st.session_state.get("chunk_count", 0)
    vault["file_signature"] = st.session_state.get("file_signature")
    vault["messages"] = st.session_state.get("messages", [])

    try:
        PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        if st.session_state.get("store") is not None:
            st.session_state.store.save_local(str(FAISS_INDEX_DIR))
        state_payload = {
            "uploaded_names": vault["uploaded_names"],
            "chunk_count": vault["chunk_count"],
            "file_signature": vault["file_signature"],
            "messages": vault["messages"],
        }
        STATE_JSON_PATH.write_text(
            json.dumps(state_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def restore_session_state() -> None:
    """Hydrate st.session_state from server vault or local disk after a browser page refresh."""
    if "initialized" in st.session_state:
        return

    vault = get_server_session_vault()

    # If server vault is empty (e.g. process restarted), restore from local disk cache
    if (
        vault["store"] is None
        and STATE_JSON_PATH.is_file()
        and FAISS_INDEX_DIR.is_dir()
        and os.getenv("OPENAI_API_KEY")
    ):
        try:
            saved_data = json.loads(STATE_JSON_PATH.read_text(encoding="utf-8"))
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            vault["store"] = FAISS.load_local(
                str(FAISS_INDEX_DIR),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            vault["uploaded_names"] = saved_data.get("uploaded_names", [])
            vault["chunk_count"] = saved_data.get("chunk_count", 0)
            vault["file_signature"] = saved_data.get("file_signature")
            vault["messages"] = saved_data.get("messages", [])
        except Exception:
            pass

    st.session_state.store = vault["store"]
    st.session_state.uploaded_names = list(vault["uploaded_names"])
    st.session_state.chunk_count = int(vault["chunk_count"])
    st.session_state.file_signature = vault["file_signature"]
    st.session_state.messages = list(vault["messages"])
    st.session_state.initialized = True


def clear_persisted_session() -> None:
    """Clear FAISS store and chat history from st.session_state, server vault, and disk."""
    vault = get_server_session_vault()
    vault["store"] = None
    vault["uploaded_names"] = []
    vault["chunk_count"] = 0
    vault["file_signature"] = None
    vault["messages"] = []

    st.session_state.store = None
    st.session_state.uploaded_names = []
    st.session_state.chunk_count = 0
    st.session_state.file_signature = None
    st.session_state.messages = []

    if PERSIST_DIR.exists():
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)


def compute_upload_signature(files: List[Any], chunk_size: int, chunk_overlap: int) -> str:
    """Compute a deterministic hash for the uploaded PDF batch and splitter settings."""
    hasher = hashlib.sha256()
    hasher.update(f"{chunk_size}:{chunk_overlap}".encode("utf-8"))
    for f in sorted(files, key=lambda item: item.name):
        hasher.update(f.name.encode("utf-8"))
        hasher.update(str(len(f.getvalue())).encode("utf-8"))
    return hasher.hexdigest()


def index_uploaded_pdfs(
    files: List[Any],
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> Tuple[FAISS, int, List[str]]:
    """Save uploaded PDFs temporarily, load with PyPDFLoader, split, and index into one FAISS store."""
    all_pages: List[Document] = []
    uploaded_names: List[str] = []

    for f in files:
        uploaded_names.append(f.name)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.getvalue())
            tmp_path = tmp.name

        try:
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            for doc in pages:
                # PyPDFLoader uses 0-based page numbers; convert to 1-based page numbers
                raw_page = doc.metadata.get("page", 0)
                page_num = int(raw_page) + 1 if isinstance(raw_page, int) else 1
                doc.metadata["source"] = f.name
                doc.metadata["page"] = page_num
            all_pages.extend(pages)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # Chunk size 900 with 150 overlap keeps complete policy/scientific paragraphs together
    # so a single answer cites the exact 1-2 relevant pages rather than fragmenting across 5+ pages.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_pages)
    if not chunks:
        raise ValueError("No extractable text found in the uploaded PDF files.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    store = FAISS.from_documents(chunks, embeddings)
    return store, len(chunks), uploaded_names


def answer_with_sources(
    store: FAISS,
    question: str,
    top_k: int = 4,
) -> Tuple[str, List[Dict[str, Any]], str]:
    """Retrieve matching chunks across all uploaded PDFs, answer from them, and list file + page sources."""
    # Maximal Marginal Relevance (MMR) ensures chunks are pulled across multiple PDFs
    # so cross-document agreements or disagreements are surfaced fairly.
    retriever = store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k, "fetch_k": max(top_k * 4, 16)},
    )
    docs = retriever.invoke(question)

    context_blocks: List[str] = []
    for idx, d in enumerate(docs, start=1):
        src_name = str(d.metadata.get("source", "document.pdf"))
        page_num = d.metadata.get("page", 1)
        context_blocks.append(
            f"[Chunk {idx} | {src_name} - page {page_num}]\n{d.page_content.strip()}"
        )
    context_text = "\n\n".join(context_blocks)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    response = (prompt | llm).invoke({"context": context_text, "question": question})
    answer_text = (
        response.content
        if isinstance(response.content, str)
        else str(response.content)
    )

    # Deduplicate (source, page) pairs in retrieval order
    seen = set()
    unique_sources: List[Dict[str, Any]] = []
    for d in docs:
        src_name = str(d.metadata.get("source", "document.pdf"))
        page_num = d.metadata.get("page", 1)
        key = (src_name, page_num)
        if key not in seen:
            seen.add(key)
            unique_sources.append(
                {
                    "source": src_name,
                    "page": page_num,
                    "snippet": d.page_content.strip()[:260],
                }
            )

    sources_line = " | ".join(
        f"{item['source']} - page {item['page']}" for item in unique_sources
    )
    return answer_text, unique_sources, sources_line


def render_react_chatbox(
    uploaded_names: List[str],
    chunk_count: int,
    messages: List[Dict[str, Any]],
) -> None:
    """Render an interactive React 18 Chatbox component displaying indexed status, Q&A, and sources."""
    payload_json = json.dumps(
        {
            "uploaded_names": uploaded_names,
            "chunk_count": chunk_count,
            "messages": messages,
        },
        ensure_ascii=False,
    )

    react_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <script src="https://unpkg.com/react@18/umd/react.production.min.js" crossorigin></script>
  <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js" crossorigin></script>
  <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: transparent;
      color: #0f172a;
    }}
    .react-shell {{
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 8px 24px -6px rgba(15, 23, 42, 0.08);
      display: flex;
      flex-direction: column;
      height: 540px;
    }}
    .react-header {{
      background: linear-gradient(135deg, #0f172a 0%, #1e293b 65%, #1e3a8a 100%);
      color: #f8fafc;
      padding: 16px 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
      border-bottom: 1px solid #334155;
    }}
    .react-title-wrap {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .react-logo {{
      width: 38px;
      height: 38px;
      border-radius: 10px;
      background: rgba(56, 189, 248, 0.18);
      border: 1px solid rgba(56, 189, 248, 0.4);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 18px;
      font-weight: 800;
      color: #38bdf8;
    }}
    .react-title {{
      font-size: 16px;
      font-weight: 700;
      color: #ffffff;
    }}
    .react-subtitle {{
      font-size: 12px;
      color: #cbd5e1;
      margin-top: 2px;
    }}
    .badge-row {{
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }}
    .status-pill {{
      font-size: 11.5px;
      font-weight: 600;
      padding: 5px 11px;
      border-radius: 999px;
      background: rgba(16, 185, 129, 0.2);
      border: 1px solid rgba(52, 211, 153, 0.45);
      color: #6ee7b7;
    }}
    .status-pill.waiting {{
      background: rgba(245, 158, 11, 0.2);
      border-color: rgba(251, 191, 36, 0.45);
      color: #fde68a;
    }}
    .index-banner {{
      background: #f8fafc;
      border-bottom: 1px solid #e2e8f0;
      padding: 10px 20px;
      font-size: 12.5px;
      color: #334155;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .index-files {{
      font-family: 'SFMono-Regular', Consolas, monospace;
      color: #0f172a;
      font-weight: 600;
    }}
    .chat-feed {{
      flex: 1;
      overflow-y: auto;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      background: #f8fafc;
    }}
    .empty-state {{
      margin: auto;
      text-align: center;
      max-width: 460px;
      padding: 28px;
      background: #ffffff;
      border: 1px dashed #cbd5e1;
      border-radius: 14px;
      color: #475569;
    }}
    .empty-state h3 {{
      font-size: 15px;
      color: #0f172a;
      margin-bottom: 6px;
    }}
    .empty-state p {{
      font-size: 13px;
      line-height: 1.5;
    }}
    .msg-row {{
      display: flex;
      flex-direction: column;
      max-width: 88%;
    }}
    .msg-row.user {{
      align-self: flex-end;
      align-items: flex-end;
    }}
    .msg-row.assistant {{
      align-self: flex-start;
      align-items: flex-start;
    }}
    .msg-label {{
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: #64748b;
      margin-bottom: 4px;
      padding: 0 4px;
    }}
    .msg-bubble {{
      padding: 13px 16px;
      border-radius: 14px;
      font-size: 14px;
      line-height: 1.55;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .msg-row.user .msg-bubble {{
      background: #1d4ed8;
      color: #ffffff;
      border-bottom-right-radius: 4px;
    }}
    .msg-row.assistant .msg-bubble {{
      background: #ffffff;
      color: #0f172a;
      border: 1px solid #e2e8f0;
      border-bottom-left-radius: 4px;
      box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }}
    .sources-box {{
      margin-top: 8px;
      background: #eff6ff;
      border: 1px solid #bfdbfe;
      border-radius: 10px;
      padding: 9px 12px;
      width: 100%;
    }}
    .sources-line {{
      font-family: 'SFMono-Regular', Consolas, monospace;
      font-size: 12px;
      font-weight: 600;
      color: #1e40af;
    }}
    .toggle-btn {{
      margin-top: 6px;
      background: none;
      border: none;
      color: #2563eb;
      font-size: 11.5px;
      font-weight: 600;
      cursor: pointer;
      padding: 0;
      text-decoration: underline;
    }}
    .snippets-list {{
      margin-top: 8px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      border-top: 1px solid #dbeafe;
      padding-top: 8px;
    }}
    .snippet-item {{
      font-size: 12px;
      color: #334155;
      background: #ffffff;
      border: 1px solid #dbeafe;
      border-radius: 6px;
      padding: 6px 9px;
    }}
    .snippet-head {{
      font-weight: 700;
      color: #1e3a8a;
      margin-bottom: 2px;
    }}
  </style>
</head>
<body>
  <div id="root"></div>
  <script type="text/babel">
    const initialData = {payload_json};

    function MessageCard({{ msg }}) {{
      const [open, setOpen] = React.useState(false);
      const isUser = msg.role === "user";
      const sources = msg.sources || [];
      const sourcesLine = msg.sources_line || "";

      return (
        <div className={{"msg-row " + (isUser ? "user" : "assistant")}}>
          <div className="msg-label">{{isUser ? "Q · You" : "A · Multi-PDF RAG Assistant"}}</div>
          <div className="msg-bubble">{{msg.content}}</div>
          {{!isUser && sourcesLine && (
            <div className="sources-box">
              <div className="sources-line">Sources: {{sourcesLine}}</div>
              {{sources.length > 0 && (
                <div>
                  <button className="toggle-btn" onClick={{() => setOpen(!open)}}>
                    {{open ? "Hide page excerpts ▲" : `Inspect ${{sources.length}} retrieved page(s) ▼`}}
                  </button>
                  {{open && (
                    <div className="snippets-list">
                      {{sources.map((s, idx) => (
                        <div key={{idx}} className="snippet-item">
                          <div className="snippet-head">{{s.source}} - page {{s.page}}</div>
                          <div>{{s.snippet}}...</div>
                        </div>
                      ))}}
                    </div>
                  )}}
                </div>
              )}}
            </div>
          )}}
        </div>
      );
    }}

    function ReactChatbox() {{
      const feedRef = React.useRef(null);
      const {{ uploaded_names, chunk_count, messages }} = initialData;

      React.useEffect(() => {{
        try {{
          localStorage.setItem("multi_pdf_rag_backup", JSON.stringify(initialData));
        }} catch (e) {{}}
        if (feedRef.current) {{
          feedRef.current.scrollTop = feedRef.current.scrollHeight;
        }}
      }}, [messages]);

      const isIndexed = chunk_count > 0 && uploaded_names.length > 0;

      return (
        <div className="react-shell">
          <div className="react-header">
            <div className="react-title-wrap">
              <div className="react-logo">⚛️</div>
              <div>
                <div className="react-title">Multi-PDF RAG Chatbox with Sources</div>
                <div className="react-subtitle">React 18 UI • LangChain + FAISS Vector Store • Session-Persisted</div>
              </div>
            </div>
            <div className="badge-row">
              <span className={{"status-pill " + (isIndexed ? "" : "waiting")}}>
                {{isIndexed ? `● FAISS Active: Indexed ${{chunk_count}} chunks` : "○ Awaiting PDF Upload"}}
              </span>
            </div>
          </div>

          {{isIndexed && (
            <div className="index-banner">
              <div>
                Uploaded: <span className="index-files">{{uploaded_names.join(", ")}}</span>
              </div>
              <div>Indexed <strong>{{chunk_count}}</strong> chunks (st.session_state persisted)</div>
            </div>
          )}}

          <div className="chat-feed" ref={{feedRef}}>
            {{messages.length === 0 ? (
              <div className="empty-state">
                <h3>{{isIndexed ? "Ready to Answer Across All Uploaded PDFs" : "Upload Your PDFs to Start"}}</h3>
                <p>
                  {{isIndexed
                    ? "Type a question in the chat bar below. Every answer will cite the exact file name and page number from the FAISS chunk metadata."
                    : "Select multiple PDF files in the uploader above. They will be chunked, embedded, and stored in a single FAISS vector database that persists across page refreshes."}}
                </p>
              </div>
            ) : (
              messages.map((m, i) => <MessageCard key={{i}} msg={{m}} />)
            )}}
          </div>
        </div>
      );
    }}

    const root = ReactDOM.createRoot(document.getElementById("root"));
    root.render(<ReactChatbox />);
  </script>
</body>
</html>
"""
    components.html(react_html, height=555, scrolling=False)


def main() -> None:
    st.set_page_config(
        page_title="Multi-PDF RAG App with Sources",
        page_icon="📚",
        layout="wide",
    )

    # 1. Restore session_state automatically on browser refresh (F5)
    restore_session_state()

    api_key = os.getenv("OPENAI_API_KEY")

    # Sidebar: Chunk settings & session persistence controls
    with st.sidebar:
        st.header("⚙️ FAISS & Session Controls")
        if api_key:
            st.success("`OPENAI_API_KEY` loaded from environment")
        else:
            st.error("`OPENAI_API_KEY` not found. Set it in your environment or `.env` file.")

        chunk_size = st.slider(
            "Chunk Size (characters)",
            min_value=400,
            max_value=1600,
            value=900,
            step=100,
            help="Tuned so each chunk holds a complete section without citing 5+ fragmented pages.",
        )
        chunk_overlap = st.slider(
            "Chunk Overlap (characters)",
            min_value=50,
            max_value=300,
            value=150,
            step=25,
        )
        top_k = st.slider(
            "Retrieved Chunks (k)",
            min_value=2,
            max_value=6,
            value=4,
        )

        if st.session_state.store is not None:
            st.divider()
            st.markdown(
                f"**Persisted Store Status:**\n"
                f"- **Files:** `{', '.join(st.session_state.uploaded_names)}`\n"
                f"- **Chunks Indexed:** `{st.session_state.chunk_count}`"
            )
            if st.button("🗑️ Clear Session & FAISS Store", use_container_width=True):
                clear_persisted_session()
                st.rerun()

    # Step 2: Accept multiple PDFs in one upload and index into a single FAISS store in st.session_state
    files = st.file_uploader(
        "Upload PDFs (Select multiple PDF files at once)",
        type="pdf",
        accept_multiple_files=True,
    )

    if files and api_key:
        current_sig = compute_upload_signature(files, chunk_size, chunk_overlap)
        if st.session_state.store is None or st.session_state.file_signature != current_sig:
            with st.spinner("Chunking PDFs and building FAISS vector database..."):
                try:
                    store, chunk_count, uploaded_names = index_uploaded_pdfs(
                        files=files,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                    st.session_state.store = store
                    st.session_state.chunk_count = chunk_count
                    st.session_state.uploaded_names = uploaded_names
                    st.session_state.file_signature = current_sig
                    save_state_to_disk_and_vault()
                except Exception as exc:
                    st.error(f"Error indexing PDFs: {exc}")

    # Report how many chunks were indexed (persists even after page refresh)
    if st.session_state.store is not None:
        st.success(
            f"Uploaded: {', '.join(st.session_state.uploaded_names)}  |  "
            f"Indexed {st.session_state.chunk_count} chunks."
        )

    # Render the React 18 Chatbox UI
    render_react_chatbox(
        uploaded_names=st.session_state.uploaded_names,
        chunk_count=st.session_state.chunk_count,
        messages=st.session_state.messages,
    )

    # Step 3: Answer questions using retrieval across every uploaded document + show sources
    question = st.chat_input("Ask a question across your uploaded PDFs...")
    if question:
        if st.session_state.store is None:
            st.warning("Please upload one or more PDF files above first.")
        else:
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question,
                    "sources": [],
                    "sources_line": "",
                }
            )
            with st.spinner("Retrieving relevant chunks from FAISS and generating answer..."):
                try:
                    answer_text, unique_sources, sources_line = answer_with_sources(
                        store=st.session_state.store,
                        question=question,
                        top_k=top_k,
                    )
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer_text,
                            "sources": unique_sources,
                            "sources_line": sources_line,
                        }
                    )
                    save_state_to_disk_and_vault()
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to generate answer: {exc}")


if __name__ == "__main__":
    main()