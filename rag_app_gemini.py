import os
import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables silently in the backend
load_dotenv()

POLICY_FILES: List[str] = [
    "01_People_and_Leave_Policy.pdf",
    "02_Hybrid_Work_and_Equipment.pdf",
    "03_Expenses_and_Travel_Policy.pdf",
]

# Fallback policy corpus used to auto-provision the 3 official PDFs in ./policies/
# if the app is launched in a folder where the PDFs have not yet been copied.
FALLBACK_POLICY_PAGES: Dict[str, List[str]] = {
    "01_People_and_Leave_Policy.pdf": [
        (
            "NORTHSTAR LABS | EFFECTIVE 01 JAN 2026\n"
            "NORTHSTAR LABS / PEOPLE OPERATIONS\n"
            "People and Leave Policy\n\n"
            "Purpose and scope\n"
            "This policy applies to all full-time employees of Northstar Labs, effective January 1, 2026.\n\n"
            "Annual leave\n"
            "Full-time employees receive 18 paid annual leave days per calendar year. New joiners receive leave "
            "prorated from their start date, rounded up to the next half day. Employees request leave through the "
            "People portal at least five working days in advance where practical; their manager approves it.\n\n"
            "Carryover\n"
            "Employees may carry up to five unused annual leave days into the next calendar year. Carried days "
            "expire on March 31. Unused days above the carryover limit expire unless local law requires otherwise."
        ),
        (
            "NORTHSTAR LABS | EFFECTIVE 01 JAN 2026\n"
            "NORTHSTAR LABS / PEOPLE OPERATIONS\n"
            "People and Leave Policy\n\n"
            "Sick leave\n"
            "Employees receive 10 paid sick days per calendar year. Notify your manager as soon as practical. For an "
            "absence longer than three consecutive working days, People Operations may request a medical "
            "certificate, subject to local law.\n\n"
            "Parental leave\n"
            "Eligible primary caregivers receive 16 weeks of paid parental leave; eligible secondary caregivers receive "
            "6 weeks. Contact People Operations at least 30 days before the anticipated leave when possible.\n\n"
            "Questions and exceptions\n"
            "Contact People Operations through the People portal. Applicable law and a signed employment "
            "agreement take precedence over this policy."
        ),
    ],
    "02_Hybrid_Work_and_Equipment.pdf": [
        (
            "NORTHSTAR LABS | EFFECTIVE 01 JAN 2026\n"
            "NORTHSTAR LABS / PEOPLE OPERATIONS\n"
            "Hybrid Work and Equipment Policy\n\n"
            "Working arrangement\n"
            "Northstar Labs uses a hybrid schedule. Employees normally work from the office on Tuesday and "
            "Wednesday and may work remotely on other working days, subject to role requirements and manager "
            "approval. Teams may agree on additional in-office days.\n\n"
            "Core collaboration hours\n"
            "Core collaboration hours are 11:00 a.m. to 3:00 p.m. local time, Monday through Friday. Outside these "
            "hours employees may arrange their schedule with their manager while meeting their contracted hours.\n\n"
            "Remote work location\n"
            "Remote work is permitted from the employee's approved country of employment. Working from another "
            "country requires written People Operations approval before travel."
        ),
        (
            "NORTHSTAR LABS | EFFECTIVE 01 JAN 2026\n"
            "NORTHSTAR LABS / PEOPLE OPERATIONS\n"
            "Hybrid Work and Equipment Policy\n\n"
            "Equipment\n"
            "The company provides a laptop and standard accessories. Employees may claim up to $250 once every "
            "two years for approved home-office equipment. Submit an itemized receipt within 30 days through the "
            "expense system; the manager must approve the purchase.\n\n"
            "Security\n"
            "Use company-managed devices, enable multi-factor authentication, and lock your screen when away. "
            "Report a lost device or suspected security incident to IT immediately through the service desk.\n\n"
            "Support\n"
            "For equipment or access problems, contact the IT service desk. Exceptions to the hybrid schedule require "
            "manager and People Operations approval."
        ),
    ],
    "03_Expenses_and_Travel_Policy.pdf": [
        (
            "NORTHSTAR LABS | EFFECTIVE 01 JAN 2026\n"
            "NORTHSTAR LABS / PEOPLE OPERATIONS\n"
            "Expenses and Travel Policy\n\n"
            "Business expenses\n"
            "Northstar Labs reimburses reasonable, necessary business expenses with manager approval. Submit an "
            "itemized receipt and business purpose in the expense system within 30 days of purchase.\n\n"
            "Travel approval\n"
            "Get manager approval before booking business travel. Flights should be economy class. Choose "
            "reasonably priced accommodation near the work location and use the company booking tool when "
            "available.\n\n"
            "Meals\n"
            "For approved overnight business travel, employees may claim actual meal costs up to $60 per day, "
            "including tax and tip. Alcohol is not reimbursable. Itemized receipts are required."
        ),
        (
            "NORTHSTAR LABS | EFFECTIVE 01 JAN 2026\n"
            "NORTHSTAR LABS / PEOPLE OPERATIONS\n"
            "Expenses and Travel Policy\n\n"
            "Home-office purchases\n"
            "The separate Hybrid Work and Equipment Policy sets the home-office equipment allowance: up to $250 "
            "once every two years for approved equipment. Submit an itemized receipt within 30 days and obtain "
            "manager approval.\n\n"
            "Non-reimbursable expenses\n"
            "Personal purchases, commuting to the usual office, fines, and expenses without a business purpose are "
            "not reimbursable.\n\n"
            "Help\n"
            "Ask Finance through the expense system before making an unusual purchase. Applicable law and a "
            "signed agreement take precedence over this policy."
        ),
    ],
}

SYSTEM_PROMPT = """You are Maya, a sharp, warm, and deeply knowledgeable People & Workplace Partner at Northstar Labs.
You are chatting directly with a fellow Northstar Labs colleague.

How you communicate:
1. Speak Like a Real Human Colleague, Never Like a Bot:
   - Use natural, conversational, empathetic language with contractions ("you'll", "we've", "it's", "I'd").
   - Never sound stiff, clinical, or scripted. Avoid robotic disclaimers like "The current Northstar Labs policies do not cover..." or "According to the provided document...".
   - You simply *know* how things work at Northstar Labs as an experienced insider.
2. Strictly No File Names, Page Numbers, or Document References:
   - NEVER mention PDF filenames, page numbers, chunks, or citations (e.g., never write `[File: ...]` or `(Page 1)` or "in our policy PDF").
   - Never refer to anything as "fictional", "sample", or a "demo".
3. Answering Workplace Questions Accurately:
   - Ground all company facts (leave days, carryover deadlines, hybrid days, core hours, $250 equipment budget, $60 meal limits, receipt timelines, approvals) strictly in the internal knowledge below.
   - Weave practical next steps naturally into your reply (e.g., "Just pop the request into the People portal at least five working days ahead so your manager can sign off").
4. Handling Casual Banter, Small Talk & Off-Topic Questions Like a Human:
   - If a colleague greets you, vents about a busy week, or asks a casual off-topic question (like yesterday's cricket score, weather, coffee, or weekend plans), respond with genuine warmth, personality, and light humor first—just like a friendly coworker at the office!
   - If they ask for live outside info you don't have (like live sports scores), playfully admit you missed the live scoreboard while holding down the fort at Northstar, and naturally keep the conversation flowing.

Internal Northstar Labs Knowledge (for your memory—never cite or quote mechanically):
{context}
"""


def sanitize_policy_text(text: str) -> str:
    """Remove any sample/demo/assignment boilerplate from extracted PDF text."""
    cleaned = text
    patterns = [
        r"\|\s*FICTIONAL SAMPLE\s*",
        r"It is a\s+sample for a demo and does not describe a real employer\.",
        r"This fictional policy applies",
        r"fictional sample policy",
        r"fictional sample",
    ]
    replacements = [
        "",
        "",
        "This policy applies",
        "company policy",
        "company policy",
    ]
    for pat, rep in zip(patterns, replacements):
        cleaned = re.sub(pat, rep, cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def strip_any_accidental_citations(text: str) -> str:
    """Guarantee that no bracketed file or page references ever appear in the chat output."""
    cleaned = re.sub(r"\s*\[[^\]]*(?:\.pdf|Page\s*\d+)[^\]]*\]", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\([^\)]*(?:\.pdf|Page\s*\d+)[^\)]*\)", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _write_minimal_pdf(file_path: Path, pages_text: List[str]) -> None:
    """Write a valid multi-page PDF without external binary dependencies."""
    objects: List[bytes] = []

    def add_obj(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    font_obj_id = add_obj(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    )
    page_obj_ids: List[int] = []
    content_obj_ids: List[int] = []

    for page_str in pages_text:
        lines = page_str.splitlines()
        stream_lines = ["BT", "/F1 11 Tf", "50 750 Td", "15 TL"]
        for line in lines:
            escaped = (
                line.replace("\\", "\\\\")
                .replace("(", "\\(")
                .replace(")", "\\)")
            )
            stream_lines.append(f"({escaped}) Tj T*")
        stream_lines.append("ET")
        stream_bytes = "\n".join(stream_lines).encode("latin-1", errors="ignore")
        content_id = add_obj(
            f"<< /Length {len(stream_bytes)} >>\nstream\n".encode("ascii")
            + stream_bytes
            + b"\nendstream"
        )
        content_obj_ids.append(content_id)

    pages_tree_id = add_obj(b"")

    for content_id in content_obj_ids:
        page_id = add_obj(
            f"<< /Type /Page /Parent {pages_tree_id} 0 R "
            f"/MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_obj_id} 0 R >> >> "
            f"/Contents {content_id} 0 R >>".encode("ascii")
        )
        page_obj_ids.append(page_id)

    kids_str = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
    objects[pages_tree_id - 1] = (
        f"<< /Type /Pages /Kids [{kids_str}] /Count {len(page_obj_ids)} >>".encode(
            "ascii"
        )
    )
    catalog_id = add_obj(
        f"<< /Type /Catalog /Pages {pages_tree_id} 0 R >>".encode("ascii")
    )

    out = bytearray(b"%PDF-1.4\n")
    offsets: List[int] = []
    for idx, obj_bytes in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{idx} 0 obj\n".encode("ascii"))
        out.extend(obj_bytes)
        out.extend(b"\nendobj\n")

    xref_pos = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for off in offsets:
        out.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    out.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n".encode("ascii")
    )
    file_path.write_bytes(bytes(out))


def resolve_backend_policy_pdfs() -> List[Path]:
    """Locate the 3 company policy PDFs silently in the backend."""
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()

    candidate_dirs = [
        cwd,
        cwd / "policies",
        cwd / "data",
        script_dir,
        script_dir / "policies",
        cwd / "user_uploaded_files" / "ttl=63d" / "user_uploaded",
        script_dir.parent.parent.parent.parent
        / "user_uploaded_files"
        / "ttl=63d"
        / "user_uploaded",
    ]

    found_by_name: Dict[str, Path] = {}
    for folder in candidate_dirs:
        if folder.is_dir():
            for pdf_name in POLICY_FILES:
                candidate = folder / pdf_name
                if candidate.is_file() and pdf_name not in found_by_name:
                    found_by_name[pdf_name] = candidate

    if len(found_by_name) < len(POLICY_FILES):
        fallback_dir = script_dir / "policies"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        for pdf_name, pages_text in FALLBACK_POLICY_PAGES.items():
            if pdf_name not in found_by_name:
                target_pdf = fallback_dir / pdf_name
                if not target_pdf.is_file():
                    _write_minimal_pdf(target_pdf, pages_text)
                found_by_name[pdf_name] = target_pdf

    return [found_by_name[name] for name in sorted(found_by_name.keys())]


def build_backend_knowledge_base(
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> Tuple[Optional[FAISS], List[Document]]:
    """Load, sanitize, chunk, and embed all backend policy PDFs into a single FAISS store."""
    pdf_paths = resolve_backend_policy_pdfs()
    all_pages: List[Document] = []

    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page_doc in pages:
            page_doc.page_content = sanitize_policy_text(page_doc.page_content)
            page_doc.metadata["source"] = pdf_path.name
        all_pages.extend(pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_pages)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, chunks

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store, chunks


def retrieve_policy_context(
    question: str,
    vector_store: Optional[FAISS],
    all_chunks: List[Document],
    domain_filter: str = "All Topics",
    top_k: int = 5,
) -> str:
    """Retrieve relevant internal knowledge silently without exposing metadata."""
    filter_map = {
        "Leave & Time Off": "01_People_and_Leave_Policy.pdf",
        "Hybrid & Equipment": "02_Hybrid_Work_and_Equipment.pdf",
        "Travel & Expenses": "03_Expenses_and_Travel_Policy.pdf",
    }
    target_source = filter_map.get(domain_filter)

    selected_chunks: List[Document] = []
    if vector_store is not None:
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": max(top_k * 2, 6), "fetch_k": 15},
        )
        candidates = retriever.invoke(question)
        if target_source:
            filtered = [
                c for c in candidates if c.metadata.get("source") == target_source
            ]
            selected_chunks = filtered[:top_k] if filtered else candidates[:top_k]
        else:
            selected_chunks = candidates[:top_k]
    else:
        query_terms = [
            w.lower() for w in re.findall(r"[a-zA-Z0-9]+", question) if len(w) > 2
        ]
        scored: List[Tuple[int, Document]] = []
        for chunk in all_chunks:
            if target_source and chunk.metadata.get("source") != target_source:
                continue
            text_lower = chunk.page_content.lower()
            score = sum(3 if term in text_lower else 0 for term in query_terms)
            scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected_chunks = [doc for _, doc in scored[:top_k]]

    # Combine all chunks so Maya always has complete context across the small policy set
    seen_texts = []
    for doc in selected_chunks + all_chunks:
        cleaned = doc.page_content.strip()
        if cleaned and cleaned not in seen_texts:
            seen_texts.append(cleaned)
    return "\n\n---\n\n".join(seen_texts)


def generate_human_response(
    question: str,
    domain_filter: str = "All Topics",
) -> str:
    """Generate a natural, human-like conversational response using RAG context and chat history."""
    vector_store: Optional[FAISS] = st.session_state.vector_store
    all_chunks: List[Document] = st.session_state.all_chunks

    context_str = retrieve_policy_context(
        question=question,
        vector_store=vector_store,
        all_chunks=all_chunks,
        domain_filter=domain_filter,
    )

    if os.getenv("OPENAI_API_KEY"):
        history_messages: List[BaseMessage] = []
        # Include recent conversation history so follow-ups feel completely natural
        for msg in st.session_state.messages[-8:]:
            if msg["role"] == "user":
                history_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                history_messages.append(AIMessage(content=msg["content"]))

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.35)
        chain = prompt | llm
        response = chain.invoke(
            {
                "context": context_str,
                "chat_history": history_messages,
                "question": question,
            }
        )
        raw_reply = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        return strip_any_accidental_citations(raw_reply)

    return (
        "Hey! It looks like my OpenAI connection isn't active right now—once "
        "`OPENAI_API_KEY` is set in your environment, we can chat about anything!"
    )


def inject_enterprise_css() -> None:
    """Apply sleek, human-centered enterprise styling with strong light/dark contrast."""
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header[data-testid="stHeader"] {background: transparent;}

        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
            max-width: 1120px;
        }

        /* Executive Concierge Hero */
        .ns-hero {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #1d4ed8 100%);
            border: 1px solid #334155;
            border-radius: 18px;
            padding: 24px 28px;
            margin-bottom: 18px;
            color: #f8fafc !important;
            box-shadow: 0 12px 28px -6px rgba(15, 23, 42, 0.28);
        }
        .ns-hero-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
            margin-bottom: 8px;
        }
        .ns-brand {
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: #7dd3fc !important;
        }
        .ns-status-badge {
            background: rgba(16, 185, 129, 0.2);
            border: 1px solid rgba(52, 211, 153, 0.45);
            color: #6ee7b7 !important;
            font-size: 0.78rem;
            font-weight: 600;
            padding: 4px 12px;
            border-radius: 999px;
        }
        .ns-hero h1 {
            font-size: 1.7rem;
            font-weight: 800;
            margin: 0 0 6px 0;
            color: #ffffff !important;
            letter-spacing: -0.02em;
        }
        .ns-hero p {
            font-size: 0.96rem;
            color: #e2e8f0 !important;
            margin: 0;
            line-height: 1.5;
        }

        /* Quick Glance Cards */
        .ns-kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-bottom: 18px;
        }
        .ns-kpi-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-left: 4px solid #2563eb;
            border-radius: 12px;
            padding: 12px 16px;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
        }
        .ns-kpi-label {
            font-size: 0.74rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #64748b !important;
            margin-bottom: 4px;
        }
        .ns-kpi-value {
            font-size: 1.12rem;
            font-weight: 800;
            color: #0f172a !important;
        }
        .ns-kpi-sub {
            font-size: 0.78rem;
            color: #475569 !important;
            margin-top: 2px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    """Initialize backend FAISS store and natural conversation state."""
    if "kb_initialized" not in st.session_state:
        vector_store, chunks = build_backend_knowledge_base()
        st.session_state.vector_store = vector_store
        st.session_state.all_chunks = chunks
        st.session_state.kb_initialized = True

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hey there! I'm **Maya**, your People & Workplace Partner here at Northstar Labs. "
                    "Whether you're planning time off, setting up your home office, sorting out travel expenses, "
                    "or just want to chat—what's on your mind today?"
                ),
            }
        ]

    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None


def render_policy_calculators() -> None:
    """Render self-service calculators without any raw document or page citations."""
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("🗓️ Annual Leave & Carryover Planner")
        st.caption(
            "Full-time team members receive 18 paid days per year (prorated and rounded up to the next half-day for new joiners) and can carry up to 5 unused days into the next year until March 31."
        )
        start_date = st.date_input(
            "Your 2026 Start Date",
            value=date(2026, 1, 1),
            min_value=date(2026, 1, 1),
            max_value=date(2026, 12, 31),
        )
        unused_days = st.number_input(
            "Expected Unused Days at Year-End",
            min_value=0.0,
            max_value=18.0,
            value=6.5,
            step=0.5,
        )

        days_in_year = 365
        active_days = (date(2026, 12, 31) - start_date).days + 1
        raw_entitlement = 18.0 * (active_days / days_in_year)
        prorated_entitlement = (
            int(raw_entitlement * 2) / 2.0
            if (raw_entitlement * 2).is_integer()
            else (int(raw_entitlement * 2) + 1) / 2.0
        )
        carried_over = min(unused_days, 5.0)
        forfeited = max(0.0, unused_days - 5.0)

        m1, m2, m3 = st.columns(3)
        m1.metric("2026 Allowance", f"{prorated_entitlement:.1f} days")
        m2.metric("Carries Over (til Mar 31)", f"{carried_over:.1f} days")
        m3.metric("Use Before Dec 31", f"{forfeited:.1f} days")

    with col2:
        st.subheader("🧾 Quick Expense Check")
        st.caption(
            "Check whether an expense can be reimbursed before submitting it in the expense portal."
        )
        expense_type = st.selectbox(
            "What are you claiming?",
            [
                "Overnight Business Travel Meals",
                "Home-Office Equipment Upgrade",
                "Business Travel Flight",
                "Daily Commute to the Office",
                "Alcohol or Personal Items",
            ],
        )
        amount = st.number_input(
            "Amount ($ USD)", min_value=0.0, max_value=5000.0, value=55.0, step=5.0
        )
        days_since = st.slider(
            "Days Since Purchase", min_value=1, max_value=60, value=14
        )

        if days_since > 30:
            st.error(
                f"⏳ **Heads-up on timing:** It's been {days_since} days since purchase—we need itemized receipts submitted within **30 days**. Reach out to Finance in the expense portal if you had an extenuating delay."
            )
        elif expense_type in (
            "Daily Commute to the Office",
            "Alcohol or Personal Items",
        ):
            st.error(
                f"🚫 **Not covered:** {expense_type} isn't eligible for reimbursement at Northstar Labs."
            )
        elif expense_type == "Overnight Business Travel Meals":
            if amount <= 60.0:
                st.success(
                    f"✅ **Good to go (${amount:.2f}):** That's comfortably within our **$60/day** overnight travel meal allowance (including tax and tip). Just attach your itemized receipt!"
                )
            else:
                excess = amount - 60.0
                st.warning(
                    f"⚠️ **Partial reimbursement ($60.00 max):** Our daily meal cap is **$60/day** including tax and tip, so **${excess:.2f}** would be out-of-pocket."
                )
        elif expense_type == "Home-Office Equipment Upgrade":
            if amount <= 250.0:
                st.success(
                    f"✅ **Good to go (${amount:.2f}):** You can claim up to **$250 once every two years** for approved home-office gear with manager approval and an itemized receipt."
                )
            else:
                excess = amount - 250.0
                st.warning(
                    f"⚠️ **Capped at $250.00:** Our 2-year home-office allowance covers up to **$250**, meaning the extra **${excess:.2f}** wouldn't be reimbursed."
                )
        else:
            st.info(
                "✈️ **Flight booking tip:** Make sure to get quick manager approval before booking, select **economy class**, and book through the company travel tool."
            )


def main() -> None:
    st.set_page_config(
        page_title="Northstar Labs | Maya — Workplace Partner",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_enterprise_css()
    init_session_state()

    # Executive Header
    st.markdown(
        """
        <div class="ns-hero">
            <div class="ns-hero-top">
                <span class="ns-brand">🧭 Northstar Labs • People &amp; Workplace Experience</span>
                <span class="ns-status-badge">● Maya is Online</span>
            </div>
            <h1>Maya — Your People &amp; Workplace Partner</h1>
            <p>Ask anything about time off, hybrid work, home-office perks, or travel—just like messaging a teammate on Slack.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # At-a-Glance Highlights
    st.markdown(
        """
        <div class="ns-kpi-grid">
            <div class="ns-kpi-card">
                <div class="ns-kpi-label">Time Off &amp; Wellness</div>
                <div class="ns-kpi-value">18 Annual • 10 Sick</div>
                <div class="ns-kpi-sub">Up to 5 days carry over to Mar 31</div>
            </div>
            <div class="ns-kpi-card">
                <div class="ns-kpi-label">Hybrid Rhythm</div>
                <div class="ns-kpi-value">Tue &amp; Wed In-Office</div>
                <div class="ns-kpi-sub">Core hours: 11:00 a.m. – 3:00 p.m.</div>
            </div>
            <div class="ns-kpi-card">
                <div class="ns-kpi-label">Home-Office Setup</div>
                <div class="ns-kpi-value">$250 Every 2 Years</div>
                <div class="ns-kpi-sub">Company laptop + accessories included</div>
            </div>
            <div class="ns-kpi-card">
                <div class="ns-kpi-label">Business Travel</div>
                <div class="ns-kpi-value">$60 / Day Meals</div>
                <div class="ns-kpi-sub">Economy flights • 30-day receipt window</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_chat, tab_tools = st.tabs(
        [
            "💬 Chat with Maya",
            "🧮 Leave & Expense Planners",
        ]
    )

    with tab_chat:
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2.2, 1.0, 0.8])
        with ctrl_col1:
            domain_filter = st.segmented_control(
                "Topic Focus",
                options=[
                    "All Topics",
                    "Leave & Time Off",
                    "Hybrid & Equipment",
                    "Travel & Expenses",
                ],
                default="All Topics",
                label_visibility="collapsed",
            )
        with ctrl_col2:
            transcript_text = "\n\n".join(
                f"{'Maya' if m['role'] == 'assistant' else 'You'}: {m['content']}"
                for m in st.session_state.messages
            )
            st.download_button(
                "⬇️ Save Conversation",
                data=transcript_text,
                file_name="chat_with_maya.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with ctrl_col3:
            if st.button("🔄 New Chat", use_container_width=True):
                st.session_state.messages = [st.session_state.messages[0]]
                st.rerun()

        st.caption("**Quick conversation starters:**")
        quick_prompts = [
            (
                "🗓️ Carrying over unused leave",
                "Hey Maya, I still have some unused vacation days left this year—how many can I carry over and when do they expire?",
            ),
            (
                "🏠 Hybrid days & flexible hours",
                "Which days do we need to be in the office, and how flexible are my hours outside the core window?",
            ),
            (
                "💻 Buying a monitor for home",
                "I want to buy an ergonomic chair or monitor for my home desk—how much can I expense and what's the process?",
            ),
            (
                "✈️ Per-diem & travel booking",
                "I have an overnight client trip next week. What's my daily meal budget and how should I book my flight?",
            ),
            (
                "👶 Parental & sick leave",
                "Could you walk me through our paid parental leave and when a doctor's note is needed for sick days?",
            ),
            (
                "🌍 Working while visiting family abroad",
                "Can I work remotely from another country for a week if I'm traveling, or do I need special sign-off?",
            ),
        ]
        q_cols = st.columns(3)
        for idx, (btn_label, prompt_text) in enumerate(quick_prompts):
            with q_cols[idx % 3]:
                if st.button(btn_label, key=f"qp_{idx}", use_container_width=True):
                    st.session_state.pending_prompt = prompt_text

        st.divider()

        # Render Conversation History with human avatars and zero PDF/page references
        for msg in st.session_state.messages:
            avatar_icon = "🧑‍💼" if msg["role"] == "assistant" else "🙂"
            with st.chat_message(msg["role"], avatar=avatar_icon):
                st.markdown(msg["content"])

        chat_box_value = st.chat_input(
            "Message Maya about anything at Northstar Labs..."
        )
        active_query = chat_box_value or st.session_state.pending_prompt
        if st.session_state.pending_prompt:
            st.session_state.pending_prompt = None

        if active_query:
            st.session_state.messages.append(
                {"role": "user", "content": active_query}
            )
            with st.chat_message("user", avatar="🙂"):
                st.markdown(active_query)

            with st.chat_message("assistant", avatar="🧑‍💼"):
                with st.spinner("Maya is typing..."):
                    try:
                        reply = generate_human_response(
                            question=active_query,
                            domain_filter=domain_filter or "All Topics",
                        )
                        st.markdown(reply)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": reply}
                        )
                    except Exception as exc:
                        err_text = f"Sorry, I hit a quick snag on my end: {exc}"
                        st.error(err_text)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": err_text}
                        )

    with tab_tools:
        render_policy_calculators()


if __name__ == "__main__":
    main()