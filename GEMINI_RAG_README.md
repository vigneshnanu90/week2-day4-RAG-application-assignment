# 🧭 Northstar Labs — Maya (People & Workplace AI Partner)

An enterprise-grade, human-centric **Multi-PDF Retrieval-Augmented Generation (RAG)** application built with **Python**, **LangChain**, **FAISS**, **OpenAI**, and **Streamlit**.

Designed for **Northstar Labs**, this portal pairs employees with **Maya**, an AI People & Workplace Partner who answers questions across company policies in a warm, natural, colleague-like voice—while handling all PDF ingestion, text sanitization, chunking, and vector indexing silently in the backend.

## ✨ Key Capabilities

* **Silent Backend Multi-PDF RAG Pipeline:** Automatically discovers, loads, sanitizes, chunks, and embeds multiple company policy PDFs into a unified **FAISS** vector index stored in `st.session_state`—with zero manual file-upload clutter exposed in the UI.
* **Human-Like Conversational Persona (Maya):** Powered by `gpt-4o-mini` and multi-turn `MessagesPlaceholder` conversation memory, Maya responds like an experienced workplace teammate rather than a scripted bot—answering policy questions accurately while handling greetings, follow-ups, and casual banter (like sports scores or small talk) with natural warmth.
* **Clean, Citation-Free User Experience:** All internal policy rules are synthesized seamlessly into natural conversation without exposing raw PDF filenames, chunk boundaries, or robotic page-number tags in the chat UI.
* **Automatic Policy Sanitization & Self-Healing Bootstrap:** Strips legacy boilerplate strings during ingestion and automatically provisions the three official Northstar Labs policy PDFs in `./policies/` if launched in a fresh directory.
* **Interactive Self-Service Calculators:**
  * **Annual Leave & Carryover Planner:** Calculates prorated 2026 annual leave (rounded up to the next half-day for new joiners) and tracks the 5-day carryover cap expiring on March 31.
  * **Quick Expense Eligibility Checker:** Validates overnight business travel meals ($60/day cap), home-office equipment upgrades ($250 every two years), economy flight rules, and the 30-day itemized receipt window.
* **Executive Portal UI:** Custom CSS hero banner, at-a-glance policy KPI cards, topic focus filter pills, one-click conversation starters, and conversation transcript export.

## 📂 Covered Company Policies

| Policy Document | Core Workplace Provisions |
| :--- | :--- |
| **People & Leave Policy** (`01_People_and_Leave_Policy.pdf`) | 18 paid annual leave days/year (prorated for new joiners, rounded up to the next 0.5 day); 5-day carryover limit expiring March 31; 10 paid sick days/year (medical certificate after 3 consecutive working days); 16 weeks primary / 6 weeks secondary paid parental leave. |
| **Hybrid Work & Equipment Policy** (`02_Hybrid_Work_and_Equipment.pdf`) | Hybrid schedule with Tuesday & Wednesday in-office; core collaboration hours from 11:00 a.m. to 3:00 p.m. local time (Mon–Fri); written People Operations approval required before working from another country; $250 home-office equipment allowance once every 2 years. |
| **Expenses & Travel Policy** (`03_Expenses_and_Travel_Policy.pdf`) | Itemized receipt and business purpose required within 30 days of purchase; manager pre-approval and economy class for flights; up to $60/day for overnight travel meals (including tax and tip); alcohol, personal items, fines, and daily office commuting are non-reimbursable. |

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Frontend & UI** | Streamlit (Custom Enterprise CSS, Tabs, Segmented Controls, Chat UI) |
| **Orchestration** | LangChain (`langchain-core`, `langchain-community`, `langchain-text-splitters`) |
| **Vector Database** | FAISS (`faiss-cpu`) stored in `st.session_state` |
| **Embeddings & LLM** | OpenAI `text-embedding-3-small` & `gpt-4o-mini` (`langchain-openai`) |
| **Document Parsing** | `pypdf` (`PyPDFLoader`) |
| **Environment Config** | `python-dotenv` (`.env` / `OPENAI_API_KEY`) |

## 📁 Project Structure

```text
northstar-workplace-partner/
├── policies/
│   ├── 01_People_and_Leave_Policy.pdf
│   ├── 02_Hybrid_Work_and_Equipment.pdf
│   └── 03_Expenses_and_Travel_Policy.pdf
├── rag_app.py
├── .env
└── README.md
```

## 🚀 Local Setup & Execution (Windows `cmd` & VS Code)

### 1. Create and Activate a Virtual Environment

Open Command Prompt (`cmd`) in your project directory and run:

```cmd
python -m venv venv
venv\Scripts\activate
```

### 2. Install Required Dependencies

```cmd
pip install streamlit langchain langchain-community langchain-core langchain-openai langchain-text-splitters faiss-cpu pypdf python-dotenv
```

### 3. Configure Your OpenAI API Key

Create a `.env` file in the root folder of the project:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

*(Alternatively, set it directly in Command Prompt via `set OPENAI_API_KEY=your_openai_api_key_here`.)*

### 4. Place Policy PDFs (Optional Auto-Bootstrap)

Place `01_People_and_Leave_Policy.pdf`, `02_Hybrid_Work_and_Equipment.pdf`, and `03_Expenses_and_Travel_Policy.pdf` in either the project root or a `./policies` subfolder. If the files are not present on first run, `rag_app.py` automatically generates them inside `./policies/`.

### 5. Run the Application

```cmd
streamlit run rag_app.py
```

The app will open automatically in your browser at `http://localhost:8501`.
