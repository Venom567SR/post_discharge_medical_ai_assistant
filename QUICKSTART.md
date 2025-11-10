# Post-Discharge Medical AI Assistant

This project is a proof-of-concept (POC) medical assistant designed to help patients after hospital discharge by providing guidance, answering medical questions, and referencing clinical material responsibly. It demonstrates multi-agent orchestration, retrieval-augmented generation (RAG), structured outputs, and safe fallback logic.

> ⚠️ **Disclaimer:** This assistant is for educational and demonstration purposes only. It is *not* a substitute for professional medical advice.

---

## ✅ Key Capabilities

| Feature | Description |
|--------|-------------|
| **Multi-Agent System** | Receptionist Agent (intake & patient context) + Clinical Agent (medical reasoning & evidence guidance) |
| **RAG over Medical PDFs** | Extracts, embeds, and retrieves evidence from a nephrology reference PDF |
| **Web Search Fallback** | Supports Tavily for up-to-date medical sources (stub fallback if no key) |
| **Structured LLM Output** | Clinical answers include citations, model trace, and disclaimers |
| **Vector DB** | Uses Qdrant for semantic search (Chroma optional alternative) |
| **Streamlit Chat UI** | Simple front-end with agent handoff indications and source transparency |
| **FastAPI Backend** | Clean endpoints for chat, logs, patient lookup, and system health |
| **Logging** | Full request/response trace for debuggability and evaluation |

---

## 🧱 Project Structure

```
postdischarge-assistant/
├── requirements.txt
├── .env.example
├── README.md
├── src/
│   ├── config.py
│   ├── logging_setup.py
│   ├── schemas.py
│   ├── utils/
│   ├── data/
│   ├── rag/
│   ├── tools/
│   ├── llm/
│   ├── agents/
│   ├── graph/
│   ├── api/
│   └── ui/
└── reports/
```

---

## 🚀 Quick Setup

### 1. Create and activate environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Keys
```bash
cp .env.example .env
```
Add any available keys to `.env`:
```
GOOGLE_API_KEY=
GROQ_API_KEY=
TAVILY_API_KEY=
```
The system runs even without keys (uses fallback/stub behavior).

### 3. Generate sample patient profiles
```bash
python -m src.utils.io --generate-patients 30
```

### 4. Add your reference medical PDF
Place under:
```
src/data/references/
```
Then index it:
```bash
python -m src.rag.retriever --build-index src/data/references/comprehensive-clinical-nephrology.pdf
```

### 5. Start backend and UI
```bash
# Backend
uvicorn src.api.main:app --reload

# UI
cd src
cd ui
streamlit run src/ui/app_streamlit.py
```

Open the app at:  
**http://localhost:8501**

---

## 💬 Example Interaction

```
User: Hi, my name is Rohan Verma.
→ System retrieves discharge record.

User: I'm experiencing swelling again. Should I be worried?
→ Clinical Agent responds with evidence + citations + disclaimer.
```

---

## 🏗 Architecture Highlights

- LangGraph manages agent routing cleanly
- Qdrant enables fast semantic retrieval
- Gemini provides structured medical guidance output
- Groq model acts as inference fallback
- UI displays clear agent transitions + citation transparency

---

## 🐛 Troubleshooting

| Issue | Fix |
|------|-----|
| API not reachable | Ensure `uvicorn` server is running |
| No patients found | Re-run `--generate-patients` |
| RAG not retrieving | Reindex the PDF |
| UI not updating | Refresh browser + check logs/app.log |

---

## 🔜 Potential Extensions

- EHR / FHIR integration
- Authentication and user identity mapping
- Differential privacy on stored patient data
- More clinical specialties (cardiology, endocrinology, etc.)

---

**This project is intended for interview demonstration, evaluation, and research exploration.**
