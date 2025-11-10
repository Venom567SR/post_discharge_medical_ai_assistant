# Post-Discharge Medical AI Assistant - POC

A multi-agent AI assistant for post-discharge patient care, featuring RAG-based medical Q&A, patient lookup, and guided follow-ups.

## 🏗️ Architecture Overview

- **Multi-Agent System**: LangGraph orchestrates Receptionist and Clinical agents with explicit handoffs
- **RAG Pipeline**: Qdrant vector store + sentence-transformers embeddings over nephrology reference PDF
- **LLM**: Gemini 2.5 Flash (primary) with Groq fallback
- **Web Search**: Tavily integration for time-sensitive queries
- **Backend**: FastAPI + Pydantic schemas
- **Frontend**: Streamlit chat interface
- **Logging**: Comprehensive loguru-based tracking

## 📋 Prerequisites

- Python 3.11+
- API Keys (optional for demo with stubs):
  - Google Gemini API key
  - Groq API key  
  - Tavily API key

## 🚀 Quick Start

### 1. Clone and Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API keys
# GOOGLE_API_KEY=your_key_here
# GROQ_API_KEY=your_key_here
# TAVILY_API_KEY=your_key_here
```

**Note**: The system works with stub responses if keys are missing - useful for testing without API costs.

### 3. Generate Patient Data

```bash
# Generate 30 dummy patient discharge reports
python -m src.utils.io --generate-patients 30
```

This creates JSON files in `src/data/patients/` with varied diagnoses, medications, and follow-up instructions.

### 4. Add Reference PDF

Place your nephrology reference PDF at:
```
src/data/references/comprehensive-clinical-nephrology.pdf
```

(Or any medical reference PDF - the system will chunk and index it)

### 5. Build RAG Index

```bash
# Process PDF and create vector embeddings
python -m src.rag.retriever --build-index src/data/references/comprehensive-clinical-nephrology.pdf
```

This creates a Qdrant collection with embedded chunks from the PDF.

### 6. Start Backend API

```bash
# Run FastAPI server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

API will be available at `http://localhost:8000`

Test with: `curl http://localhost:8000/health`

### 7. Launch Streamlit UI

```bash
# In a new terminal (with venv activated)
cd src
cd ui
streamlit run app_streamlit.py
```

UI opens at `http://localhost:8501`

## 🎯 Demo Workflow (5 minutes)

### Happy Path Demonstration

1. **Launch UI** (`streamlit run src.ui.app_streamlit.py`)

2. **Configure Keys** (Sidebar)
   - Enter API keys (or leave empty for stub mode)
   - Enable "Use RAG" and "Use Web Search" toggles

3. **Patient Identification**
   ```
   User: "Hi, my name is John Smith"
   Assistant: [Receptionist] Searches patient database...
   Found discharge report for John Smith
   ```

4. **Guided Follow-up**
   ```
   Assistant: I see you were discharged on [date] with [condition].
   How are you managing your medications?
   Are you experiencing any warning signs?
   ```

5. **Clinical Query with RAG**
   ```
   User: "What are the early signs of kidney dysfunction?"
   Assistant: [Clinical] Based on nephrology references...
   [Shows answer with citations: Ref p.14, Ref p.27]
   ```

6. **Web Search Fallback**
   ```
   User: "What are the latest treatment guidelines for CKD?"
   Assistant: [Clinical] Using web search for current information...
   [Shows results labeled as (Web Source)]
   ```

7. **View Logs**
   - Check `logs/app.log` for full interaction trace
   - Or use `/logs` API endpoint

### Expected Output Screenshots

**Patient Lookup:**
```
✓ Found patient: John Smith
  Condition: Chronic Kidney Disease Stage 3
  Discharge Date: 2024-01-15
  Medications: Lisinopril 10mg, Furosemide 40mg
```

**RAG Citation:**
```
Early signs of kidney dysfunction include decreased urine output,
swelling in extremities, and elevated creatinine levels [Ref p.14].
Regular monitoring of GFR is essential [Ref p.27].

Sources:
• comprehensive-clinical-nephrology.pdf, page 14 (score: 0.89)
• comprehensive-clinical-nephrology.pdf, page 27 (score: 0.82)
```

**Disclaimer (always shown):**
```
⚠️ This assistant is for educational purposes only.
   Always consult healthcare professionals for medical advice.
```

## 📡 API Endpoints

### Health Check
```bash
GET http://localhost:8000/health
Response: {"status": "ok", "timestamp": "2025-01-10T12:00:00"}
```

### Chat
```bash
POST http://localhost:8000/chat
Body: {
  "user_id": "user123",
  "message": "What are symptoms of kidney disease?",
  "session_id": "session456"
}
Response: {
  "answer": "...",
  "sources": [...],
  "agent": "clinical",
  "handoffs": ["receptionist->clinical"],
  "logs_pointer": "logs/app.log"
}
```

### Retrieve Patient
```bash
POST http://localhost:8000/retrieve_patient
Body: {"name": "John Smith"}
Response: {
  "patient": {...},
  "errors": []
}
```

### View Logs
```bash
GET http://localhost:8000/logs?n=50
Response: {
  "logs": [...]
}
```

## 🗂️ Project Structure

```
postdischarge-assistant/
├── requirements.txt              # Dependencies
├── .env.example                  # API key template
├── README.md                     # This file
├── src/
│   ├── config.py                # Configuration settings
│   ├── logging_setup.py         # Loguru configuration
│   ├── schemas.py               # Pydantic models
│   ├── utils/
│   │   ├── chunking.py         # Text chunking utilities
│   │   ├── io.py               # Patient data generation CLI
│   │   └── timing.py           # Performance timing
│   ├── data/
│   │   ├── patients/           # Generated patient JSONs
│   │   └── references/         # comprehensive-clinical-nephrology.pdf
│   ├── rag/
│   │   ├── embeddings.py       # Sentence transformer wrapper
│   │   ├── vectorstore_qdrant.py  # Qdrant vector store
│   │   ├── vectorstore_chroma.py  # ChromaDB alternate
│   │   └── retriever.py        # RAG pipeline
│   ├── tools/
│   │   ├── patient_db.py       # Patient lookup tool
│   │   ├── web_search.py       # Tavily search wrapper
│   │   └── citations.py        # Citation formatting
│   ├── llm/
│   │   ├── gemini.py           # Gemini 2.5 Flash wrapper
│   │   └── groq_fallback.py    # Groq fallback wrapper
│   ├── agents/
│   │   ├── base_agent.py       # Base agent class
│   │   ├── receptionist_agent.py  # Intake & routing
│   │   └── clinical_agent.py   # Medical Q&A with RAG
│   ├── graph/
│   │   ├── state.py            # LangGraph state schema
│   │   └── langgraph_builder.py  # Graph construction
│   ├── api/
│   │   └── main.py             # FastAPI application
│   └── ui/
│       └── app_streamlit.py    # Streamlit interface
└── reports/
    ├── Architecture_Brief.docx  # Design decisions
    ├── DESIGN_RATIONALE.md      # Design rationale and trade-offs
    └── README.md                # Reports directory documentation
```

## 🔧 Configuration

### Environment Variables (.env)
```bash
# Required for full functionality
GOOGLE_API_KEY=...
GROQ_API_KEY=...
TAVILY_API_KEY=...

# Optional overrides
VECTOR_STORE=qdrant          # or chroma
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_PRIMARY=gemini-2.5-flash
LLM_FALLBACK=llama-3.1-8b-instant
```

### Stub Mode (No API Keys)
The system gracefully degrades when API keys are missing:
- **LLM**: Returns canned medical responses
- **Web Search**: Returns explanatory stub with `source_type: "web_stub"`
- **Embeddings**: Still work offline (sentence-transformers)
- **Vector Store**: Still builds and queries

This allows testing the full flow without API costs.

## 📊 Logging

All interactions are logged to `logs/app.log` with:
- Timestamp and user_id
- Agent actions and handoffs
- Tool calls (patient lookup, RAG retrieval, web search)
- Retrieval scores and sources
- Errors with stack traces

Log rotation: 5MB max, 3 backup files.

View recent logs:
```bash
tail -f logs/app.log
# or
curl http://localhost:8000/logs?n=100
```

## 🧪 Testing

### Quick Smoke Test
```bash
# 1. Generate patients
python -m src.utils.io --generate-patients 5

# 2. Build index (with dummy PDF or real one)
python -m src.rag.retriever --build-index src/data/references/comprehensive-clinical-nephrology.pdf

# 3. Start API
uvicorn src.api.main:app --reload &

# 4. Test endpoints
curl http://localhost:8000/health
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test", "message": "Hi, my name is Sarah Johnson", "session_id": "test1"}'
```

### Manual Testing Scenarios
1. **Patient not found**: Use name not in generated data
2. **Multiple matches**: Use common first name only
3. **Clinical query**: Ask about symptoms or treatments
4. **Citation validation**: Check that page numbers match PDF
5. **Web search trigger**: Ask about "latest" or "current" guidelines
6. **Error handling**: Remove API keys, observe stub responses

## 🐛 Troubleshooting

### "No patients found"
- Check `src/data/patients/` has JSON files
- Run `python -m src.utils.io --generate-patients 30`

### "Vector store not initialized"
- Check `python -m src.rag.retriever --build-index` completed
- Verify `src/data/references/comprehensive-clinical-nephrology.pdf` exists

### "LLM unavailable"
- Verify .env has `GOOGLE_API_KEY` or `GROQ_API_KEY`
- Check API key validity
- System falls back to stubs if both missing

### ImportError
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt` again
- Check Python version: `python --version` (should be 3.11+)

### Streamlit won't start
- Check port 8501 is free: `lsof -i :8501`
- Try alternate port: `streamlit run src.ui.app_streamlit.py --server.port 8502`

## 📖 Documentation

- **Architecture Decisions**: See `reports/Architecture_Brief.docx`
- **Design Rationale**: See `reports/DESIGN_RATIONALE.md`
- **Reports Overview**: See `reports/README.md`
- **API Reference**: FastAPI auto-docs at `http://localhost:8000/docs`

## ⚠️ Disclaimers

This is a **proof-of-concept** educational tool. It should not be used for actual medical advice or patient care.

- Always consult licensed healthcare professionals
- Do not rely on this system for critical medical decisions
- Patient data is dummy/synthetic for demonstration only
- Not HIPAA compliant or validated for clinical use

## 📝 Development Notes

- Type hints used throughout
- Pydantic schemas for validation
- Graceful fallbacks for missing dependencies
- Deterministic behavior in stub mode
- Minimal external dependencies
- Clean separation of concerns (tools/agents/graph)

## 🤝 Contributing

This is a POC for demonstration. For production use:
1. Add authentication and authorization
2. Implement proper patient data security (encryption, HIPAA compliance)
3. Add comprehensive test suite
4. Enhance error handling and monitoring
5. Optimize vector store performance
6. Add conversation memory/context management
7. Implement feedback collection


## 🙏 Acknowledgments

- LangChain/LangGraph for agent orchestration
- Qdrant for vector similarity search
- Anthropic Claude for documentation assistance
- Google Gemini and Groq for LLM capabilities
