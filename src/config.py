import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "src" / "data"
PATIENTS_DIR = DATA_DIR / "patients"
REFERENCES_DIR = DATA_DIR / "references"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
PATIENTS_DIR.mkdir(parents=True, exist_ok=True)
REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# API Keys (optional - system works with stubs if missing)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Vector store configuration
VECTOR_STORE = os.getenv("VECTOR_STORE", "qdrant")  # qdrant or chroma
QDRANT_PATH = str(PROJECT_ROOT / "qdrant_storage")
CHROMA_PATH = str(PROJECT_ROOT / "chroma_db")

# Embedding configuration
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", 
    "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DIMENSION = 384  # all-MiniLM-L6-v2 dimension

# LLM configuration
LLM_PRIMARY = os.getenv("LLM_PRIMARY", "gemini-2.5-flash")
LLM_FALLBACK = os.getenv("LLM_FALLBACK", "llama-3.1-8b-instant")
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 10000

# RAG configuration
CHUNK_SIZE = 512
CHUNK_OVERLAP = 150
RAG_TOP_K = 5
RAG_SCORE_THRESHOLD = 0.3

# Web search configuration
WEB_SEARCH_MAX_RESULTS = 5

# Logging configuration
LOG_FILE = str(LOGS_DIR / "app.log")
LOG_ROTATION = "5 MB"
LOG_RETENTION = 3
LOG_LEVEL = "INFO"

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000

# Agent prompts and constants
RECEPTIONIST_SYSTEM_PROMPT = """You are a friendly and professional medical receptionist AI assistant for post-discharge patient care.

Your responsibilities:
1. Greet patients warmly and collect their name
2. Look up their discharge information from our patient database
3. Ask 1-2 relevant follow-up questions based on their discharge report (medications, warning signs, follow-up appointments)
4. Route clinical/medical questions to the Clinical Agent
5. Maintain a supportive and empathetic tone

Important guidelines:
- Always be warm and professional
- Keep questions focused and relevant to their discharge
- If you don't find a patient, politely ask them to verify their name
- For medical questions, clearly indicate you're transferring to the clinical specialist
- Never provide medical advice yourself - route to Clinical Agent

Remember: You are the first point of contact. Make patients feel heard and supported."""

CLINICAL_SYSTEM_PROMPT = """You are an expert clinical AI assistant providing evidence-based medical information to post-discharge patients.

Your responsibilities:
1. Answer medical questions using the nephrology reference database (RAG)
2. Always cite your sources with page numbers [Ref p.X]
3. Use web search for time-sensitive or guideline-related queries
4. Clearly distinguish between reference material and web sources
5. Maintain a professional yet accessible tone

Critical guidelines:
- ALWAYS include inline citations in your answers [Ref p.X]
- For web sources, label as (Web Source: URL)
- Keep explanations clear and patient-friendly
- Acknowledge uncertainty when retrieval confidence is low
- Never make definitive diagnoses or prescribe treatments
- Always include the medical disclaimer

Your goal is educational support, not medical diagnosis or treatment."""

MEDICAL_DISCLAIMER = "This assistant is for educational purposes only. Always consult healthcare professionals for medical advice."

# Patient database schema expectations
PATIENT_SCHEMA_FIELDS = [
    "patient_id",
    "name",
    "discharge_date",
    "admission_date",
    "primary_diagnosis",
    "secondary_diagnoses",
    "procedures",
    "medications",
    "warning_signs",
    "follow_up_instructions",
    "next_appointment",
    "discharge_summary"
]

# Handoff messages
HANDOFF_TO_CLINICAL = "Transferring to clinical specialist for medical question..."
HANDOFF_TO_RECEPTIONIST = "Returning to receptionist for further assistance..."

# Error messages
ERROR_PATIENT_NOT_FOUND = "I couldn't find a patient with that name in our system. Could you please verify the spelling?"
ERROR_MULTIPLE_MATCHES = "I found multiple patients with similar names. Could you provide your full name?"
ERROR_LLM_UNAVAILABLE = "I'm having trouble connecting to my knowledge base right now. Please try again shortly."
ERROR_RAG_FAILED = "I couldn't retrieve information from our references at this time."
ERROR_WEB_SEARCH_FAILED = "Web search is temporarily unavailable."

def has_google_key() -> bool:
    """Check if Google API key is available."""
    return bool(GOOGLE_API_KEY)

def has_groq_key() -> bool:
    """Check if Groq API key is available."""
    return bool(GROQ_API_KEY)

def has_tavily_key() -> bool:
    """Check if Tavily API key is available."""
    return bool(TAVILY_API_KEY)

def has_any_llm_key() -> bool:
    """Check if any LLM API key is available."""
    return has_google_key() or has_groq_key()
