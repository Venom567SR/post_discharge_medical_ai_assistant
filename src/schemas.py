from typing import Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field


# ============ Patient Data Models ============

class PatientRecord(BaseModel):
    """Patient discharge record schema."""
    patient_id: str
    name: str
    discharge_date: str
    admission_date: str
    primary_diagnosis: str
    secondary_diagnoses: list[str] = []
    procedures: list[str] = []
    medications: list[str] = []
    warning_signs: list[str] = []
    follow_up_instructions: list[str] = []
    next_appointment: Optional[str] = None
    discharge_summary: str

class PatientLookupResult(BaseModel):
    """Result of patient database lookup."""
    success: bool
    patient: Optional[PatientRecord] = None
    error: Optional[str] = None
    error_type: Optional[str] = None  # not_found, multiple_matches, system_error


# ============ RAG/Citation Models ============

class Citation(BaseModel):
    """Citation for a piece of retrieved information."""
    source_type: str = Field(..., description="'reference' or 'web'")
    reference_id: Optional[str] = Field(None, description="e.g., 'comprehensive-clinical-nephrology.pdf'")
    page: Optional[int] = Field(None, description="Page number if from PDF")
    url: Optional[str] = Field(None, description="URL if from web search")
    score: Optional[float] = Field(None, description="Retrieval similarity score")
    snippet: Optional[str] = Field(None, description="Text snippet")

class RetrievedChunk(BaseModel):
    """A chunk of text retrieved from vector store."""
    text: str
    source: str  # filename
    page: Optional[int] = None
    score: float
    metadata: dict[str, Any] = {}


# ============ LLM Response Models ============

class ClinicalResponse(BaseModel):
    """Structured output from Clinical Agent (Gemini structured output)."""
    answer: str = Field(..., description="Conversational, safe, plain-language response")
    citations: list[Citation] = Field(default_factory=list)
    model_used: str = Field(..., description="e.g., gemini-2.5-flash, llama-3.1-8b-groq")
    disclaimer: str = Field(default="This assistant is for educational purposes only. Always consult a licensed medical professional.")


# ============ Agent State Models ============

class ConversationMessage(BaseModel):
    """Single message in conversation history."""
    role: str  # user, assistant, system
    content: str
    agent: Optional[str] = None  # receptionist, clinical
    timestamp: datetime = Field(default_factory=datetime.now)

class AgentState(BaseModel):
    """State passed between agents in LangGraph."""
    user_id: str
    session_id: str
    patient_name: Optional[str] = None
    patient_record: Optional[PatientRecord] = None
    conversation_history: list[ConversationMessage] = []
    latest_query: str = ""
    current_agent: str = "receptionist"
    handoffs: list[str] = []  # track agent transitions
    rag_enabled: bool = True
    web_search_enabled: bool = True
    metadata: dict[str, Any] = {}

    class Config:
        arbitrary_types_allowed = True


# ============ API Request/Response Models ============

class ChatRequest(BaseModel):
    """Request to /chat endpoint."""
    user_id: str
    message: str
    session_id: str
    rag_enabled: bool = True
    web_search_enabled: bool = True

class ChatResponse(BaseModel):
    """Response from /chat endpoint."""
    answer: str
    sources: list[Citation] = []
    agent: str
    handoffs: list[str] = []
    logs_pointer: str = "logs/app.log"
    timestamp: datetime = Field(default_factory=datetime.now)

class PatientRetrieveRequest(BaseModel):
    """Request to /retrieve_patient endpoint."""
    name: str

class PatientRetrieveResponse(BaseModel):
    """Response from /retrieve_patient endpoint."""
    patient: Optional[PatientRecord] = None
    errors: list[str] = []

class HealthResponse(BaseModel):
    """Response from /health endpoint."""
    status: str = "ok"
    timestamp: datetime = Field(default_factory=datetime.now)
    vector_store: str
    llm_available: bool
    web_search_available: bool

class LogsResponse(BaseModel):
    """Response from /logs endpoint."""
    logs: list[str]
    count: int


# ============ Tool Call Models ============

class ToolCallResult(BaseModel):
    """Generic tool call result."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: dict[str, Any] = {}

class WebSearchResult(BaseModel):
    """Result from web search tool."""
    title: str
    url: str
    snippet: str
    source_type: str = "web"  # or "web_stub" if using fallback

class WebSearchResponse(BaseModel):
    """Response from web search tool."""
    results: list[WebSearchResult]
    query: str
    is_stub: bool = False  # True if no API key available
