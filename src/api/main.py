from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.schemas import (
    ChatRequest,
    ChatResponse,
    PatientRetrieveRequest,
    PatientRetrieveResponse,
    HealthResponse,
    LogsResponse
)
from src.config import VECTOR_STORE, has_any_llm_key, has_tavily_key
from src.graph.langgraph_builder import process_message
from src.graph.session_manager import get_session_manager
from src.tools.patient_db import get_patient_database
from src.logging_setup import get_recent_logs, setup_logging
from src.tools.citations import format_citation_list
from src.schemas import Citation

# Initialize logging
setup_logging()

# Create FastAPI app
app = FastAPI(
    title="Post-Discharge Medical AI Assistant",
    description="Multi-agent assistant for post-discharge patient care with RAG and web search",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting Post-Discharge Medical AI Assistant API")
    
    # Load patient database
    db = get_patient_database()
    logger.info(f"Loaded {db.count()} patients")
    
    # Check LLM availability
    if not has_any_llm_key():
        logger.warning("No LLM API keys configured - system will use stub responses")
    
    # Check web search availability
    if not has_tavily_key():
        logger.warning("No Tavily API key - web search will use stub responses")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Post-Discharge Medical AI Assistant API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns system status and availability of services.
    """
    return HealthResponse(
        status="ok",
        vector_store=VECTOR_STORE,
        llm_available=has_any_llm_key(),
        web_search_available=has_tavily_key()
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Process a chat message through the multi-agent system.
    
    Args:
        request: ChatRequest with user_id, message, session_id
    
    Returns:
        ChatResponse with answer, sources, agent info, and handoffs
    """
    try:
        logger.info(f"Chat request from user {request.user_id}: {request.message[:100]}")
        
        # Process through LangGraph
        result = process_message(
            user_id=request.user_id,
            session_id=request.session_id,
            message=request.message,
            rag_enabled=request.rag_enabled,
            web_search_enabled=request.web_search_enabled
        )
        
        # Extract citations from metadata if available
        sources = []
        if "citations" in result.get("metadata", {}):
            citations_data = result["metadata"]["citations"]
            sources = [Citation(**c) for c in citations_data]
        
        response = ChatResponse(
            answer=result["answer"],
            sources=sources,
            agent=result["agent"],
            handoffs=result["handoffs"],
            logs_pointer="logs/app.log"
        )
        
        logger.info(f"Chat response sent to user {request.user_id}")
        return response
    
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve_patient", response_model=PatientRetrieveResponse, tags=["Patient"])
async def retrieve_patient(request: PatientRetrieveRequest):
    """
    Retrieve patient information by name.
    
    Args:
        request: PatientRetrieveRequest with name
    
    Returns:
        PatientRetrieveResponse with patient data or errors
    """
    try:
        logger.info(f"Patient retrieval request: {request.name}")
        
        db = get_patient_database()
        result = db.get_patient_by_name(request.name, user_id="api")
        
        if result.success:
            return PatientRetrieveResponse(
                patient=result.patient,
                errors=[]
            )
        else:
            return PatientRetrieveResponse(
                patient=None,
                errors=[result.error]
            )
    
    except Exception as e:
        logger.error(f"Error retrieving patient: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/logs", response_model=LogsResponse, tags=["Logs"])
async def get_logs(n: int = 100):
    """
    Retrieve recent log entries.
    
    Args:
        n: Number of log lines to retrieve (default: 100, max: 1000)
    
    Returns:
        LogsResponse with log lines
    """
    try:
        if n > 1000:
            n = 1000
        
        logs = get_recent_logs(n)
        
        return LogsResponse(
            logs=logs,
            count=len(logs)
        )
    
    except Exception as e:
        logger.error(f"Error retrieving logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/patients/count", tags=["Patient"])
async def get_patient_count():
    """Get the number of patients in the database."""
    db = get_patient_database()
    return {"count": db.count()}


@app.get("/patients/list", tags=["Patient"])
async def list_patient_names():
    """List all patient names in the database."""
    db = get_patient_database()
    return {"names": db.list_all_names()}


@app.get("/sessions/count", tags=["Session"])
async def get_session_count():
    """Get the number of active sessions."""
    session_manager = get_session_manager()
    return {"count": session_manager.count_active_sessions()}


@app.delete("/sessions/{session_id}", tags=["Session"])
async def clear_session(session_id: str):
    """Clear a specific session."""
    session_manager = get_session_manager()
    session_manager.clear_session(session_id)
    return {"message": f"Session {session_id} cleared"}


@app.delete("/sessions", tags=["Session"])
async def clear_all_sessions():
    """Clear all expired sessions."""
    session_manager = get_session_manager()
    session_manager.cleanup_expired()
    return {"message": "All expired sessions cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
