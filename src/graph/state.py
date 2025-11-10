from typing import TypedDict, List, Optional, Dict, Any, Annotated
from operator import add

from src.schemas import PatientRecord, ConversationMessage


class GraphState(TypedDict):
    """
    State passed between nodes in the LangGraph.
    
    Uses TypedDict for LangGraph compatibility.
    """
    # User identification
    user_id: str
    session_id: str
    
    # Patient information
    patient_name: Optional[str]
    patient_record: Optional[PatientRecord]
    
    # Conversation tracking
    conversation_history: Annotated[List[ConversationMessage], add]  # Append-only list
    latest_query: str
    latest_response: str
    
    # Agent routing
    current_agent: str  # 'receptionist' or 'clinical'
    handoffs: Annotated[List[str], add]  # Track agent transitions
    
    # Feature flags
    rag_enabled: bool
    web_search_enabled: bool
    
    # Metadata
    metadata: Dict[str, Any]


def create_initial_state(
    user_id: str,
    session_id: str,
    message: str,
    rag_enabled: bool = True,
    web_search_enabled: bool = True
) -> GraphState:
    """
    Create initial state for a new conversation turn.
    
    Args:
        user_id: User identifier
        session_id: Session identifier
        message: User's message
        rag_enabled: Whether RAG is enabled
        web_search_enabled: Whether web search is enabled
    
    Returns:
        Initial GraphState
    """
    return GraphState(
        user_id=user_id,
        session_id=session_id,
        patient_name=None,
        patient_record=None,
        conversation_history=[],
        latest_query=message,
        latest_response="",
        current_agent="receptionist",
        handoffs=[],
        rag_enabled=rag_enabled,
        web_search_enabled=web_search_enabled,
        metadata={}
    )


def state_to_agent_state(graph_state: GraphState):
    """
    Convert GraphState to AgentState for agent processing.
    
    Args:
        graph_state: LangGraph state
    
    Returns:
        AgentState object
    """
    from src.schemas import AgentState
    
    return AgentState(
        user_id=graph_state["user_id"],
        session_id=graph_state["session_id"],
        patient_name=graph_state.get("patient_name"),
        patient_record=graph_state.get("patient_record"),
        conversation_history=graph_state.get("conversation_history", []),
        latest_query=graph_state["latest_query"],
        current_agent=graph_state["current_agent"],
        handoffs=graph_state.get("handoffs", []),
        rag_enabled=graph_state["rag_enabled"],
        web_search_enabled=graph_state["web_search_enabled"],
        metadata=graph_state.get("metadata", {})
    )


def agent_state_to_updates(agent_state) -> Dict[str, Any]:
    """
    Convert AgentState back to state updates for GraphState.
    
    Args:
        agent_state: AgentState object
    
    Returns:
        Dictionary of updates
    """
    return {
        "patient_name": agent_state.patient_name,
        "patient_record": agent_state.patient_record,
        "conversation_history": agent_state.conversation_history,
        "current_agent": agent_state.current_agent,
        "handoffs": agent_state.handoffs,
        "metadata": agent_state.metadata
    }
