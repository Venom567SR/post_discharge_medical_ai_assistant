from typing import Literal
from loguru import logger

from langgraph.graph import StateGraph, END
from src.graph.state import GraphState, state_to_agent_state, create_initial_state
from src.graph.session_manager import get_session_manager
from src.agents.receptionist_agent import ReceptionistAgent
from src.agents.clinical_agent import ClinicalAgent
from src.schemas import ConversationMessage


# Initialize agents
receptionist_agent = ReceptionistAgent()
clinical_agent = ClinicalAgent()


def receptionist_node(state: GraphState) -> GraphState:
    """
    Process state through Receptionist Agent.

    Args:
        state: Current graph state

    Returns:
        Updated graph state
    """
    logger.info(f"Receptionist node processing: {state['latest_query'][:100]}")

    # Convert to AgentState
    agent_state = state_to_agent_state(state)

    # Process with receptionist agent
    updates = receptionist_agent.process(agent_state)

    # Update conversation history only if response is not empty
    # (empty response means handoff to another agent)
    if updates.get("latest_response") and updates["latest_response"].strip():
        message = ConversationMessage(
            role="assistant",
            content=updates["latest_response"],
            agent="receptionist"
        )
        state["conversation_history"].append(message)
        state["latest_response"] = updates["latest_response"]

    # Apply other updates
    for key, value in updates.items():
        if key in state and key != "conversation_history":
            state[key] = value

    return state


def clinical_node(state: GraphState) -> GraphState:
    """
    Process state through Clinical Agent.
    
    Args:
        state: Current graph state
    
    Returns:
        Updated graph state
    """
    logger.info(f"Clinical node processing: {state['latest_query'][:100]}")
    
    # Convert to AgentState
    agent_state = state_to_agent_state(state)
    
    # Process with clinical agent
    updates = clinical_agent.process(agent_state)
    
    # Update conversation history
    if updates.get("latest_response"):
        message = ConversationMessage(
            role="assistant",
            content=updates["latest_response"],
            agent="clinical"
        )
        state["conversation_history"].append(message)
        state["latest_response"] = updates["latest_response"]
    
    # Apply other updates
    for key, value in updates.items():
        if key in state and key != "conversation_history":
            state[key] = value
    
    return state


def route_node(state: GraphState) -> Literal["receptionist", "clinical", "end"]:
    """
    Determine next node based on current state.
    
    Args:
        state: Current graph state
    
    Returns:
        Next node name
    """
    current_agent = state.get("current_agent", "receptionist")
    
    logger.info(f"Routing from current_agent: {current_agent}")
    
    # If current agent is receptionist and should route to clinical
    if current_agent == "clinical":
        # After clinical response, end the turn (could add logic to return to receptionist)
        return "end"
    
    # Default: stay with receptionist or route to clinical if indicated
    return current_agent


def build_graph() -> StateGraph:
    """
    Build the LangGraph for multi-agent orchestration.
    
    Returns:
        Compiled StateGraph
    """
    # Create graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("receptionist", receptionist_node)
    workflow.add_node("clinical", clinical_node)
    
    # Set entry point
    workflow.set_entry_point("receptionist")
    
    # Add conditional edges from receptionist
    workflow.add_conditional_edges(
        "receptionist",
        lambda state: state.get("current_agent", "receptionist"),
        {
            "receptionist": END,  # If staying with receptionist, end turn
            "clinical": "clinical"  # If routing to clinical, go there
        }
    )
    
    # Add edge from clinical to end
    workflow.add_edge("clinical", END)
    
    # Compile the graph
    graph = workflow.compile()
    
    logger.info("LangGraph compiled successfully")
    return graph


# Global graph instance
_graph = None

def get_graph():
    """Get or create the global graph instance."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def process_message(
    user_id: str,
    session_id: str,
    message: str,
    rag_enabled: bool = True,
    web_search_enabled: bool = True
) -> dict:
    """
    Process a user message through the agent graph.

    Args:
        user_id: User identifier
        session_id: Session identifier
        message: User's message
        rag_enabled: Whether RAG is enabled
        web_search_enabled: Whether web search is enabled

    Returns:
        Dictionary with response and metadata
    """
    session_manager = get_session_manager()

    # Check for existing session state
    previous_state = session_manager.get_session(session_id)

    if previous_state:
        # Merge with existing session state
        logger.info(f"Continuing session {session_id} with existing patient record")
        initial_state = previous_state.copy()
        initial_state["latest_query"] = message
        initial_state["latest_response"] = ""
        initial_state["current_agent"] = "receptionist"  # Always start with receptionist
    else:
        # Create fresh initial state
        logger.info(f"Starting new session {session_id}")
        initial_state = create_initial_state(
            user_id=user_id,
            session_id=session_id,
            message=message,
            rag_enabled=rag_enabled,
            web_search_enabled=web_search_enabled
        )

    # Add user message to conversation history
    user_message = ConversationMessage(
        role="user",
        content=message,
        agent=None
    )
    initial_state["conversation_history"].append(user_message)

    # Get graph and execute
    graph = get_graph()

    try:
        # Run the graph
        final_state = graph.invoke(initial_state)

        # Save session state for next turn
        session_manager.save_session(session_id, final_state)

        # Extract response
        return {
            "answer": final_state.get("latest_response", ""),
            "agent": final_state.get("current_agent", "receptionist"),
            "handoffs": final_state.get("handoffs", []),
            "metadata": final_state.get("metadata", {}),
            "patient_found": final_state.get("patient_record") is not None
        }

    except Exception as e:
        logger.error(f"Error processing message through graph: {e}")
        return {
            "answer": "I encountered an error processing your request. Please try again.",
            "agent": "error",
            "handoffs": [],
            "metadata": {"error": str(e)},
            "patient_found": False
        }
