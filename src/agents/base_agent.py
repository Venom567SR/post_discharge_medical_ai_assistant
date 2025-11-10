from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from loguru import logger

from src.schemas import AgentState
from src.logging_setup import log_agent_action


class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    def __init__(self, name: str):
        """
        Initialize base agent.
        
        Args:
            name: Agent identifier (e.g., 'receptionist', 'clinical')
        """
        self.name = name
        logger.info(f"Initialized {name} agent")
    
    @abstractmethod
    def process(self, state: AgentState) -> Dict[str, Any]:
        """
        Process the current state and return updates.
        
        Args:
            state: Current agent state
        
        Returns:
            Dictionary of state updates
        """
        pass
    
    def log_action(self, user_id: str, action: str, details: Dict[str, Any] = None):
        """
        Log an agent action.
        
        Args:
            user_id: User ID
            action: Action description
            details: Additional details
        """
        log_agent_action(
            user_id=user_id,
            agent=self.name,
            action=action,
            details=details or {}
        )
    
    def update_conversation(
        self,
        state: AgentState,
        role: str,
        content: str
    ) -> AgentState:
        """
        Add a message to conversation history.
        
        Args:
            state: Current state
            role: Message role (user/assistant)
            content: Message content
        
        Returns:
            Updated state
        """
        from src.schemas import ConversationMessage
        from datetime import datetime
        
        message = ConversationMessage(
            role=role,
            content=content,
            agent=self.name if role == "assistant" else None,
            timestamp=datetime.now()
        )
        
        state.conversation_history.append(message)
        return state
    
    def should_handoff(self, state: AgentState, query: str) -> Optional[str]:
        """
        Determine if this agent should hand off to another agent.
        
        Args:
            state: Current state
            query: User query
        
        Returns:
            Name of target agent, or None if no handoff needed
        """
        # Override in subclasses
        return None
    
    def format_conversation_context(self, state: AgentState, last_n: int = 5) -> str:
        """
        Format recent conversation history as context.
        
        Args:
            state: Current state
            last_n: Number of recent messages to include
        
        Returns:
            Formatted conversation string
        """
        if not state.conversation_history:
            return ""
        
        recent = state.conversation_history[-last_n:]
        lines = []
        
        for msg in recent:
            role_label = msg.role.capitalize()
            if msg.agent:
                role_label += f" ({msg.agent})"
            lines.append(f"{role_label}: {msg.content}")
        
        return "\n".join(lines)
