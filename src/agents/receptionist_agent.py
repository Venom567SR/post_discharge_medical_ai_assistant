import re
from typing import Dict, Any, Optional
from loguru import logger

from src.agents.base_agent import BaseAgent
from src.schemas import AgentState
from src.config import (
    RECEPTIONIST_SYSTEM_PROMPT,
    HANDOFF_TO_CLINICAL
)
from src.tools.patient_db import lookup_patient
from src.llm.gemini import get_gemini_client
from src.llm.groq_fallback import get_groq_client
from src.logging_setup import log_handoff


class ReceptionistAgent(BaseAgent):
    """Receptionist agent for patient intake and routing."""
    
    def __init__(self):
        super().__init__("receptionist")
        self.llm = get_gemini_client()
        self.fallback_llm = get_groq_client()
    
    def process(self, state: AgentState) -> Dict[str, Any]:
        """
        Process receptionist tasks: greet, identify, look up patient, route queries.
        
        Args:
            state: Current agent state
        
        Returns:
            Dictionary of state updates
        """
        query = state.latest_query
        user_id = state.user_id
        
        self.log_action(user_id, "processing_query", {"query": query[:100]})
        
        # Step 1: Check if we need to extract patient name
        if not state.patient_name:
            name = self._extract_name(query)
            if name:
                state.patient_name = name
                self.log_action(user_id, "extracted_name", {"name": name})
                
                # Look up patient
                lookup_result = lookup_patient(name, user_id)
                
                if lookup_result.success:
                    state.patient_record = lookup_result.patient
                    self.log_action(user_id, "patient_found", {
                        "patient_id": lookup_result.patient.patient_id,
                        "diagnosis": lookup_result.patient.primary_diagnosis
                    })
                    
                    # Generate response with patient info
                    response = self._generate_patient_greeting(state)
                else:
                    # Patient not found or multiple matches
                    response = lookup_result.error
                    self.log_action(user_id, "patient_lookup_failed", {
                        "error_type": lookup_result.error_type
                    })
                
                return {
                    "patient_name": state.patient_name,
                    "patient_record": state.patient_record,
                    "latest_response": response,
                    "current_agent": "receptionist"
                }
        
        # Step 2: Check if this is a clinical question that should be routed
        if self._is_clinical_query(query):
            self.log_action(user_id, "routing_to_clinical", {"query": query[:100]})
            log_handoff(user_id, "receptionist", "clinical", "clinical_question_detected")
            
            state.handoffs.append("receptionist->clinical")
            
            # Provide transparent handoff message
            handoff_message = "This sounds like a medical concern. Let me connect you with our Clinical AI Agent."
            
            return {
                "current_agent": "clinical",
                "handoffs": state.handoffs,
                "latest_response": handoff_message,
                "show_handoff": True  # Signal to UI to show handoff status
            }
        
        # Step 3: Generate conversational response for follow-up questions
        response = self._generate_response(state, query)
        
        return {
            "latest_response": response,
            "current_agent": "receptionist"
        }
    
    def _extract_name(self, text: str) -> Optional[str]:
        """
        Extract patient name from text.
        
        Args:
            text: User message
        
        Returns:
            Extracted name or None
        """
        # Look for common patterns
        patterns = [
            r"(?:my name is|i'm|i am|this is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)$",  # Just a name
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1)
                # Capitalize properly
                return " ".join(word.capitalize() for word in name.split())
        
        return None
    
    def _is_clinical_query(self, query: str) -> bool:
        """
        Determine if query is clinical and should be routed to Clinical Agent.
        
        Args:
            query: User query
        
        Returns:
            True if clinical query
        """
        query_lower = query.lower()
        
        clinical_keywords = [
            # Symptoms and conditions
            "symptom", "pain", "swelling", "fever", "nausea", "headache",
            "kidney", "disease", "infection", "dysfunction", "failure",
            "chronic", "acute", "diagnosis", "condition",
            
            # Medical questions
            "what is", "what are", "how does", "why", "explain",
            "treatment", "medication", "side effect", "warning sign",
            "blood pressure", "dialysis", "creatinine", "gfr",
            
            # Time-sensitive
            "latest", "current", "guideline", "recommendation"
        ]
        
        return any(keyword in query_lower for keyword in clinical_keywords)
    
    def _generate_patient_greeting(self, state: AgentState) -> str:
        """
        Generate greeting with patient discharge information.
        
        Args:
            state: Current state with patient record
        
        Returns:
            Greeting message
        """
        patient = state.patient_record
        if not patient:
            return "I couldn't find your discharge information."
        
        # Build a warm, informative greeting
        greeting = f"""Hello {patient.name}! I found your discharge record from {patient.discharge_date}.

I see you were discharged with a diagnosis of {patient.primary_diagnosis}."""
        
        # Add follow-up questions based on discharge info
        if patient.medications:
            greeting += "\n\nHow are you managing your medications? Are you experiencing any issues?"
        
        if patient.warning_signs:
            greeting += "\n\nAre you experiencing any of the warning signs we discussed?"
        
        if patient.next_appointment:
            greeting += f"\n\nReminder: Your next appointment is scheduled for {patient.next_appointment}."
        
        greeting += "\n\nHow can I help you today?"
        
        return greeting
    
    def _generate_response(self, state: AgentState, query: str) -> str:
        """
        Generate conversational response using LLM.
        
        Args:
            state: Current state
            query: User query
        
        Returns:
            Generated response
        """
        # Build context
        context = self._build_context(state)
        
        # Try primary LLM (Gemini)
        try:
            response = self.llm.generate(
                system_prompt=RECEPTIONIST_SYSTEM_PROMPT,
                user_prompt=f"Context:\n{context}\n\nUser: {query}",
                user_id=state.user_id
            )
            return response
        except Exception as e:
            logger.warning(f"Primary LLM failed, trying fallback: {e}")
            
            # Try fallback (Groq)
            try:
                response = self.fallback_llm.generate(
                    system_prompt=RECEPTIONIST_SYSTEM_PROMPT,
                    user_prompt=f"Context:\n{context}\n\nUser: {query}",
                    user_id=state.user_id
                )
                return response
            except Exception as e2:
                logger.error(f"Fallback LLM also failed: {e2}")
                return "I'm having trouble processing your request right now. Please try again."
    
    def _build_context(self, state: AgentState) -> str:
        """
        Build context for LLM from state.
        
        Args:
            state: Current state
        
        Returns:
            Formatted context string
        """
        parts = []
        
        if state.patient_record:
            patient = state.patient_record
            parts.append(f"Patient: {patient.name}")
            parts.append(f"Diagnosis: {patient.primary_diagnosis}")
            parts.append(f"Discharge Date: {patient.discharge_date}")
            
            if patient.medications:
                parts.append(f"Medications: {', '.join(patient.medications[:3])}")
            
            if patient.warning_signs:
                parts.append(f"Warning Signs: {', '.join(patient.warning_signs[:2])}")
        
        # Add recent conversation
        if state.conversation_history:
            parts.append("\nRecent Conversation:")
            parts.append(self.format_conversation_context(state, last_n=3))
        
        return "\n".join(parts)