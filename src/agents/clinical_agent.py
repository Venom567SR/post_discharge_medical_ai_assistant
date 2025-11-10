from typing import Dict, Any
from loguru import logger

from src.agents.base_agent import BaseAgent
from src.schemas import AgentState, ClinicalResponse
from src.config import (
    CLINICAL_SYSTEM_PROMPT,
    RAG_TOP_K,
    RAG_SCORE_THRESHOLD
)
from src.rag.retriever import get_rag_retriever
from src.tools.web_search import search_web, should_use_web_search, format_search_results
from src.tools.citations import chunks_to_citations, deduplicate_citations
from src.llm.gemini import get_gemini_client
from src.llm.groq_fallback import get_groq_client
from src.logging_setup import log_retrieval


class ClinicalAgent(BaseAgent):
    """Clinical agent for evidence-based medical Q&A."""
    
    def __init__(self):
        super().__init__("clinical")
        self.llm = get_gemini_client()
        self.fallback_llm = get_groq_client()
        self.rag_retriever = get_rag_retriever()
    
    def process(self, state: AgentState) -> Dict[str, Any]:
        """
        Process clinical queries with RAG and web search.
        
        Args:
            state: Current agent state
        
        Returns:
            Dictionary of state updates
        """
        query = state.latest_query
        user_id = state.user_id
        
        processing_steps = []  # Track steps for transparency
        
        self.log_action(user_id, "processing_clinical_query", {"query": query[:100]})
        
        # Step 1: RAG retrieval
        chunks = []
        rag_citations = []
        
        if state.rag_enabled:
            try:
                processing_steps.append("searching_references")
                chunks, rag_citations = self.rag_retriever.retrieve_with_citations(
                    query=query,
                    k=RAG_TOP_K,
                    score_threshold=RAG_SCORE_THRESHOLD
                )
                
                top_scores = [chunk.score for chunk in chunks[:3]]
                log_retrieval(user_id, query, len(chunks), top_scores)
                
                self.log_action(user_id, "rag_retrieval", {
                    "chunks_found": len(chunks),
                    "top_scores": top_scores
                })
            except Exception as e:
                logger.error(f"RAG retrieval failed: {e}")
        
        # Step 2: Web search (if enabled and query is time-sensitive)
        web_results = []
        web_citations = []
        requires_web_search = should_use_web_search(query)
        
        if state.web_search_enabled and requires_web_search:
            try:
                processing_steps.append("searching_web")
                web_response = search_web(query, user_id=user_id)
                web_results = web_response.results
                
                # Convert to citations
                for result in web_results:
                    from src.schemas import Citation
                    citation = Citation(
                        source_type=result.source_type,
                        reference_id=None,
                        page=None,
                        url=result.url,
                        score=None
                    )
                    web_citations.append(citation)
                
                self.log_action(user_id, "web_search", {
                    "results_found": len(web_results),
                    "is_stub": web_response.is_stub
                })
            except Exception as e:
                logger.error(f"Web search failed: {e}")
        
        # Step 3: Build context for LLM
        context = self._build_context(state, chunks, web_results)
        
        # Step 4: Generate structured response with LLM
        clinical_response = self._generate_structured_response(
            state,
            query,
            context,
            rag_citations + web_citations
        )
        
        # Deduplicate citations
        clinical_response.citations = deduplicate_citations(clinical_response.citations)
        
        self.log_action(user_id, "clinical_response_generated", {
            "model": clinical_response.model_used,
            "citations_count": len(clinical_response.citations)
        })
        
        return {
            "latest_response": self._format_response(clinical_response),
            "current_agent": "clinical",
            "metadata": {
                "model_used": clinical_response.model_used,
                "citations": [c.dict() for c in clinical_response.citations],
                "rag_chunks": len(chunks),
                "web_results": len(web_results),
                "processing_steps": processing_steps,  # For transparency
                "required_web_search": requires_web_search
            }
        }
    
    def _build_context(self, state: AgentState, chunks, web_results) -> str:
        """
        Build context from RAG and web search results.
        
        Args:
            state: Current state
            chunks: Retrieved RAG chunks
            web_results: Web search results
        
        Returns:
            Formatted context string
        """
        parts = []
        
        # Add patient context if available
        if state.patient_record:
            patient = state.patient_record
            parts.append(f"Patient Context: {patient.name}, diagnosed with {patient.primary_diagnosis}")
        
        # Add RAG context
        if chunks:
            rag_context = self.rag_retriever.format_context(chunks)
            parts.append(f"Reference Database Information:\n{rag_context}")
        
        # Add web search context
        if web_results:
            web_context = format_search_results(
                type("Response", (), {
                    "results": web_results,
                    "query": state.latest_query,
                    "is_stub": any(r.source_type == "web_stub" for r in web_results)
                })()
            )
            parts.append(f"\n{web_context}")
        
        return "\n\n".join(parts)
    
    def _generate_structured_response(
        self,
        state: AgentState,
        query: str,
        context: str,
        citations: list
    ) -> ClinicalResponse:
        """
        Generate structured clinical response using LLM.
        
        Args:
            state: Current state
            query: User query
            context: Built context
            citations: List of citations
        
        Returns:
            ClinicalResponse object
        """
        # Build enhanced prompt
        prompt = f"""{context}

User Query: {query}

Please provide a comprehensive, evidence-based answer to the user's query.

Instructions:
1. Use the reference database information provided above
2. Include inline citations in your answer like [Ref p.14]
3. For web sources, mention them as (Web Source)
4. Keep language clear and patient-friendly
5. Acknowledge if information is limited
6. Always maintain a professional, supportive tone

Remember: Your response will be structured with citations automatically extracted."""
        
        # Try primary LLM (Gemini) with structured output
        try:
            response = self.llm.generate_structured(
                system_prompt=CLINICAL_SYSTEM_PROMPT,
                user_prompt=prompt,
                user_id=state.user_id
            )
            
            # Merge in the citations we collected
            if citations and not response.citations:
                response.citations = citations
            
            return response
        
        except Exception as e:
            logger.warning(f"Primary LLM failed, trying fallback: {e}")
            
            # Try fallback (Groq)
            try:
                response = self.fallback_llm.generate_structured(
                    system_prompt=CLINICAL_SYSTEM_PROMPT,
                    user_prompt=prompt,
                    user_id=state.user_id
                )
                
                if citations and not response.citations:
                    response.citations = citations
                
                return response
            
            except Exception as e2:
                logger.error(f"Fallback LLM also failed: {e2}")
                
                # Return stub response
                from src.schemas import Citation
                return ClinicalResponse(
                    answer="I'm having trouble accessing my knowledge base right now. Please try again or consult your healthcare provider directly.",
                    citations=citations,
                    model_used="stub_error",
                    disclaimer="This assistant is for educational purposes only. Always consult a licensed medical professional."
                )
    
    def _format_response(self, clinical_response: ClinicalResponse) -> str:
        """
        Format the clinical response for display.
        
        Args:
            clinical_response: ClinicalResponse object
        
        Returns:
            Formatted string
        """
        from src.tools.citations import format_citation_list
        
        parts = []
        
        # Add transparency prefix for web search
        has_web_sources = any(c.source_type == "web" for c in clinical_response.citations)
        if has_web_sources:
            parts.append("*This answer includes recent information from web sources.*\n")
        
        parts.append(clinical_response.answer)
        
        # Add citations if present
        if clinical_response.citations:
            citation_text = format_citation_list(clinical_response.citations)
            parts.append(f"\n\n{citation_text}")
        
        # Add disclaimer
        parts.append(f"\n\n⚠️ {clinical_response.disclaimer}")
        
        return "\n".join(parts)