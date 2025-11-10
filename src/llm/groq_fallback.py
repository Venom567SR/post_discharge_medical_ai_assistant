import time
import json
from typing import Optional
from loguru import logger

from src.config import (
    GROQ_API_KEY,
    LLM_FALLBACK,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    has_groq_key,
    ERROR_LLM_UNAVAILABLE
)
from src.schemas import ClinicalResponse
from src.logging_setup import log_llm_call


class GroqClient:
    """Client for Groq API as LLM fallback."""
    
    def __init__(self, api_key: str = GROQ_API_KEY, model: str = LLM_FALLBACK):
        """
        Initialize Groq client.
        
        Args:
            api_key: Groq API key
            model: Model identifier (e.g., llama-3.1-8b-instant)
        """
        self.api_key = api_key
        self.model = model
        self.client = None
        
        if has_groq_key():
            self._init_client()
    
    def _init_client(self):
        """Initialize the Groq client."""
        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
            logger.info(f"Initialized Groq client with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            self.client = None
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        user_id: str = "unknown"
    ) -> str:
        """
        Generate text completion (plain text).
        
        Args:
            system_prompt: System instructions
            user_prompt: User message
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            user_id: User ID for logging
        
        Returns:
            Generated text
        """
        if not has_groq_key() or self.client is None:
            logger.warning("Groq not available, returning stub response")
            return self._stub_response()
        
        try:
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            result = response.choices[0].message.content
            
            # Log the call
            latency_ms = (time.time() - start_time) * 1000
            log_llm_call(
                user_id=user_id,
                model=self.model,
                prompt_length=len(system_prompt) + len(user_prompt),
                response_length=len(result),
                latency_ms=latency_ms
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Groq generation failed: {e}")
            return self._stub_response()
    
    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        user_id: str = "unknown"
    ) -> ClinicalResponse:
        """
        Generate structured response for Clinical Agent.
        
        Note: Groq doesn't have native structured output, so we'll request JSON
        in the prompt and parse it.
        
        Args:
            system_prompt: System instructions
            user_prompt: User message
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            user_id: User ID for logging
        
        Returns:
            ClinicalResponse object
        """
        if not has_groq_key() or self.client is None:
            logger.warning("Groq not available, returning stub structured response")
            return self._stub_structured_response()
        
        try:
            start_time = time.time()
            
            # Enhanced system prompt for JSON output
            enhanced_system = f"""{system_prompt}

CRITICAL INSTRUCTION: You MUST respond with ONLY a valid JSON object in this exact format:
{{
  "answer": "Your conversational, safe, plain-language response here with inline citations like [Ref p.14]",
  "citations": [
    {{
      "source_type": "reference or web",
      "reference_id": "filename or null",
      "page": page_number_or_null,
      "url": "url_or_null",
      "score": score_or_null
    }}
  ],
  "model_used": "{self.model}",
  "disclaimer": "This assistant is for educational purposes only. Always consult a licensed medical professional."
}}

Do NOT include any text outside of this JSON structure. Do NOT use markdown code blocks."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": enhanced_system},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            result_text = response.choices[0].message.content
            
            # Parse JSON response
            try:
                # Try to extract JSON if wrapped in markdown code blocks
                if "```json" in result_text:
                    json_start = result_text.find("```json") + 7
                    json_end = result_text.find("```", json_start)
                    result_text = result_text[json_start:json_end].strip()
                elif "```" in result_text:
                    json_start = result_text.find("```") + 3
                    json_end = result_text.find("```", json_start)
                    result_text = result_text[json_start:json_end].strip()
                
                result_dict = json.loads(result_text)
                clinical_response = ClinicalResponse(**result_dict)
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse JSON from Groq: {e}")
                logger.error(f"Raw response: {result_text}")
                # Fallback: wrap text response in structure
                clinical_response = ClinicalResponse(
                    answer=result_text,
                    citations=[],
                    model_used=self.model,
                    disclaimer="This assistant is for educational purposes only. Always consult a licensed medical professional."
                )
            
            # Log the call
            latency_ms = (time.time() - start_time) * 1000
            log_llm_call(
                user_id=user_id,
                model=self.model,
                prompt_length=len(enhanced_system) + len(user_prompt),
                response_length=len(result_text),
                latency_ms=latency_ms
            )
            
            return clinical_response
        
        except Exception as e:
            logger.error(f"Groq structured generation failed: {e}")
            return self._stub_structured_response()
    
    def _stub_response(self) -> str:
        """Return a stub response when Groq is unavailable."""
        return (
            "I'm currently unable to access my knowledge base. "
            "This is a demonstration system. To enable fallback LLM functionality, "
            "please add your GROQ_API_KEY to the .env file."
        )
    
    def _stub_structured_response(self) -> ClinicalResponse:
        """Return a stub structured response when Groq is unavailable."""
        return ClinicalResponse(
            answer=(
                "I'm currently unable to access my medical knowledge base. "
                "Both primary and fallback LLM services are unavailable. "
                "Please configure GOOGLE_API_KEY or GROQ_API_KEY in the .env file. "
                "For actual medical advice, please consult with your healthcare provider."
            ),
            citations=[],
            model_used="stub_no_api_key",
            disclaimer="This assistant is for educational purposes only. Always consult a licensed medical professional."
        )


# Global singleton
_groq_client = None

def get_groq_client() -> GroqClient:
    """Get or create the global Groq client instance."""
    global _groq_client
    if _groq_client is None:
        _groq_client = GroqClient()
    return _groq_client


class LLMUnavailable(Exception):
    """Exception raised when no LLM is available."""
    pass
