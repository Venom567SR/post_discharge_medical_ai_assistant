import time
import json
from typing import Optional, Dict, Any
from loguru import logger

from src.config import (
    GOOGLE_API_KEY,
    LLM_PRIMARY,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    has_google_key
)
from src.schemas import ClinicalResponse
from src.logging_setup import log_llm_call


class GeminiClient:
    """Client for Google Gemini API with structured output."""
    
    def __init__(self, api_key: str = GOOGLE_API_KEY, model: str = LLM_PRIMARY):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Google API key
            model: Model identifier
        """
        self.api_key = api_key
        self.model = model
        self.client = None
        
        if has_google_key():
            self._init_client()
    
    def _init_client(self):
        """Initialize the Gemini client."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.client = genai
            logger.info(f"Initialized Gemini client with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
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
        Generate text completion (plain text, for Receptionist Agent).
        
        Args:
            system_prompt: System instructions
            user_prompt: User message
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            user_id: User ID for logging
        
        Returns:
            Generated text
        """
        if not has_google_key() or self.client is None:
            logger.warning("Gemini not available, returning stub response")
            return self._stub_response()
        
        try:
            start_time = time.time()
            
            # Combine system and user prompts
            full_prompt = f"{system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
            
            model = self.client.GenerativeModel(self.model)
            response = model.generate_content(
                full_prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                }
            )

            # Check if response was blocked by safety filters
            if not response.candidates or not response.candidates[0].content.parts:
                finish_reason = response.candidates[0].finish_reason if response.candidates else "UNKNOWN"
                logger.warning(f"Gemini response blocked (finish_reason: {finish_reason})")
                raise Exception(f"Response blocked by safety filters (finish_reason: {finish_reason})")

            result = response.text
            
            # Log the call
            latency_ms = (time.time() - start_time) * 1000
            log_llm_call(
                user_id=user_id,
                model=self.model,
                prompt_length=len(full_prompt),
                response_length=len(result),
                latency_ms=latency_ms
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
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
        
        Uses Gemini's structured output feature with the ClinicalResponse schema.
        
        Args:
            system_prompt: System instructions
            user_prompt: User message
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            user_id: User ID for logging
        
        Returns:
            ClinicalResponse object
        """
        if not has_google_key() or self.client is None:
            logger.warning("Gemini not available, returning stub structured response")
            return self._stub_structured_response()
        
        try:
            start_time = time.time()
            
            # Full prompt with instructions for structured output
            full_prompt = f"""{system_prompt}

User Query: {user_prompt}

Please provide your response in the following JSON format:
{{
  "answer": "Your conversational, safe, plain-language response here",
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

IMPORTANT: Respond ONLY with valid JSON. Do not include any text outside the JSON object."""
            
            model = self.client.GenerativeModel(
                self.model,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                    "response_mime_type": "application/json"
                }
            )

            response = model.generate_content(full_prompt)

            # Check if response was blocked by safety filters
            if not response.candidates or not response.candidates[0].content.parts:
                finish_reason = response.candidates[0].finish_reason if response.candidates else "UNKNOWN"
                logger.warning(f"Gemini response blocked (finish_reason: {finish_reason})")
                raise Exception(f"Response blocked by safety filters (finish_reason: {finish_reason})")

            result_text = response.text

            # Parse JSON response
            try:
                result_dict = json.loads(result_text)
                clinical_response = ClinicalResponse(**result_dict)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from Gemini: {e}")
                logger.error(f"Raw response: {result_text[:500]}...")

                # Try to repair truncated JSON
                try:
                    # Attempt to extract the answer field even from incomplete JSON
                    import re
                    answer_match = re.search(r'"answer":\s*"((?:[^"\\]|\\.)*)"', result_text)
                    if answer_match:
                        answer = answer_match.group(1)
                        logger.info("Extracted answer from truncated JSON")
                        clinical_response = ClinicalResponse(
                            answer=answer,
                            citations=[],
                            model_used=self.model,
                            disclaimer="This assistant is for educational purposes only. Always consult a licensed medical professional."
                        )
                    else:
                        # Complete fallback
                        clinical_response = ClinicalResponse(
                            answer="I was able to research your question but encountered a formatting error. Please try asking again or rephrase your question.",
                            citations=[],
                            model_used=self.model,
                            disclaimer="This assistant is for educational purposes only. Always consult a licensed medical professional."
                        )
                except Exception as repair_error:
                    logger.error(f"Failed to repair JSON: {repair_error}")
                    clinical_response = ClinicalResponse(
                        answer=result_text if len(result_text) < 1000 else result_text[:1000],
                        citations=[],
                        model_used=self.model,
                        disclaimer="This assistant is for educational purposes only. Always consult a licensed medical professional."
                    )
            
            # Log the call
            latency_ms = (time.time() - start_time) * 1000
            log_llm_call(
                user_id=user_id,
                model=self.model,
                prompt_length=len(full_prompt),
                response_length=len(result_text),
                latency_ms=latency_ms
            )
            
            return clinical_response
        
        except Exception as e:
            logger.error(f"Gemini structured generation failed: {e}")
            return self._stub_structured_response()
    
    def _stub_response(self) -> str:
        """Return a stub response when Gemini is unavailable."""
        return (
            "I'm currently unable to access my knowledge base. "
            "This is a demonstration system. To enable full functionality, "
            "please add your GOOGLE_API_KEY to the .env file."
        )
    
    def _stub_structured_response(self) -> ClinicalResponse:
        """Return a stub structured response when Gemini is unavailable."""
        return ClinicalResponse(
            answer=(
                "I'm currently unable to access my medical knowledge base. "
                "This is a demonstration system. To enable full clinical Q&A functionality, "
                "please add your GOOGLE_API_KEY to the .env file. "
                "For actual medical advice, please consult with your healthcare provider."
            ),
            citations=[],
            model_used="stub_no_api_key",
            disclaimer="This assistant is for educational purposes only. Always consult a licensed medical professional."
        )


# Global singleton
_gemini_client = None

def get_gemini_client() -> GeminiClient:
    """Get or create the global Gemini client instance."""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiClient()
    return _gemini_client
