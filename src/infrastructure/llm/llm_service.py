"""
LLM Service for generating responses using OpenAI API.

This service provides a unified interface for LLM calls used in:
- TPAO loop reasoning
- Plan generation
- Step extraction
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    model: str
    tokens_used: int
    finish_reason: str


class LLMService:
    """
    Service for making LLM calls to OpenAI.
    
    Handles:
    - Chat completions
    - Structured output parsing
    - Error handling and retries
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5.4-mini",
        temperature: float = 0.7,
        max_completion_tokens: int = 1000
    ):
        """
        Initialize LLM service.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use (default: gpt-5.4-mini)
            temperature: Sampling temperature
            max_completion_tokens: Maximum tokens in response (for newer models use max_completion_tokens)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        
        self.model = model
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        
        logger.info(f"LLM Service initialized with model: {self.model}")
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_completion_tokens: Optional[int] = None
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            system_prompt: System instructions
            user_prompt: User query/request
            temperature: Override default temperature
            max_completion_tokens: Override default max completion tokens
        
        Returns:
            LLMResponse with generated content
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature or self.temperature,
                max_completion_tokens=max_completion_tokens or self.max_completion_tokens
            )
            
            content = response.choices[0].message.content or ""
            tokens = response.usage.total_tokens if response.usage else 0
            
            return LLMResponse(
                content=content,
                model=response.model,
                tokens_used=tokens,
                finish_reason=response.choices[0].finish_reason
            )
            
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            raise
    
    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate a JSON response from the LLM.
        
        Uses response_format to ensure valid JSON output.
        
        Args:
            system_prompt: System instructions (should request JSON output)
            user_prompt: User query/request
            temperature: Override default temperature
        
        Returns:
            Parsed JSON dict
        """
        response = self.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature or 0.3  # Lower temp for structured output
        )
        
        try:
            # Try to extract JSON from markdown code blocks if present
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json
            if content.startswith("```"):
                content = content[3:]  # Remove ```
            if content.endswith("```"):
                content = content[:-3]  # Remove trailing ```
            
            return json.loads(content.strip())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            logger.error(f"Response content: {response.content}")
            raise ValueError(f"LLM did not return valid JSON: {e}")
    
    def generate_simple(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_completion_tokens: Optional[int] = None
    ) -> LLMResponse:
        """
        Generate a response with a simple prompt (no separate system/user split).
        
        Args:
            prompt: The full prompt
            temperature: Override default temperature
            max_completion_tokens: Override default max completion tokens
        
        Returns:
            LLMResponse with generated content
        """
        return self.generate(
            system_prompt="You are a helpful AI assistant for hotel customer service.",
            user_prompt=prompt,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens
        )
    
    def generate_json_simple(
        self,
        prompt: str,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate a JSON response with a simple prompt.
        
        Args:
            prompt: The full prompt (should request JSON output)
            temperature: Override default temperature
        
        Returns:
            Parsed JSON dict
        """
        return self.generate_json(
            system_prompt="You are a helpful AI assistant for hotel customer service. Always respond with valid JSON.",
            user_prompt=prompt,
            temperature=temperature
        )
