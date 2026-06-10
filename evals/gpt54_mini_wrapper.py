"""
Custom LLM Wrapper for gpt-5.4-mini compatibility with RAGAS

RAGAS uses langchain's ChatOpenAI which sends 'max_tokens' parameter,
but gpt-5.4-mini requires 'max_completion_tokens' instead.

This wrapper intercepts the request payload and renames the parameter.
"""

from langchain_openai import ChatOpenAI
from typing import Any, Dict, Optional, List


class Gpt54MiniLLM(ChatOpenAI):
    """
    Custom ChatOpenAI wrapper for gpt-5.4-mini compatibility.
    
    Overrides _get_request_payload to rename 'max_tokens' to 'max_completion_tokens'
    which is required by the gpt-5.4-mini model.
    
    Usage:
        from evals.gpt54_mini_wrapper import Gpt54MiniLLM
        
        llm = Gpt54MiniLLM(model="gpt-5.4-mini", temperature=0)
        # Use with RAGAS or any langchain application
    """
    
    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Override to rename max_tokens → max_completion_tokens for gpt-5.4-mini.
        
        Args:
            input_: The input to send to the model
            stop: Optional stop sequences
            **kwargs: Additional parameters
            
        Returns:
            Modified payload with correct parameter name
        """
        # Get the standard payload from parent class
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        
        # Rename the parameter for gpt-5.4-mini compatibility
        if "max_tokens" in payload:
            payload["max_completion_tokens"] = payload.pop("max_tokens")
        
        return payload

# Made with Bob
