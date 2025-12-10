"""
LLM module for SLB Agent.

Contains the LLM client interface and implementations for spec generation,
revision, and explanation summary.
"""

from app.llm.interface import LLMClient
from app.llm.mock import MockLLMClient

__all__ = [
    "LLMClient",
    "MockLLMClient",
]
