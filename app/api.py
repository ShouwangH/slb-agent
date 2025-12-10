"""
FastAPI Application for SLB Agent.

This module provides the HTTP API layer as defined in DESIGN.md Section 10.
It is a thin layer that delegates all business logic to the orchestrator.

Endpoints:
    POST /program - Run a funding program
"""

import os

from fastapi import FastAPI, HTTPException

from app.config import DEFAULT_ENGINE_CONFIG
from app.llm.interface import LLMClient
from app.llm.mock import MockLLMClient
from app.llm.openai_client import OpenAILLMClient
from app.models import ErrorResponse, ProgramRequest, ProgramResponse
from app.orchestrator import run_program
from app.validation import ValidationError


def get_llm_client() -> LLMClient:
    """
    Get the LLM client based on environment configuration.

    Returns OpenAILLMClient if OPENAI_API_KEY is set, otherwise MockLLMClient.
    """
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAILLMClient()
    return MockLLMClient()

# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="SLB Agent API",
    version="1.0.0",
    description="Real Estate Funding Workflow Agent (SLB Template)",
)


# =============================================================================
# Endpoints
# =============================================================================


@app.post(
    "/program",
    response_model=ProgramResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
)
def create_program(request: ProgramRequest) -> ProgramResponse:
    """
    Run a funding program.

    Takes a program request with assets, corporate state, and description.
    Returns the selector spec, outcome, and explanation.

    Status Codes:
        200: Success (includes infeasible outcomes)
        400: Invalid request (bad program_type, validation failure)
        422: Pydantic validation error (automatic)
        500: Internal error (LLM failure, numeric error)
    """
    # Get LLM client based on environment (OpenAI if key present, else mock)
    llm = get_llm_client()

    try:
        return run_program(
            request=request,
            llm=llm,
            config=DEFAULT_ENGINE_CONFIG,
        )
    except ValueError as e:
        # Unsupported program type or other value errors
        error_response = ErrorResponse(
            error="Invalid request",
            detail=str(e),
            code="UNSUPPORTED_PROGRAM_TYPE",
        )
        raise HTTPException(status_code=400, detail=error_response.model_dump())
    except ValidationError as e:
        # Validation failures from assets, corporate state, or spec
        error_response = ErrorResponse(
            error="Validation failed",
            detail=str(e.errors),
            code="VALIDATION_FAILED",
        )
        raise HTTPException(status_code=400, detail=error_response.model_dump())
    except Exception as e:
        # Catch-all for unexpected errors
        error_response = ErrorResponse(
            error="Internal error",
            detail=str(e),
            code="INTERNAL_ERROR",
        )
        raise HTTPException(status_code=500, detail=error_response.model_dump())
