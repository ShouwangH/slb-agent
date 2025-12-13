"""
FastAPI Application for SLB Agent.

This module provides the HTTP API layer as defined in DESIGN.md Section 10.
It is a thin layer that delegates all business logic to the orchestrator.

Endpoints:
    POST /program - Run a funding program (legacy)
    POST /api/runs - Create and execute a program run
    GET /api/runs/{run_id} - Get a single run by ID
    GET /api/runs - List runs
"""

import logging
import os
from pathlib import Path

# Load .env file if it exists
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"[ENV] Loaded .env from {env_path}")
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, status

from app.config import DEFAULT_ENGINE_CONFIG
from app.exceptions import InfrastructureError
from app.llm.interface import LLMClient
from app.llm.mock import MockLLMClient
from app.llm.openai_client import OpenAILLMClient
from app.models import ErrorResponse, ProgramRequest, ProgramResponse
from app.orchestrator import run_program
from app.run_store import RunRecord, run_store
from app.validation import ValidationError


logger = logging.getLogger(__name__)


def get_llm_client() -> LLMClient:
    """
    Get the LLM client based on environment configuration.

    Returns OpenAILLMClient if OPENAI_API_KEY is set, otherwise MockLLMClient.
    """
    if os.environ.get("OPENAI_API_KEY"):
        print("[LLM] Using OpenAILLMClient (OPENAI_API_KEY is set)")
        return OpenAILLMClient()
    print("[LLM] Using MockLLMClient (no OPENAI_API_KEY)")
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


@app.get(
    "/health",
    summary="Health check",
    description="Returns service health status",
    tags=["System"],
)
def health_check() -> dict:
    """
    Health check endpoint.

    Returns basic service status. Can be extended with additional checks
    (database connectivity, LLM availability, etc.) in production.
    """
    return {
        "status": "healthy",
        "service": "SLB Agent API",
        "version": "1.0.0",
    }


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


# =============================================================================
# Runs API Endpoints
# =============================================================================


@app.post(
    "/api/runs",
    status_code=status.HTTP_201_CREATED,
    summary="Create and execute a program run",
    description="Creates a run record and executes the program. Returns 201 for all "
    "successfully created runs (even if engine result is INFEASIBLE).",
    tags=["Runs"],
)
def create_run(
    request: ProgramRequest,
    fund_id: Optional[str] = Query(None, description="Optional fund identifier"),
) -> dict:
    """
    Create and execute a program run.

    Returns 201 for all successfully created runs (even if engine result is INFEASIBLE).
    Returns 503 for infrastructure failures (LLM unreachable, etc.).

    Status Codes:
        201: Run created (check response.outcome.status for actual result)
        422: Pydantic validation error (malformed JSON)
        503: Infrastructure error (LLM unreachable, etc.)
    """
    run_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    logger.info(f"Creating run {run_id} for fund={fund_id}")

    try:
        llm = get_llm_client()
        response = run_program(request, llm, DEFAULT_ENGINE_CONFIG)

        run_store.create(
            RunRecord(
                run_id=run_id,
                fund_id=fund_id,
                program_description=request.program_description,
                response=response,
                error=None,
                created_at=created_at,
            )
        )

        logger.info(f"Run {run_id} completed: outcome.status={response.outcome.status}")

        # 201: Run created successfully (engine outcome may be OK, INFEASIBLE, or NUMERIC_ERROR)
        return {"run_id": run_id, "status": "completed", "response": response}

    except ValidationError as e:
        # Business validation failed - still a "created" run, but failed
        error_msg = str(e.errors)
        run_store.create(
            RunRecord(
                run_id=run_id,
                fund_id=fund_id,
                program_description=request.program_description,
                response=None,
                error=error_msg,
                created_at=created_at,
            )
        )
        logger.info(f"Run {run_id} failed (validation): {error_msg}")
        return {"run_id": run_id, "status": "failed", "error": error_msg}

    except ValueError as e:
        # Unsupported program type or other value errors - also a failed run
        error_msg = str(e)
        run_store.create(
            RunRecord(
                run_id=run_id,
                fund_id=fund_id,
                program_description=request.program_description,
                response=None,
                error=error_msg,
                created_at=created_at,
            )
        )
        logger.info(f"Run {run_id} failed (value error): {error_msg}")
        return {"run_id": run_id, "status": "failed", "error": error_msg}

    except InfrastructureError as e:
        # Infrastructure failure - don't create run record, return 503
        logger.error(f"Run {run_id} failed (infrastructure): {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "Infrastructure error",
                "detail": str(e),
                "code": "LLM_UNAVAILABLE",
            },
        )

    except Exception as e:
        # Unexpected error - treat as infrastructure failure
        logger.exception(f"Run {run_id} failed (unexpected)")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "Internal error",
                "detail": str(e),
                "code": "INTERNAL_ERROR",
            },
        )


@app.get(
    "/api/runs/{run_id}",
    summary="Get a single run by ID",
    description="Returns the full run record including audit trace.",
    tags=["Runs"],
)
def get_run(run_id: str) -> dict:
    """
    Get a single run by ID.

    Status Codes:
        200: Run found
        404: Run not found
    """
    record = run_store.get(run_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Run not found",
        )

    return {
        "run_id": record.run_id,
        "fund_id": record.fund_id,
        "program_description": record.program_description,
        "status": "completed" if record.response else "failed",
        "response": record.response,
        "error": record.error,
        "created_at": record.created_at,
        # Scenario metadata (null for single-scenario runs)
        "scenario_set_id": record.scenario_set_id,
        "scenario_kind": record.scenario_kind.value if record.scenario_kind else None,
        "scenario_label": record.scenario_label,
    }


@app.get(
    "/api/runs",
    summary="List runs",
    description="Lists runs, optionally filtered by fund_id.",
    tags=["Runs"],
)
def list_runs(
    fund_id: Optional[str] = Query(None, description="Filter by fund ID"),
    limit: int = Query(10, ge=1, le=100, description="Maximum runs to return"),
) -> list[dict]:
    """
    List runs, optionally filtered by fund_id.

    Returns a summary list (without full response payload).

    Status Codes:
        200: Success (may be empty list)
    """
    records = run_store.list_runs(fund_id=fund_id, limit=limit)

    return [
        {
            "run_id": r.run_id,
            "fund_id": r.fund_id,
            "program_description": r.program_description,
            "status": "completed" if r.response else "failed",
            "created_at": r.created_at,
            # Scenario metadata (null for single-scenario runs)
            "scenario_set_id": r.scenario_set_id,
            "scenario_kind": r.scenario_kind.value if r.scenario_kind else None,
            "scenario_label": r.scenario_label,
        }
        for r in records
    ]
