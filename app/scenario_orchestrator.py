"""
Multi-Scenario Orchestrator for SLB Agent.

This module coordinates the execution of multiple scenario variants from a single brief.
It builds on the existing orchestrator (run_program) by adding scenario generation
and per-scenario request building.

Key design decisions:
- BASE scenario runs deterministically (floor = target, no revision)
- Variant scenarios use the agentic loop (floor = target * 0.9)
- Assets and corporate_state are NEVER modified between scenarios
- Each scenario is a different "capital ask" on the same portfolio
"""

import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from app.config import DEFAULT_ENGINE_CONFIG, EngineConfig
from app.llm.interface import LLMClient
from app.models import (
    ProgramRequest,
    ProgramResponse,
    ScenarioDefinition,
    ScenarioKind,
    ScenarioSetSummary,
)
from app.orchestrator import run_program, summarize_assets
from app.run_store import RunRecord, run_store

logger = logging.getLogger(__name__)


def build_scenario_request(
    base_request: ProgramRequest,
    scenario: ScenarioDefinition,
) -> ProgramRequest:
    """
    Build a concrete ProgramRequest from base request + scenario definition.

    Args:
        base_request: The original ProgramRequest (assets, corporate_state)
        scenario: ScenarioDefinition with target and constraints

    Returns:
        New ProgramRequest ready for run_program()

    Invariants:
        - assets and corporate_state are NEVER modified
        - program_description gets scenario label appended
        - BASE scenario: floor = target (sacred, no revision)
        - Variant scenarios: floor = target * 0.9 (allow agentic exploration)
    """
    # BASE scenario: no revision flexibility (floor = target)
    # Variant scenarios: allow 10% reduction during revision
    if scenario.kind == ScenarioKind.BASE:
        floor = scenario.target_amount  # Sacred - no revision
    else:
        floor = scenario.target_amount * 0.9  # Allow agentic exploration

    # Build description with scenario label
    description = f"{base_request.program_description} [{scenario.label}]"

    return ProgramRequest(
        # Unchanged from base - NEVER modify these
        assets=base_request.assets,
        corporate_state=base_request.corporate_state,
        program_type=base_request.program_type,
        # Updated description
        program_description=description,
        # Scenario overrides
        floor_override=floor,
        max_leverage_override=scenario.max_leverage,  # None = use LLM inference
        min_coverage_override=scenario.min_coverage,  # None = use LLM inference
    )


def run_scenario_set(
    brief: str,
    base_request: ProgramRequest,
    llm: LLMClient,
    config: EngineConfig = DEFAULT_ENGINE_CONFIG,
    num_scenarios: int = 3,
    fund_id: Optional[str] = None,
) -> tuple[ScenarioSetSummary, list[dict]]:
    """
    Execute a complete multi-scenario run.

    Args:
        brief: Natural language program description
        base_request: Base ProgramRequest with assets and corporate state
        llm: LLM client
        config: Engine configuration
        num_scenarios: Target number of scenarios (1-5)
        fund_id: Optional fund identifier

    Returns:
        (ScenarioSetSummary, list of run result dicts)

    Flow:
        1. Generate scenario_set_id
        2. Generate scenario definitions via LLM
        3. For each scenario:
           a. Build request with build_scenario_request()
           b. Run program with run_program()
           c. Store RunRecord with scenario metadata
        4. Store ScenarioSetSummary
        5. Return (summary, run_results)

    Error handling:
        - Individual scenario failures don't stop the entire set
        - Failed scenarios are recorded with error, not response
        - LLM generation failure raises (fails entire set)
    """
    # Step 1: Generate scenario set ID
    scenario_set_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    logger.info(f"Starting scenario set {scenario_set_id} with {num_scenarios} scenarios")

    # Step 2: Generate scenario definitions via LLM
    asset_summary = summarize_assets(base_request.assets)
    scenarios = llm.generate_scenario_definitions(
        brief=brief,
        asset_summary=asset_summary,
        num_scenarios=num_scenarios,
    )

    logger.info(f"Generated {len(scenarios)} scenarios: {[s.label for s in scenarios]}")

    # Step 3: Execute each scenario
    run_ids: list[str] = []
    run_results: list[dict] = []

    for scenario in scenarios:
        run_id = str(uuid4())
        run_ids.append(run_id)

        logger.info(f"Running scenario '{scenario.label}' (kind={scenario.kind.value})")

        # Build the request for this scenario
        scenario_request = build_scenario_request(base_request, scenario)

        # Try to run the program
        response: Optional[ProgramResponse] = None
        error: Optional[str] = None

        try:
            response = run_program(
                request=scenario_request,
                llm=llm,
                config=config,
            )
            status = "completed"
        except Exception as e:
            logger.error(f"Scenario '{scenario.label}' failed: {e}")
            error = str(e)
            status = "failed"

        # Store RunRecord with scenario metadata
        record = RunRecord(
            run_id=run_id,
            fund_id=fund_id,
            program_description=scenario_request.program_description,
            response=response,
            error=error,
            created_at=created_at,
            # Scenario metadata
            scenario_set_id=scenario_set_id,
            scenario_kind=scenario.kind,
            scenario_label=scenario.label,
        )
        run_store.create(record)

        # Build result dict for API response
        result = {
            "run_id": run_id,
            "status": status,
            "scenario_set_id": scenario_set_id,
            "scenario_kind": scenario.kind.value,
            "scenario_label": scenario.label,
        }
        if response is not None:
            result["response"] = response
        if error is not None:
            result["error"] = error

        run_results.append(result)

    # Step 4: Store ScenarioSetSummary
    summary = ScenarioSetSummary(
        id=scenario_set_id,
        brief=brief,
        created_at=created_at,
        run_ids=run_ids,
    )
    run_store.create_scenario_set(summary)

    logger.info(f"Completed scenario set {scenario_set_id}: {len(run_results)} runs")

    # Step 5: Return results
    return summary, run_results
