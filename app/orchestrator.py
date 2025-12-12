"""
Orchestrator for SLB Agent.

This module contains the main entry point for the agentic loop as defined
in DESIGN.md Section 8.

The orchestrator is a thin coordination layer that delegates all business
logic to other modules:
- LLM calls: app.llm (generate_selector_spec, revise_selector_spec, etc.)
- Selection: app.engine.selector (select_assets)
- Policy: app.revision_policy (enforce_revision_policy)
- Explanations: app.engine.explanations (generate_explanation_nodes)
- Validation: app.validation (validate_spec, validate_assets, etc.)

The orchestrator's job is coordination only: call LLM, call engine, call policy, repeat.
"""

from collections import Counter
from datetime import datetime, timezone
from typing import Literal

from app.config import DEFAULT_ENGINE_CONFIG, EngineConfig
from app.engine.explanations import generate_explanation_nodes
from app.engine.selector import select_assets
from app.llm.interface import LLMClient
from app.models import (
    Asset,
    AuditTrace,
    AuditTraceEntry,
    Explanation,
    OutcomeSnapshot,
    PolicyViolation,
    ProgramOutcome,
    ProgramRequest,
    ProgramResponse,
    ProgramType,
    SelectionStatus,
    SelectorSpec,
    SpecSnapshot,
)
from app.revision_policy import enforce_revision_policy
from app.validation import (
    ValidationError,
    validate_assets,
    validate_corporate_state,
    validate_spec,
)


# =============================================================================
# Helper Functions
# =============================================================================


def summarize_assets(assets: list[Asset]) -> str:
    """
    Generate summary of assets for LLM context.

    Provides high-level statistics about the asset pool so the LLM
    can understand asset types and portfolio composition.

    Args:
        assets: List of assets in the portfolio

    Returns:
        Human-readable summary string for LLM prompt context
    """
    if not assets:
        return "No assets available."

    total_book_value = sum(a.book_value for a in assets)
    total_noi = sum(a.noi for a in assets)
    avg_criticality = sum(a.criticality for a in assets) / len(assets)
    avg_leaseability = sum(a.leaseability_score for a in assets) / len(assets)

    # Count asset types
    type_counts = Counter(a.asset_type.value for a in assets)
    type_summary = ", ".join(f"{count} {atype}" for atype, count in type_counts.items())

    # Count unique markets
    unique_markets = len(set(a.market for a in assets))

    return (
        f"{len(assets)} assets across {unique_markets} markets. "
        f"Types: {type_summary}. "
        f"Total book value: ${total_book_value:,.0f}. "
        f"Total NOI: ${total_noi:,.0f}. "
        f"Avg criticality: {avg_criticality:.2f}. "
        f"Avg leaseability: {avg_leaseability:.2f}."
    )


# =============================================================================
# Main Entry Point
# =============================================================================


def run_program(
    request: ProgramRequest,
    llm: LLMClient,
    config: EngineConfig = DEFAULT_ENGINE_CONFIG,
) -> ProgramResponse:
    """
    Main entry point for the agentic loop.

    Orchestrates spec generation, engine runs, revision loop, and
    explanation generation. This function is a thin coordinator that
    delegates all business logic to other modules.

    Args:
        request: The program request with assets, corporate state, and description
        llm: LLM client for spec generation and revision
        config: Engine configuration (defaults to DEFAULT_ENGINE_CONFIG)

    Returns:
        ProgramResponse with selector_spec, outcome, and explanation

    Raises:
        ValidationError: If input validation fails
        ValueError: If program_type is not supported

    Flow:
        1. Validate inputs
        2. Summarize assets for LLM context
        3. Generate initial SelectorSpec via LLM
        4. Validate spec
        5. Capture immutable hard constraints
        6. Agentic loop: run engine → check status → revise if needed
        7. Generate explanation
        8. Return response
    """
    # =========================================================================
    # Step 1: Validate inputs
    # =========================================================================

    # 1a. Validate program type (only SLB supported in v1)
    if request.program_type != ProgramType.SLB:
        raise ValueError(
            f"program_type '{request.program_type.value}' not supported in v1. "
            "Only 'slb' is supported."
        )

    # 1b. Validate assets
    asset_errors = validate_assets(request.assets)
    if asset_errors:
        raise ValidationError(asset_errors)

    # 1c. Validate corporate state
    state_errors = validate_corporate_state(request.corporate_state)
    if state_errors:
        raise ValidationError(state_errors)

    # =========================================================================
    # Step 2: Summarize assets for LLM context
    # =========================================================================

    asset_summary = summarize_assets(request.assets)

    # =========================================================================
    # Step 3: Generate initial SelectorSpec via LLM
    # =========================================================================

    initial_spec = llm.generate_selector_spec(
        program_type=request.program_type,
        program_description=request.program_description,
        asset_summary=asset_summary,
    )

    # =========================================================================
    # Step 3b: Apply explicit constraint overrides
    # =========================================================================
    #
    # Note: target_amount is NOT overridden here - the LLM-extracted target
    # from the natural language description is used directly. The user can
    # optionally specify a floor_override to allow flexibility.

    if request.max_leverage_override is not None:
        initial_spec.hard_constraints.max_net_leverage = request.max_leverage_override

    if request.min_coverage_override is not None:
        initial_spec.hard_constraints.min_fixed_charge_coverage = request.min_coverage_override

    # =========================================================================
    # Step 3c: Determine target source and floor fraction
    # =========================================================================
    #
    # New semantics:
    # - Target from NL description is ALWAYS sacred by default (floor = 100%)
    # - User can optionally specify floor_override to allow flexibility
    #
    # Examples:
    # - No floor_override: target is sacred, revision cannot reduce it
    # - floor_override = $8M with target = $10M: allows reduction to $8M

    target_source: Literal["user_override", "llm_extraction"] = "llm_extraction"
    floor_fraction: float

    if request.floor_override is not None:
        # User specified a custom floor - validate it's <= target
        if request.floor_override > initial_spec.target_amount:
            from fastapi import HTTPException

            raise HTTPException(
                status_code=400,
                detail=f"Floor override ({request.floor_override:,.0f}) cannot exceed target amount ({initial_spec.target_amount:,.0f})",
            )
        floor_fraction = request.floor_override / initial_spec.target_amount
    else:
        # No floor override - target is sacred (cannot reduce)
        floor_fraction = 1.0

    # =========================================================================
    # Step 4: Validate spec
    # =========================================================================

    spec_errors = validate_spec(initial_spec)
    if spec_errors:
        raise ValidationError(spec_errors)

    # =========================================================================
    # Step 5: Capture immutable hard constraints
    # =========================================================================
    #
    # IMPORTANT: original_target is now anchored on the clamped value
    # (i.e., the override if present, or LLM output if not).
    # The revision policy uses this as the floor reference.

    immutable_hard = initial_spec.hard_constraints.model_copy(deep=True)
    original_target = initial_spec.target_amount

    # Defensive assertion: original_target must be positive
    assert original_target > 0, "original_target must be positive after clamping"

    # =========================================================================
    # Step 5b: Initialize audit trace
    # =========================================================================

    started_at = datetime.now(timezone.utc).isoformat()
    floor_target = original_target * floor_fraction

    audit_trace = AuditTrace(
        entries=[],
        original_target=original_target,
        floor_target=floor_target,
        floor_fraction=floor_fraction,
        target_source=target_source,
        started_at=started_at,
        completed_at=None,
    )

    # =========================================================================
    # Step 6: Agentic loop
    # =========================================================================

    current_spec = initial_spec
    outcome: ProgramOutcome
    policy_violations: list[PolicyViolation] = []  # Track violations for current iteration

    for attempt in range(current_spec.max_iterations):
        # 6a. Run engine
        outcome = select_assets(
            assets=request.assets,
            corporate_state=request.corporate_state,
            spec=current_spec,
            config=config,
        )

        # 6b. Record audit trace entry
        # Determine target_before: None for initial iteration, previous target otherwise
        target_before = None if attempt == 0 else audit_trace.entries[-1].target_after

        entry = AuditTraceEntry(
            iteration=attempt,
            phase="initial" if attempt == 0 else "revision",
            spec_snapshot=SpecSnapshot.from_spec(current_spec),
            outcome_snapshot=OutcomeSnapshot.from_outcome(outcome),
            policy_violations=policy_violations,  # From previous iteration's revision
            target_before=target_before,
            target_after=current_spec.target_amount,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        audit_trace.entries.append(entry)

        # Reset policy violations for next iteration
        policy_violations = []

        # 6c. Check result
        if outcome.status == SelectionStatus.OK:
            # Success - exit loop
            break

        if outcome.status == SelectionStatus.NUMERIC_ERROR:
            # Numeric error - cannot recover, exit loop
            break

        # 6d. INFEASIBLE: attempt revision if iterations remain
        if attempt < current_spec.max_iterations - 1:
            # Call LLM to revise spec
            revised_spec = llm.revise_selector_spec(
                original_description=request.program_description,
                previous_spec=current_spec,
                outcome=outcome,
            )

            # Enforce revision policy with floor_fraction
            policy_result = enforce_revision_policy(
                immutable_hard=immutable_hard,
                original_target=original_target,
                prev_spec=current_spec,
                new_spec=revised_spec,
                floor_fraction=floor_fraction,
            )

            # Capture policy violations for next iteration's entry
            policy_violations = policy_result.violations

            if not policy_result.valid:
                # Policy violation - cannot revise further, exit loop
                # Return with last infeasible outcome
                break

            # Update spec for next iteration
            # policy_result.spec is guaranteed non-None when valid=True
            current_spec = policy_result.spec  # type: ignore[assignment]

    # =========================================================================
    # Step 7: Generate explanation
    # =========================================================================

    # 7a. Generate structured explanation nodes (engine)
    explanation_nodes = generate_explanation_nodes(
        spec=current_spec,
        outcome=outcome,
        state=request.corporate_state,
        config=config,
    )

    # 7b. Generate summary via LLM
    summary = llm.generate_explanation_summary(explanation_nodes)

    # 7c. Build Explanation object
    explanation = Explanation(
        summary=summary,
        nodes=explanation_nodes,
    )

    # =========================================================================
    # Step 8: Complete audit trace and return response
    # =========================================================================

    # 8a. Complete audit trace
    audit_trace.completed_at = datetime.now(timezone.utc).isoformat()

    # 8b. Return response
    return ProgramResponse(
        selector_spec=current_spec,
        outcome=outcome,
        explanation=explanation,
        audit_trace=audit_trace,
    )
