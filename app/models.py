"""
Pydantic models for SLB Agent.

This module contains all data models as defined in DESIGN.md Section 4.
Models handle validation and serialization only - no business logic.
"""

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Enums (Section 4.1)
# =============================================================================


class AssetType(str, Enum):
    """Property classification for real estate assets."""

    STORE = "store"
    DISTRIBUTION_CENTER = "distribution_center"
    OFFICE = "office"
    MIXED_USE = "mixed_use"
    OTHER = "other"


class MarketTier(int, Enum):
    """Market classification by size/liquidity."""

    TIER_1 = 1  # Primary markets (NYC, LA, Chicago)
    TIER_2 = 2  # Secondary markets (Austin, Nashville)
    TIER_3 = 3  # Tertiary markets


class ProgramType(str, Enum):
    """Funding program type."""

    SLB = "slb"
    # Future: REVOLVER = "revolver", CMBS = "cmbs", etc.


class Objective(str, Enum):
    """Optimization objective for asset selection."""

    MAXIMIZE_PROCEEDS = "maximize_proceeds"
    MINIMIZE_RISK = "minimize_risk"
    BALANCED = "balanced"


class SelectionStatus(str, Enum):
    """Outcome status from the selection engine."""

    OK = "ok"
    INFEASIBLE = "infeasible"
    NUMERIC_ERROR = "numeric_error"


class PolicyViolationCode(str, Enum):
    """
    Codes for revision policy violations.

    These represent deterministic bounds enforced by the revision policy,
    not LLM decisions. Used for audit trace visibility.
    """

    # Target violations
    TARGET_INCREASED = "target_increased"
    TARGET_DROP_EXCEEDED = "target_drop_exceeded"
    TARGET_BELOW_FLOOR = "target_below_floor"

    # Hard constraint violations (attempts to relax immutable constraints)
    LEVERAGE_RELAXED = "leverage_relaxed"
    INTEREST_COVERAGE_RELAXED = "interest_coverage_relaxed"
    FIXED_CHARGE_COVERAGE_RELAXED = "fixed_charge_coverage_relaxed"
    CRITICAL_FRACTION_RELAXED = "critical_fraction_relaxed"
    CONSTRAINT_DELETED = "constraint_deleted"

    # Filter violations (bounded relaxation exceeded)
    CRITICALITY_STEP_EXCEEDED = "criticality_step_exceeded"
    LEASEABILITY_STEP_EXCEEDED = "leaseability_step_exceeded"
    FILTER_DELETED = "filter_deleted"

    # Immutable field violations
    PROGRAM_TYPE_CHANGED = "program_type_changed"


# =============================================================================
# Input Models (Section 4.2-4.3)
# =============================================================================


class Asset(BaseModel):
    """
    A single real estate property in the portfolio.

    V1 Active Fields: asset_id, asset_type, market, market_tier, noi,
    book_value, criticality, leaseability_score
    """

    # Required fields (V1 active)
    asset_id: str = Field(..., description="Unique identifier")
    asset_type: AssetType = Field(..., description="Property classification")
    market: str = Field(..., description="Location (e.g., 'Dallas, TX')")
    noi: float = Field(..., gt=0, description="Annual Net Operating Income ($)")
    book_value: float = Field(..., gt=0, description="Accounting book value ($)")
    criticality: float = Field(
        ..., ge=0, le=1, description="Mission-criticality score [0, 1]"
    )
    leaseability_score: float = Field(
        ..., ge=0, le=1, description="Re-lease/repurpose ease [0, 1]"
    )

    # Optional fields
    name: Optional[str] = Field(None, description="Human-readable name")
    country: Optional[str] = Field(None, description="Country code")
    market_tier: Optional[MarketTier] = Field(None, description="Market classification")
    tenant_name: Optional[str] = Field(None, description="Primary tenant")
    tenant_credit_score: Optional[float] = Field(
        None, ge=0, le=1, description="Tenant credit [0, 1]"
    )
    wault_years: Optional[float] = Field(
        None, ge=0, description="Weighted avg unexpired lease term"
    )
    demographic_index: Optional[float] = Field(
        None, description="Future: location demographics"
    )
    esg_risk_score: Optional[float] = Field(None, description="Future: ESG risk")
    current_ltv: Optional[float] = Field(
        None, ge=0, le=1, description="Future: existing leverage"
    )
    existing_debt_amount: Optional[float] = Field(
        None, ge=0, description="Future: encumbered debt"
    )
    encumbrance_type: Optional[str] = Field(
        None, description="Future: mortgage, ground lease"
    )


class CorporateState(BaseModel):
    """
    Pre-transaction corporate financial position.

    Validation allows ebitda to be negative (distressed company),
    but downstream metrics will be None/undefined in that case.
    """

    net_debt: float = Field(..., ge=0, description="Total debt minus cash ($)")
    ebitda: float = Field(..., description="Trailing 12-month EBITDA ($)")
    interest_expense: float = Field(..., ge=0, description="Annual interest expense ($)")
    lease_expense: Optional[float] = Field(
        None, ge=0, description="Pre-SLB operating lease expense (default: 0)"
    )


# =============================================================================
# Spec Models (Section 4.4)
# =============================================================================


class HardConstraints(BaseModel):
    """
    Immutable constraints that the engine must satisfy.
    Once set from initial spec, the LLM cannot relax these.
    """

    max_net_leverage: Optional[float] = Field(
        None, gt=0, lt=10, description="net_debt_after / ebitda <= max_net_leverage"
    )
    min_interest_coverage: Optional[float] = Field(
        None, gt=0, lt=50, description="ebitda / interest_after >= min_interest_coverage"
    )
    min_fixed_charge_coverage: Optional[float] = Field(
        None,
        gt=0,
        lt=20,
        description="ebitda / (interest_after + lease_expense) >= min_fixed_charge_coverage",
    )
    max_critical_fraction: Optional[float] = Field(
        None,
        gt=0,
        le=1,
        description="Critical NOI fraction limit",
    )
    # V2: max_asset_type_share: Optional[Dict[AssetType, float]] = None


class SoftPreferences(BaseModel):
    """
    Adjustable preferences that influence scoring.
    LLM may relax these during revision.
    """

    prefer_low_criticality: bool = Field(
        True, description="Favor non-critical assets in scoring"
    )
    prefer_high_leaseability: bool = Field(
        True, description="Favor easily re-leased assets in scoring"
    )
    weight_criticality: float = Field(
        1.0, ge=0, description="Weight for criticality in scoring function"
    )
    weight_leaseability: float = Field(
        1.0, ge=0, description="Weight for leaseability in scoring function"
    )


class AssetFilters(BaseModel):
    """
    Pre-selection filters applied before scoring.
    LLM may relax these during revision.
    """

    include_asset_types: Optional[list[AssetType]] = Field(
        None, description="Whitelist (None = all)"
    )
    exclude_asset_types: Optional[list[AssetType]] = Field(None, description="Blacklist")
    exclude_markets: Optional[list[str]] = Field(None, description="Markets to exclude")
    min_leaseability_score: Optional[float] = Field(
        None, ge=0, le=1, description="Floor for eligibility"
    )
    max_criticality: Optional[float] = Field(
        None, ge=0, le=1, description="Ceiling for eligibility"
    )


class SelectorSpec(BaseModel):
    """
    Structured representation of program parameters.
    Generated/revised by LLM.
    """

    program_type: ProgramType
    objective: Objective
    target_amount: float = Field(..., gt=0, description="Target SLB proceeds in dollars")
    hard_constraints: HardConstraints
    soft_preferences: SoftPreferences
    asset_filters: AssetFilters
    time_horizon_years: Optional[int] = Field(None, gt=0)
    max_iterations: int = Field(3, ge=1, le=10, description="Bounds the agentic revision loop")


# =============================================================================
# Internal Metrics Models (for engine computation)
# =============================================================================


class BaselineMetrics(BaseModel):
    """
    Pre-transaction corporate metrics computed from CorporateState.

    Coverage metrics are Optional because they become undefined when
    the denominator is near zero (e.g., zero interest expense).
    """

    leverage: Optional[float] = Field(
        None, description="net_debt / ebitda (None if ebitda ≈ 0)"
    )
    interest_coverage: Optional[float] = Field(
        None, description="ebitda / interest_expense (None if interest ≈ 0)"
    )
    fixed_charge_coverage: Optional[float] = Field(
        None, description="ebitda / (interest + lease_expense) (None if denominator ≈ 0)"
    )


class PostMetrics(BaseModel):
    """
    Post-transaction corporate metrics after SLB execution.

    Computed from CorporateState + selected assets + config.
    """

    net_debt: float = Field(..., ge=0, description="Net debt after proceeds applied")
    interest_expense: float = Field(..., ge=0, description="Interest after debt paydown")
    total_lease_expense: float = Field(..., ge=0, description="Original lease + SLB rent")
    leverage: Optional[float] = Field(
        None, description="net_debt_after / ebitda (None if ebitda ≈ 0)"
    )
    interest_coverage: Optional[float] = Field(
        None, description="ebitda / interest_after (None if interest ≈ 0)"
    )
    fixed_charge_coverage: Optional[float] = Field(
        None, description="ebitda / (interest + leases) (None if denominator ≈ 0)"
    )


class PortfolioMetrics(BaseModel):
    """
    Combined before/after metrics for constraint checking and reporting.

    This is the primary metrics container passed between engine functions.
    """

    # Pre-transaction
    baseline: BaselineMetrics

    # Post-transaction
    post: PostMetrics

    # Selection-specific
    total_proceeds: float = Field(..., ge=0)
    total_slb_rent: float = Field(..., ge=0)
    critical_fraction: float = Field(..., ge=0, le=1)


# =============================================================================
# Output Models (Section 4.5)
# =============================================================================


class ConstraintViolation(BaseModel):
    """A single constraint violation with details."""

    code: str = Field(..., description="Constraint identifier (e.g., 'MAX_NET_LEVERAGE')")
    detail: str = Field(..., description="Human-readable explanation")
    actual: float = Field(..., description="Computed value")
    limit: float = Field(..., description="Constraint threshold")


class PolicyViolation(BaseModel):
    """
    Structured policy violation from the revision policy.

    Replaces unstructured strings for better UI rendering and type safety.
    These represent deterministic bounds enforced by policy, not LLM decisions.
    """

    code: PolicyViolationCode = Field(..., description="Violation type code")
    detail: str = Field(..., description="Human-readable explanation")
    field: str = Field(..., description="Which field was violated")
    attempted: Optional[float] = Field(None, description="What LLM tried to set")
    limit: Optional[float] = Field(None, description="The bound that was enforced")
    adjusted_to: Optional[float] = Field(
        None, description="What we clamped it to (None if invalid/rejected)"
    )


class AssetSLBMetrics(BaseModel):
    """Per-asset SLB economics computed by the engine."""

    market_value: float = Field(..., ge=0)
    proceeds: float = Field(..., ge=0)
    slb_rent: float = Field(..., ge=0)
    cap_rate: float = Field(..., gt=0, lt=1)


class AssetSelection(BaseModel):
    """A selected asset with its SLB economics."""

    asset: Asset
    proceeds: float = Field(..., ge=0, description="SLB proceeds from this asset")
    slb_rent: float = Field(..., ge=0, description="Annual leaseback rent obligation")


class ProgramOutcome(BaseModel):
    """
    Full outcome from the selection engine (Section 4.5.3).

    Contains selection results, before/after metrics, and any violations.
    Coverage metrics are Optional because they become undefined when
    the denominator is near zero.
    """

    status: SelectionStatus = Field(..., description="ok / infeasible / numeric_error")
    selected_assets: list[AssetSelection] = Field(
        default_factory=list, description="Assets in the SLB pool"
    )
    proceeds: float = Field(..., ge=0, description="Total SLB proceeds")

    # Pre-transaction metrics
    leverage_before: Optional[float] = Field(None, description="Pre-transaction net leverage")
    interest_coverage_before: Optional[float] = Field(
        None, description="Pre-transaction interest coverage (EBITDA / Interest)"
    )
    fixed_charge_coverage_before: Optional[float] = Field(
        None, description="Pre-transaction fixed charge coverage (EBITDA / (Interest + Leases))"
    )

    # Post-transaction metrics
    leverage_after: Optional[float] = Field(None, description="Post-transaction net leverage")
    interest_coverage_after: Optional[float] = Field(
        None, description="Post-transaction interest coverage"
    )
    fixed_charge_coverage_after: Optional[float] = Field(
        None, description="Post-transaction fixed charge coverage"
    )

    # Selection metrics
    critical_fraction: float = Field(
        0.0, ge=0, le=1, description="Critical NOI / Total selected NOI"
    )

    # Issues
    violations: list[ConstraintViolation] = Field(
        default_factory=list, description="Constraint violations (empty if OK)"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Non-fatal warnings (e.g., 'surplus proceeds ignored')"
    )


# =============================================================================
# Explanation Models (Section 4.6)
# =============================================================================


class ExplanationNode(BaseModel):
    """
    A single explanation item with machine-readable metadata.
    The engine produces these; LLM/UI renders them to human text.
    """

    id: str = Field(..., description="Unique identifier for this node")
    label: str = Field(..., description="Human-readable label")
    severity: Literal["info", "warning", "error"] = Field(..., description="Severity level")
    category: Literal["constraint", "driver", "risk", "alternative"] = Field(
        ..., description="Node category"
    )
    metric: Optional[str] = Field(
        None, description="Related metric name (e.g., 'fixed_charge_coverage')"
    )
    baseline_value: Optional[float] = Field(None, description="Pre-transaction value")
    post_value: Optional[float] = Field(None, description="Post-transaction value")
    threshold: Optional[float] = Field(None, description="Constraint threshold if applicable")
    asset_ids: Optional[list[str]] = Field(None, description="Related asset IDs")
    detail: Optional[str] = Field(None, description="Additional context")


class Explanation(BaseModel):
    """
    Structured explanation data returned by the engine.
    LLM generates summary from nodes.
    """

    summary: str = Field(..., description="2-3 sentence executive summary")
    nodes: list[ExplanationNode] = Field(
        default_factory=list, description="Structured explanation items"
    )


# =============================================================================
# Audit Trace Models (for orchestration visibility)
# =============================================================================


class SpecSnapshot(BaseModel):
    """
    Lightweight snapshot of spec fields relevant to audit trail.

    IMPORTANT: Use from_spec() factory to create. Never construct directly
    in orchestrator code - this ensures field alignment.
    """

    target_amount: float = Field(..., description="Target SLB proceeds")

    # Asset filters (soft, can be relaxed within bounds)
    max_criticality: Optional[float] = Field(None, description="Ceiling for eligibility")
    min_leaseability_score: Optional[float] = Field(None, description="Floor for eligibility")

    # Hard constraints (immutable after initial spec)
    max_net_leverage: Optional[float] = Field(None, description="Max net_debt / ebitda")
    min_interest_coverage: Optional[float] = Field(None, description="Min ebitda / interest")
    min_fixed_charge_coverage: Optional[float] = Field(
        None, description="Min ebitda / (interest + leases)"
    )
    max_critical_fraction: Optional[float] = Field(None, description="Critical NOI fraction limit")

    @classmethod
    def from_spec(cls, spec: "SelectorSpec") -> "SpecSnapshot":
        """
        Single derivation point from SelectorSpec.

        This is the ONLY way to create SpecSnapshot. Do not call __init__ directly.
        """
        return cls(
            target_amount=spec.target_amount,
            max_criticality=spec.asset_filters.max_criticality,
            min_leaseability_score=spec.asset_filters.min_leaseability_score,
            max_net_leverage=spec.hard_constraints.max_net_leverage,
            min_interest_coverage=spec.hard_constraints.min_interest_coverage,
            min_fixed_charge_coverage=spec.hard_constraints.min_fixed_charge_coverage,
            max_critical_fraction=spec.hard_constraints.max_critical_fraction,
        )


class OutcomeSnapshot(BaseModel):
    """
    Lightweight snapshot of outcome fields relevant to audit trail.

    IMPORTANT: Use from_outcome() factory to create. Never construct directly
    in orchestrator code - this ensures field alignment with ProgramOutcome.
    """

    status: SelectionStatus = Field(..., description="ok / infeasible / numeric_error")
    proceeds: float = Field(..., ge=0, description="Total SLB proceeds")
    leverage_after: Optional[float] = Field(None, description="Post-transaction net leverage")
    interest_coverage_after: Optional[float] = Field(
        None, description="Post-transaction interest coverage"
    )
    fixed_charge_coverage_after: Optional[float] = Field(
        None, description="Post-transaction fixed charge coverage"
    )
    critical_fraction: float = Field(0.0, ge=0, le=1, description="Critical NOI fraction")
    violations: list[ConstraintViolation] = Field(
        default_factory=list, description="Engine constraint violations"
    )

    @classmethod
    def from_outcome(cls, outcome: "ProgramOutcome") -> "OutcomeSnapshot":
        """
        Single derivation point from ProgramOutcome.

        This is the ONLY way to create OutcomeSnapshot. Do not call __init__ directly.
        """
        return cls(
            status=outcome.status,
            proceeds=outcome.proceeds,
            leverage_after=outcome.leverage_after,
            interest_coverage_after=outcome.interest_coverage_after,
            fixed_charge_coverage_after=outcome.fixed_charge_coverage_after,
            critical_fraction=outcome.critical_fraction,
            violations=outcome.violations,
        )


class AuditTraceEntry(BaseModel):
    """
    Single iteration in the orchestration audit trail.

    Records the spec used, engine outcome, and any policy violations
    for one iteration of the agentic loop.
    """

    iteration: int = Field(..., ge=0, description="0 = initial, 1+ = revisions")
    phase: Literal["initial", "revision"] = Field(..., description="Loop phase")

    # Spec state for this iteration (use SpecSnapshot.from_spec())
    spec_snapshot: SpecSnapshot = Field(..., description="Spec used in this iteration")

    # Engine result (use OutcomeSnapshot.from_outcome())
    outcome_snapshot: OutcomeSnapshot = Field(..., description="Engine result")

    # Policy enforcement - structured violations
    policy_violations: list[PolicyViolation] = Field(
        default_factory=list, description="Policy violations (deterministic bounds)"
    )

    # Target tracking
    target_before: Optional[float] = Field(
        None, description="Target before this iteration (None for initial)"
    )
    target_after: float = Field(..., description="Target used in this iteration")

    timestamp: str = Field(..., description="ISO format timestamp")


class AuditTrace(BaseModel):
    """
    Complete audit trail for an orchestration run.

    Tracks all iterations, numeric invariants, and determinism boundaries.
    """

    entries: list[AuditTraceEntry] = Field(
        default_factory=list, description="Iteration entries"
    )

    # Numeric invariants (captured once at start)
    original_target: float = Field(..., gt=0, description="Original target amount")
    floor_target: float = Field(..., ge=0, description="Minimum allowed target")
    floor_fraction: float = Field(
        ..., ge=0, le=1, description="1.0 for user_override, 0.75 for llm_extraction"
    )
    target_source: Literal["user_override", "llm_extraction"] = Field(
        ..., description="Where the target came from"
    )

    # Timing
    started_at: str = Field(..., description="ISO format start timestamp")
    completed_at: Optional[str] = Field(None, description="ISO format completion timestamp")


# =============================================================================
# API Models (Section 4.7)
# =============================================================================


class ProgramRequest(BaseModel):
    """Request body for POST /program endpoint."""

    assets: list[Asset] = Field(..., min_length=1)
    corporate_state: CorporateState
    program_type: ProgramType
    program_description: str = Field(..., min_length=1)

    # Explicit numeric overrides
    floor_override: Optional[float] = Field(
        None,
        gt=0,
        description="Minimum acceptable target amount. If not set, LLM-extracted target is sacred (floor = target).",
    )
    max_leverage_override: Optional[float] = Field(
        None, gt=0, lt=10, description="Explicit max net leverage constraint (overrides LLM inference)"
    )
    min_coverage_override: Optional[float] = Field(
        None,
        gt=0,
        lt=20,
        description="Explicit min fixed charge coverage constraint (overrides LLM inference)",
    )


class ProgramResponse(BaseModel):
    """Response body for POST /program endpoint (Section 4.7)."""

    selector_spec: SelectorSpec = Field(..., description="The spec used for selection")
    outcome: ProgramOutcome = Field(..., description="Selection results and metrics")
    explanation: Explanation = Field(..., description="Structured explanation with summary")
    audit_trace: Optional[AuditTrace] = Field(
        None, description="Audit trail of orchestration loop (None until PR3 integrates)"
    )


class ErrorResponse(BaseModel):
    """Error response body for API errors (Section 10.2)."""

    error: str = Field(..., description="Short error description")
    detail: Optional[str] = Field(None, description="Detailed error message")
    code: str = Field(..., description="Error code (e.g., 'UNSUPPORTED_PROGRAM_TYPE')")
