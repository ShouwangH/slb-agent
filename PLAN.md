# SLB Scenario Planner Implementation Plan

## Task Summary

Add visibility into the orchestration loop via an audit trace system, plus build a front-end scenario planner that visualizes the multi-step process, numeric invariants, and determinism boundaries.

**Scope:**
- Backend: Add audit trace to orchestrator (minimal change, ~40 lines)
- Backend: Add runs API for tracking multiple scenarios (in-memory)
- Frontend: React SPA for scenario planning with audit trace visualization
- Deployment: Docker Compose setup

**Out of Scope:**
- Database persistence (in-memory only)
- WebSockets/streaming (polling only)
- Seed data (cold-start capable)
- Changes to core numeric/selection logic

---

## Design Decisions

### D1: Override Semantics — "Sacred" User Overrides

**Decision:** User-provided overrides are immutable ("sacred"). LLM-extracted values can be revised within bounds.

| Source | `floor_fraction` | Behavior |
|--------|------------------|----------|
| `target_amount_override` provided | `1.0` | Cannot reduce at all |
| LLM extracts from description | `0.75` | Can reduce to 75% of original |

**Rationale:** If user explicitly says "$500M", don't give them $375M. If infeasible, return INFEASIBLE status.

**New field:** `target_source: Literal["user_override", "llm_extraction"]`

---

### D2: Hard Constraints Are Immutable

**Decision:** Hard constraints captured from initial spec can never be relaxed. Attempts to relax are clamped back with a policy violation recorded.

**Enforcement:** Already in `enforce_revision_policy()`, but add explicit tests.

---

### D3: `audit_trace` Is Always Present

**Decision:** `ProgramResponse.audit_trace` is NOT optional. The orchestrator always produces one.

```python
class ProgramResponse(BaseModel):
    audit_trace: AuditTrace  # NOT Optional
```

---

### D4: Single Source of Truth for Derived Models

**Decision:** Use factory methods to derive snapshots. Never manually copy fields.

```
PortfolioMetrics (engine internal)
       ↓ (engine builds)
ProgramOutcome (API response)
       ↓ (factory method)
OutcomeSnapshot (audit trace)
```

**Rule:** If you add a field to `ProgramOutcome`, the factory method enforces you update `OutcomeSnapshot`.

---

### D5: Structured Policy Violations

**Decision:** Replace `policy_violations: list[str]` with structured `PolicyViolation` type.

**Rationale:** Codifies determinism boundaries in types, not just docs. UI can render violations by code.

---

### D6: Full Field Alignment

**Decision:** All coverage metrics present in Python must be present in TypeScript.

Added to both: `interest_coverage_before`, `interest_coverage_after`

---

### D7: SpecSnapshot Captures All Hard Constraints

**Decision:** SpecSnapshot tracks ALL four hard constraints from HardConstraints, not a subset.

| HardConstraints Field | In SpecSnapshot | Rationale |
|-----------------------|-----------------|-----------|
| `max_net_leverage` | Yes | Primary covenant, always shown |
| `min_interest_coverage` | Yes | May be None if not enforced, but tracked |
| `min_fixed_charge_coverage` | Yes | Primary covenant, always shown |
| `max_critical_fraction` | Yes | Portfolio concentration limit, tracked |

**Rationale:** All hard constraints are immutable after initial spec. The audit UI should show when any of them were attempted to be relaxed (and clamped). Omitting some would hide policy enforcement actions.

**Consequence:** If a constraint is `None` in the spec, it appears as `null` in the snapshot. The UI can render this as "not enforced" rather than hiding the row.

---

### D8: HTTP Status Code Semantics

**Decision:** Distinguish between successful runs, engine failures, and infrastructure errors.

| Scenario | HTTP Status | Response Body |
|----------|-------------|---------------|
| Run created, engine succeeded | `201 Created` | `{ run_id, status: "completed", response }` |
| Run created, engine returned INFEASIBLE/NUMERIC_ERROR | `201 Created` | `{ run_id, status: "completed", response }` (outcome.status shows actual result) |
| Run created, but execution failed (e.g., validation error) | `201 Created` | `{ run_id, status: "failed", error }` |
| Infrastructure error (LLM unreachable, DB down) | `503 Service Unavailable` | `{ error, detail, code }` |
| Invalid request (bad JSON, missing fields) | `422 Unprocessable Entity` | FastAPI validation error |
| Run not found | `404 Not Found` | `{ detail: "Run not found" }` |

**Rationale:**
- `201` = "we created a run record" regardless of whether the engine found a feasible solution
- Engine's INFEASIBLE status is a valid business outcome, not an error
- Only infrastructure failures (can't reach LLM, can't parse response) are 5xx
- Keeps client logic simple: 2xx = check `response.outcome.status`, 5xx = retry/alert

---

## Part 1: Backend — Audit Trace System

### 1.1 New Models (`app/models.py`)

```python
# =============================================================================
# Policy Violation Types (deterministic bounds, codified in types)
# =============================================================================

class PolicyViolationCode(str, Enum):
    """Codes for revision policy violations - these are deterministic bounds."""

    # Target violations
    TARGET_INCREASED = "target_increased"
    TARGET_DROP_EXCEEDED = "target_drop_exceeded"
    TARGET_BELOW_FLOOR = "target_below_floor"

    # Hard constraint violations
    LEVERAGE_RELAXED = "leverage_relaxed"
    INTEREST_COVERAGE_RELAXED = "interest_coverage_relaxed"
    FIXED_CHARGE_COVERAGE_RELAXED = "fixed_charge_coverage_relaxed"
    CRITICAL_FRACTION_RELAXED = "critical_fraction_relaxed"
    CONSTRAINT_DELETED = "constraint_deleted"

    # Filter violations
    CRITICALITY_STEP_EXCEEDED = "criticality_step_exceeded"
    LEASEABILITY_STEP_EXCEEDED = "leaseability_step_exceeded"
    FILTER_DELETED = "filter_deleted"

    # Immutable field violations
    PROGRAM_TYPE_CHANGED = "program_type_changed"


class PolicyViolation(BaseModel):
    """
    Structured policy violation with deterministic bounds.
    Replaces unstructured strings for better UI rendering and type safety.
    """
    code: PolicyViolationCode
    detail: str
    field: str  # which field was violated
    attempted: Optional[float] = None  # what LLM tried
    limit: Optional[float] = None  # the bound that was enforced
    adjusted_to: Optional[float] = None  # what we clamped it to (None if invalid)


# =============================================================================
# Audit Trace Models (derived from existing types via factory methods)
# =============================================================================

class SpecSnapshot(BaseModel):
    """
    Lightweight snapshot of spec fields relevant to audit trail.

    IMPORTANT: Use from_spec() factory to create. Never construct directly
    in orchestrator code - this ensures field alignment.
    """
    target_amount: float

    # Asset filters (soft, can be relaxed within bounds)
    max_criticality: Optional[float] = None
    min_leaseability_score: Optional[float] = None

    # Hard constraints (immutable after initial spec)
    max_net_leverage: Optional[float] = None
    min_interest_coverage: Optional[float] = None
    min_fixed_charge_coverage: Optional[float] = None
    max_critical_fraction: Optional[float] = None

    @classmethod
    def from_spec(cls, spec: "SelectorSpec") -> "SpecSnapshot":
        """Single derivation point from SelectorSpec."""
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
    status: SelectionStatus
    proceeds: float
    leverage_after: Optional[float] = None
    interest_coverage_after: Optional[float] = None
    fixed_charge_coverage_after: Optional[float] = None
    critical_fraction: float = 0.0
    violations: list[ConstraintViolation] = Field(default_factory=list)

    @classmethod
    def from_outcome(cls, outcome: "ProgramOutcome") -> "OutcomeSnapshot":
        """Single derivation point from ProgramOutcome."""
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
    """
    iteration: int  # 0 = initial, 1+ = revisions
    phase: Literal["initial", "revision"]

    # Spec state for this iteration (use SpecSnapshot.from_spec())
    spec_snapshot: SpecSnapshot

    # Engine result (use OutcomeSnapshot.from_outcome())
    outcome_snapshot: OutcomeSnapshot

    # Policy enforcement - structured violations
    policy_violations: list[PolicyViolation] = Field(default_factory=list)

    # Target tracking
    target_before: Optional[float] = None
    target_after: float

    timestamp: str  # ISO format


class AuditTrace(BaseModel):
    """
    Complete audit trail for an orchestration run.
    """
    entries: list[AuditTraceEntry] = Field(default_factory=list)

    # Numeric invariants (captured once at start)
    original_target: float
    floor_target: float
    floor_fraction: float  # 1.0 for user_override, 0.75 for llm_extraction
    target_source: Literal["user_override", "llm_extraction"]

    # Timing
    started_at: str
    completed_at: Optional[str] = None
```

### 1.2 Update ProgramResponse (`app/models.py`)

```python
class ProgramResponse(BaseModel):
    """Response body for POST /program endpoint."""

    selector_spec: SelectorSpec
    outcome: ProgramOutcome
    explanation: Explanation
    audit_trace: AuditTrace  # REQUIRED - always present from orchestrator
```

### 1.3 Update Orchestrator (`app/orchestrator.py`)

```python
from datetime import datetime, timezone
from app.models import (
    AuditTrace, AuditTraceEntry, SpecSnapshot, OutcomeSnapshot,
    PolicyViolation,
)
from app.config import DEFAULT_REVISION_POLICY_CONFIG

def run_program(...) -> ProgramResponse:
    # ... existing validation code (steps 1-5) ...

    # After capturing original_target (line ~199):

    # Determine target source and floor behavior
    if request.target_amount_override is not None:
        target_source = "user_override"
        floor_fraction = 1.0  # Sacred - cannot reduce
    else:
        target_source = "llm_extraction"
        floor_fraction = DEFAULT_REVISION_POLICY_CONFIG.global_target_floor_fraction

    audit_trace = AuditTrace(
        original_target=original_target,
        floor_target=original_target * floor_fraction,
        floor_fraction=floor_fraction,
        target_source=target_source,
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    prev_target: Optional[float] = None
    current_spec = initial_spec
    outcome: ProgramOutcome

    for attempt in range(current_spec.max_iterations):
        # 6a. Run engine (existing)
        outcome = select_assets(
            assets=request.assets,
            corporate_state=request.corporate_state,
            spec=current_spec,
            config=config,
        )

        # Record this iteration using factory methods
        audit_trace.entries.append(AuditTraceEntry(
            iteration=attempt,
            phase="initial" if attempt == 0 else "revision",
            spec_snapshot=SpecSnapshot.from_spec(current_spec),
            outcome_snapshot=OutcomeSnapshot.from_outcome(outcome),
            target_before=prev_target,
            target_after=current_spec.target_amount,
            policy_violations=[],
            timestamp=datetime.now(timezone.utc).isoformat(),
        ))

        # 6b. Check result (existing)
        if outcome.status == SelectionStatus.OK:
            break
        if outcome.status == SelectionStatus.NUMERIC_ERROR:
            break

        # 6c. Revision (existing)
        if attempt < current_spec.max_iterations - 1:
            revised_spec = llm.revise_selector_spec(
                original_description=request.program_description,
                previous_spec=current_spec,
                outcome=outcome,
            )

            # Pass floor info to policy enforcement
            policy_result = enforce_revision_policy(
                immutable_hard=immutable_hard,
                original_target=original_target,
                floor_fraction=floor_fraction,  # NEW parameter
                prev_spec=current_spec,
                new_spec=revised_spec,
            )

            # Record structured policy violations
            audit_trace.entries[-1].policy_violations = policy_result.violations

            if not policy_result.valid:
                break

            prev_target = current_spec.target_amount
            current_spec = policy_result.spec

    audit_trace.completed_at = datetime.now(timezone.utc).isoformat()

    # ... existing explanation generation ...

    return ProgramResponse(
        selector_spec=current_spec,
        outcome=outcome,
        explanation=explanation,
        audit_trace=audit_trace,
    )
```

### 1.4 Update Revision Policy (`app/revision_policy.py`)

Update to return structured `PolicyViolation` objects:

```python
from app.models import PolicyViolation, PolicyViolationCode

@dataclass
class PolicyResult:
    valid: bool
    spec: Optional[SelectorSpec]
    violations: list[PolicyViolation]  # Changed from list[str]


def enforce_revision_policy(
    immutable_hard: HardConstraints,
    original_target: float,
    floor_fraction: float,  # NEW: 1.0 for sacred, 0.75 for LLM
    prev_spec: SelectorSpec,
    new_spec: SelectorSpec,
    config: RevisionPolicyConfig = DEFAULT_REVISION_POLICY_CONFIG,
) -> PolicyResult:
    violations: list[PolicyViolation] = []
    adjusted = new_spec.model_copy(deep=True)

    # Target amount: must decrease or stay same
    if new_spec.target_amount > prev_spec.target_amount:
        violations.append(PolicyViolation(
            code=PolicyViolationCode.TARGET_INCREASED,
            detail=f"target_amount cannot increase",
            field="target_amount",
            attempted=new_spec.target_amount,
            limit=prev_spec.target_amount,
            adjusted_to=prev_spec.target_amount,
        ))
        adjusted.target_amount = prev_spec.target_amount

    # Global floor (uses floor_fraction parameter)
    global_floor = original_target * floor_fraction
    if adjusted.target_amount < global_floor:
        violations.append(PolicyViolation(
            code=PolicyViolationCode.TARGET_BELOW_FLOOR,
            detail=f"target_amount below {floor_fraction:.0%} floor",
            field="target_amount",
            attempted=new_spec.target_amount,
            limit=global_floor,
            adjusted_to=None,  # Cannot adjust - invalid
        ))
        return PolicyResult(valid=False, spec=None, violations=violations)

    # ... rest of policy enforcement with structured violations ...
```

### 1.5 Runs API

**New file: `app/run_store.py`** (unchanged from before)

**Updated endpoints in `app/api.py`** with proper status codes:

```python
from uuid import uuid4
from datetime import datetime, timezone
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from app.run_store import run_store, RunRecord
from app.validation import ValidationError

class InfrastructureError(Exception):
    """Raised when external services (LLM, etc.) are unreachable."""
    pass

@app.post("/api/runs", status_code=status.HTTP_201_CREATED)
async def create_run(request: ProgramRequest, fund_id: Optional[str] = None):
    """
    Create and execute a program run.

    Returns 201 for all successfully created runs (even if engine result is INFEASIBLE).
    Returns 503 for infrastructure failures (LLM unreachable, etc.).
    """
    run_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    try:
        response = run_program(request, get_llm_client())

        run_store.create(RunRecord(
            run_id=run_id,
            fund_id=fund_id,
            program_description=request.program_description,
            response=response,
            error=None,
            created_at=created_at,
        ))

        # 201: Run created successfully (engine outcome may be OK, INFEASIBLE, or NUMERIC_ERROR)
        return {"run_id": run_id, "status": "completed", "response": response}

    except ValidationError as e:
        # Business validation failed - still a "created" run, but failed
        run_store.create(RunRecord(
            run_id=run_id,
            fund_id=fund_id,
            program_description=request.program_description,
            response=None,
            error=str(e),
            created_at=created_at,
        ))
        return {"run_id": run_id, "status": "failed", "error": str(e)}

    except InfrastructureError as e:
        # Infrastructure failure - don't create run record, return 503
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "Infrastructure error", "detail": str(e), "code": "LLM_UNAVAILABLE"},
        )

    except Exception as e:
        # Unexpected error - treat as infrastructure failure
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "Internal error", "detail": str(e), "code": "INTERNAL_ERROR"},
        )

@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    """Get a single run by ID."""
    record = run_store.get(run_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    return {
        "run_id": record.run_id,
        "fund_id": record.fund_id,
        "program_description": record.program_description,
        "response": record.response,
        "error": record.error,
        "created_at": record.created_at,
    }

@app.get("/api/runs")
async def list_runs(fund_id: Optional[str] = None, limit: int = 10):
    """List runs, optionally filtered by fund_id."""
    records = run_store.list_runs(fund_id=fund_id, limit=limit)
    return [
        {
            "run_id": r.run_id,
            "fund_id": r.fund_id,
            "program_description": r.program_description,
            "status": "completed" if r.response else "failed",
            "created_at": r.created_at,
        }
        for r in records
    ]
```

**LLM client should raise `InfrastructureError` on connection failures:**

```python
# In app/llm/openai_client.py

from app.api import InfrastructureError  # or define in a shared module

class OpenAILLMClient:
    def generate_selector_spec(self, ...):
        try:
            response = self.client.chat.completions.create(...)
            return self._parse_spec(response)
        except openai.APIConnectionError as e:
            raise InfrastructureError(f"Cannot reach OpenAI API: {e}")
        except openai.APIStatusError as e:
            if e.status_code >= 500:
                raise InfrastructureError(f"OpenAI API error: {e}")
            raise  # 4xx errors are likely our fault, re-raise
```

### 1.6 Backend Tests

#### `tests/test_model_alignment.py` (NEW - critical for preventing drift)

```python
"""
Tests to ensure derived models stay in sync with their sources.
These tests prevent field drift between ProgramOutcome, OutcomeSnapshot, etc.
"""
import pytest
from app.models import (
    SpecSnapshot, OutcomeSnapshot, SelectorSpec, ProgramOutcome,
    HardConstraints, SoftPreferences, AssetFilters, Objective, ProgramType,
    SelectionStatus, ConstraintViolation,
)


class TestOutcomeSnapshotAlignment:
    """Ensure OutcomeSnapshot stays in sync with ProgramOutcome."""

    @pytest.fixture
    def sample_outcome(self):
        return ProgramOutcome(
            status=SelectionStatus.OK,
            selected_assets=[],
            proceeds=1_000_000,
            leverage_before=3.0,
            leverage_after=2.5,
            interest_coverage_before=5.0,
            interest_coverage_after=6.0,
            fixed_charge_coverage_before=2.5,
            fixed_charge_coverage_after=3.0,
            critical_fraction=0.15,
            violations=[],
            warnings=[],
        )

    def test_from_outcome_copies_all_fields(self, sample_outcome):
        """Every OutcomeSnapshot field must come from ProgramOutcome."""
        snapshot = OutcomeSnapshot.from_outcome(sample_outcome)

        assert snapshot.status == sample_outcome.status
        assert snapshot.proceeds == sample_outcome.proceeds
        assert snapshot.leverage_after == sample_outcome.leverage_after
        assert snapshot.interest_coverage_after == sample_outcome.interest_coverage_after
        assert snapshot.fixed_charge_coverage_after == sample_outcome.fixed_charge_coverage_after
        assert snapshot.critical_fraction == sample_outcome.critical_fraction
        assert snapshot.violations == sample_outcome.violations

    def test_snapshot_fields_subset_of_outcome(self):
        """OutcomeSnapshot fields must exist in ProgramOutcome."""
        snapshot_fields = set(OutcomeSnapshot.model_fields.keys())
        outcome_fields = set(ProgramOutcome.model_fields.keys())

        for field in snapshot_fields:
            assert field in outcome_fields, (
                f"OutcomeSnapshot.{field} not found in ProgramOutcome - "
                f"add to ProgramOutcome or remove from snapshot"
            )


class TestSpecSnapshotAlignment:
    """Ensure SpecSnapshot stays in sync with SelectorSpec."""

    @pytest.fixture
    def sample_spec(self):
        return SelectorSpec(
            program_type=ProgramType.SLB,
            objective=Objective.BALANCED,
            target_amount=50_000_000,
            hard_constraints=HardConstraints(
                max_net_leverage=4.0,
                min_interest_coverage=2.0,
                min_fixed_charge_coverage=3.0,
                max_critical_fraction=0.3,
            ),
            soft_preferences=SoftPreferences(),
            asset_filters=AssetFilters(
                max_criticality=0.5,
                min_leaseability_score=0.6,
            ),
        )

    def test_from_spec_copies_all_fields(self, sample_spec):
        """Every SpecSnapshot field must come from SelectorSpec."""
        snapshot = SpecSnapshot.from_spec(sample_spec)

        assert snapshot.target_amount == sample_spec.target_amount
        assert snapshot.max_criticality == sample_spec.asset_filters.max_criticality
        assert snapshot.min_leaseability_score == sample_spec.asset_filters.min_leaseability_score
        assert snapshot.max_net_leverage == sample_spec.hard_constraints.max_net_leverage
        assert snapshot.min_interest_coverage == sample_spec.hard_constraints.min_interest_coverage
        assert snapshot.min_fixed_charge_coverage == sample_spec.hard_constraints.min_fixed_charge_coverage
        assert snapshot.max_critical_fraction == sample_spec.hard_constraints.max_critical_fraction


class TestCrossModelConsistency:
    """Integration tests for full derivation chain."""

    def test_audit_trace_last_entry_matches_response_outcome(self):
        """Last audit entry must match ProgramResponse.outcome."""
        from app.orchestrator import run_program
        from app.llm.mock import MockLLMClient
        from app.models import ProgramRequest, Asset, CorporateState, AssetType

        request = ProgramRequest(
            assets=[Asset(
                asset_id="A1",
                asset_type=AssetType.STORE,
                market="NYC",
                noi=5_000_000,
                book_value=50_000_000,
                criticality=0.3,
                leaseability_score=0.8,
            )],
            corporate_state=CorporateState(
                net_debt=100_000_000,
                ebitda=50_000_000,
                interest_expense=6_000_000,
            ),
            program_type=ProgramType.SLB,
            program_description="Raise $30M",
            target_amount_override=30_000_000,
        )

        response = run_program(request, MockLLMClient())

        last_entry = response.audit_trace.entries[-1]
        outcome = response.outcome

        # These must be equal - derived from same source
        assert last_entry.outcome_snapshot.status == outcome.status
        assert last_entry.outcome_snapshot.proceeds == outcome.proceeds
        assert last_entry.outcome_snapshot.leverage_after == outcome.leverage_after
        assert last_entry.outcome_snapshot.interest_coverage_after == outcome.interest_coverage_after
        assert last_entry.outcome_snapshot.violations == outcome.violations


class TestHardConstraintImmutability:
    """Ensure hard constraints cannot be relaxed."""

    def test_leverage_cannot_be_relaxed(self):
        """max_net_leverage cannot increase beyond original."""
        from app.revision_policy import enforce_revision_policy
        from app.models import HardConstraints

        immutable = HardConstraints(max_net_leverage=4.0)
        prev_spec = _make_spec(max_net_leverage=4.0)
        new_spec = _make_spec(max_net_leverage=5.0)  # attempted relaxation

        result = enforce_revision_policy(
            immutable_hard=immutable,
            original_target=100,
            floor_fraction=0.75,
            prev_spec=prev_spec,
            new_spec=new_spec,
        )

        assert result.spec.hard_constraints.max_net_leverage == 4.0
        assert any(v.code.value == "leverage_relaxed" for v in result.violations)

    def test_coverage_cannot_be_relaxed(self):
        """min_fixed_charge_coverage cannot decrease below original."""
        from app.revision_policy import enforce_revision_policy
        from app.models import HardConstraints

        immutable = HardConstraints(min_fixed_charge_coverage=3.0)
        prev_spec = _make_spec(min_fixed_charge_coverage=3.0)
        new_spec = _make_spec(min_fixed_charge_coverage=2.0)  # attempted relaxation

        result = enforce_revision_policy(
            immutable_hard=immutable,
            original_target=100,
            floor_fraction=0.75,
            prev_spec=prev_spec,
            new_spec=new_spec,
        )

        assert result.spec.hard_constraints.min_fixed_charge_coverage == 3.0
        assert any(v.code.value == "fixed_charge_coverage_relaxed" for v in result.violations)


class TestSacredOverrides:
    """Ensure user overrides are respected."""

    def test_user_override_floor_is_100_percent(self):
        """When target_amount_override provided, floor = target (no reduction allowed)."""
        from app.orchestrator import run_program
        from app.llm.mock import MockLLMClient
        from app.models import ProgramRequest, Asset, CorporateState, AssetType, ProgramType

        request = ProgramRequest(
            assets=[Asset(
                asset_id="A1",
                asset_type=AssetType.STORE,
                market="NYC",
                noi=5_000_000,
                book_value=50_000_000,
                criticality=0.3,
                leaseability_score=0.8,
            )],
            corporate_state=CorporateState(
                net_debt=100_000_000,
                ebitda=50_000_000,
                interest_expense=6_000_000,
            ),
            program_type=ProgramType.SLB,
            program_description="Raise money",
            target_amount_override=50_000_000,  # User explicitly says $50M
        )

        response = run_program(request, MockLLMClient())

        assert response.audit_trace.target_source == "user_override"
        assert response.audit_trace.floor_fraction == 1.0
        assert response.audit_trace.floor_target == response.audit_trace.original_target

    def test_llm_extraction_floor_is_75_percent(self):
        """When no override, floor = 75% of LLM-extracted target."""
        from app.orchestrator import run_program
        from app.llm.mock import MockLLMClient
        from app.models import ProgramRequest, Asset, CorporateState, AssetType, ProgramType

        request = ProgramRequest(
            assets=[Asset(
                asset_id="A1",
                asset_type=AssetType.STORE,
                market="NYC",
                noi=5_000_000,
                book_value=50_000_000,
                criticality=0.3,
                leaseability_score=0.8,
            )],
            corporate_state=CorporateState(
                net_debt=100_000_000,
                ebitda=50_000_000,
                interest_expense=6_000_000,
            ),
            program_type=ProgramType.SLB,
            program_description="Raise $50M via SLB",
            # No target_amount_override - LLM extracts from description
        )

        response = run_program(request, MockLLMClient())

        assert response.audit_trace.target_source == "llm_extraction"
        assert response.audit_trace.floor_fraction == 0.75
        assert response.audit_trace.floor_target == response.audit_trace.original_target * 0.75
```

---

## Part 2: Frontend — React Scenario Planner

### 2.1 Project Setup

```
frontend/
├── src/
│   ├── api/
│   │   └── runs.ts
│   ├── components/
│   │   ├── ScenarioForm.tsx
│   │   ├── RunList.tsx
│   │   ├── AuditTraceTimeline.tsx
│   │   ├── NumericInvariantsCard.tsx
│   │   ├── MetricsCard.tsx
│   │   └── AssetTable.tsx
│   ├── pages/
│   │   └── ScenarioPlannerPage.tsx
│   ├── types/
│   │   └── index.ts
│   ├── App.tsx
│   └── main.tsx
├── package.json
├── vite.config.ts
├── Dockerfile
└── nginx.conf
```

### 2.2 TypeScript Types (`frontend/src/types/index.ts`)

Types aligned with backend - includes all fields from Python models:

```typescript
// =============================================================================
// Enums (match backend exactly)
// =============================================================================

export type SelectionStatus = "ok" | "infeasible" | "numeric_error";
export type AssetType = "store" | "distribution_center" | "office" | "mixed_use" | "other";
export type TargetSource = "user_override" | "llm_extraction";

// =============================================================================
// Policy Violation (structured, matches PolicyViolation model)
// =============================================================================

export type PolicyViolationCode =
  | "target_increased"
  | "target_drop_exceeded"
  | "target_below_floor"
  | "leverage_relaxed"
  | "interest_coverage_relaxed"
  | "fixed_charge_coverage_relaxed"
  | "critical_fraction_relaxed"
  | "constraint_deleted"
  | "criticality_step_exceeded"
  | "leaseability_step_exceeded"
  | "filter_deleted"
  | "program_type_changed";

export interface PolicyViolation {
  code: PolicyViolationCode;
  detail: string;
  field: string;
  attempted: number | null;
  limit: number | null;
  adjusted_to: number | null;
}

// =============================================================================
// Constraint Violation (from engine, matches ConstraintViolation model)
// =============================================================================

export interface ConstraintViolation {
  code: string;
  detail: string;
  actual: number;
  limit: number;
}

// =============================================================================
// Audit Trace Types (match backend exactly)
// =============================================================================

export interface SpecSnapshot {
  target_amount: number;
  max_criticality: number | null;
  min_leaseability_score: number | null;
  max_net_leverage: number | null;
  min_interest_coverage: number | null;
  min_fixed_charge_coverage: number | null;
  max_critical_fraction: number | null;
}

export interface OutcomeSnapshot {
  status: SelectionStatus;
  proceeds: number;
  leverage_after: number | null;
  interest_coverage_after: number | null;  // ALIGNED with Python
  fixed_charge_coverage_after: number | null;
  critical_fraction: number;
  violations: ConstraintViolation[];
}

export interface AuditTraceEntry {
  iteration: number;
  phase: "initial" | "revision";
  spec_snapshot: SpecSnapshot;
  outcome_snapshot: OutcomeSnapshot;
  policy_violations: PolicyViolation[];  // Structured, not strings
  target_before: number | null;
  target_after: number;
  timestamp: string;
}

export interface AuditTrace {
  entries: AuditTraceEntry[];
  original_target: number;
  floor_target: number;
  floor_fraction: number;
  target_source: TargetSource;
  started_at: string;
  completed_at: string | null;
}

// =============================================================================
// Response Types (match backend exactly)
// =============================================================================

export interface AssetSelection {
  asset: Asset;
  proceeds: number;
  slb_rent: number;
}

export interface ProgramOutcome {
  status: SelectionStatus;
  selected_assets: AssetSelection[];
  proceeds: number;

  // Pre-transaction metrics (ALIGNED - all coverage fields present)
  leverage_before: number | null;
  interest_coverage_before: number | null;
  fixed_charge_coverage_before: number | null;

  // Post-transaction metrics
  leverage_after: number | null;
  interest_coverage_after: number | null;
  fixed_charge_coverage_after: number | null;

  critical_fraction: number;
  violations: ConstraintViolation[];
  warnings: string[];
}

export interface ProgramResponse {
  selector_spec: SelectorSpec;
  outcome: ProgramOutcome;
  explanation: Explanation;
  audit_trace: AuditTrace;  // NOT nullable - always present
}

// =============================================================================
// Type-level alignment check (compile error if drift)
// =============================================================================

// OutcomeSnapshot fields must be subset of ProgramOutcome fields
type AssertSubset<T, U> = keyof T extends keyof U ? true : never;
type _CheckOutcomeSnapshot = AssertSubset<
  Omit<OutcomeSnapshot, 'status' | 'violations'>,  // these have same name
  ProgramOutcome
>;

// =============================================================================
// Run Types
// =============================================================================

export interface RunRecord {
  run_id: string;
  fund_id: string | null;
  program_description: string;
  response: ProgramResponse | null;
  error: string | null;
  created_at: string;
}

export interface RunListItem {
  run_id: string;
  fund_id: string | null;
  program_description: string;
  status: "completed" | "failed";
  created_at: string;
}
```

### 2.3 Updated Components

#### AuditTraceTimeline.tsx (with structured violations)

```typescript
interface Props {
  auditTrace: AuditTrace;
}

export function AuditTraceTimeline({ auditTrace }: Props) {
  const { entries, original_target, floor_target, floor_fraction, target_source } = auditTrace;

  return (
    <div className="audit-timeline">
      <div className="invariants-header">
        <div className="target-info">
          <span>Original Target: ${original_target.toLocaleString()}</span>
          <span className={`source-badge ${target_source}`}>
            {target_source === 'user_override' ? 'User Override' : 'LLM Extracted'}
          </span>
        </div>
        <div className="floor-info">
          Floor ({(floor_fraction * 100).toFixed(0)}%): ${floor_target.toLocaleString()}
          {target_source === 'user_override' && (
            <span className="sacred-badge">Sacred</span>
          )}
        </div>
      </div>

      <div className="timeline">
        {entries.map((entry) => (
          <div key={entry.iteration} className={`entry ${entry.phase}`}>
            <div className="entry-header">
              <span className="iteration">Iteration {entry.iteration}</span>
              <span className={`status ${entry.outcome_snapshot.status}`}>
                {entry.outcome_snapshot.status}
              </span>
            </div>

            <div className="target-change">
              {entry.target_before !== null && (
                <span className="before">${entry.target_before.toLocaleString()}</span>
              )}
              <span className="arrow">→</span>
              <span className="after">${entry.target_after.toLocaleString()}</span>
            </div>

            <div className="metrics">
              <div>Proceeds: ${entry.outcome_snapshot.proceeds.toLocaleString()}</div>
              {entry.outcome_snapshot.leverage_after !== null && (
                <div>Leverage: {entry.outcome_snapshot.leverage_after.toFixed(2)}x</div>
              )}
              {entry.outcome_snapshot.interest_coverage_after !== null && (
                <div>Int Cov: {entry.outcome_snapshot.interest_coverage_after.toFixed(2)}x</div>
              )}
              {entry.outcome_snapshot.fixed_charge_coverage_after !== null && (
                <div>FCC: {entry.outcome_snapshot.fixed_charge_coverage_after.toFixed(2)}x</div>
              )}
            </div>

            {/* Engine constraint violations */}
            {entry.outcome_snapshot.violations.length > 0 && (
              <div className="violations engine-violations">
                <div className="label">Constraint violations:</div>
                {entry.outcome_snapshot.violations.map((v, i) => (
                  <div key={i} className="violation">
                    <code>{v.code}</code>: {v.detail}
                    <span className="values">({v.actual.toFixed(2)} vs limit {v.limit.toFixed(2)})</span>
                  </div>
                ))}
              </div>
            )}

            {/* Policy violations (structured) */}
            {entry.policy_violations.length > 0 && (
              <div className="violations policy-violations">
                <div className="label">Policy adjustments (deterministic):</div>
                {entry.policy_violations.map((v, i) => (
                  <div key={i} className={`violation ${v.adjusted_to !== null ? 'clamped' : 'rejected'}`}>
                    <code>{v.code}</code>: {v.field}
                    {v.attempted !== null && <span> tried {v.attempted.toLocaleString()}</span>}
                    {v.limit !== null && <span> (limit: {v.limit.toLocaleString()})</span>}
                    {v.adjusted_to !== null && <span> → clamped to {v.adjusted_to.toLocaleString()}</span>}
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
```

#### NumericInvariantsCard.tsx (with source indicator)

```typescript
interface Props {
  auditTrace: AuditTrace;
  finalProceeds: number;
}

export function NumericInvariantsCard({ auditTrace, finalProceeds }: Props) {
  const { original_target, floor_target, floor_fraction, target_source } = auditTrace;
  const lastEntry = auditTrace.entries[auditTrace.entries.length - 1];
  const currentTarget = lastEntry?.target_after ?? original_target;

  const isSacred = target_source === 'user_override';

  return (
    <div className="invariants-card">
      <h3>Numeric Invariants</h3>

      <div className="invariant">
        <label>
          Requested Target
          <span className={`source ${target_source}`}>
            {isSacred ? '(User Override - Sacred)' : '(LLM Extracted)'}
          </span>
        </label>
        <value>${original_target.toLocaleString()}</value>
      </div>

      <div className="invariant">
        <label>Current Target</label>
        <value>${currentTarget.toLocaleString()}</value>
        {currentTarget < original_target && (
          <delta className="negative">
            {(((currentTarget - original_target) / original_target) * 100).toFixed(1)}%
          </delta>
        )}
      </div>

      <div className="invariant">
        <label>
          Floor ({(floor_fraction * 100).toFixed(0)}% of original)
          {isSacred && <span className="locked">🔒</span>}
        </label>
        <value>${floor_target.toLocaleString()}</value>
        <note>{isSacred ? 'Cannot reduce - user override is sacred' : 'Hard limit'}</note>
      </div>

      <div className="invariant">
        <label>Actual Proceeds</label>
        <value>${finalProceeds.toLocaleString()}</value>
      </div>

      {currentTarget === floor_target && !isSacred && (
        <div className="warning">Floor reached - cannot reduce target further</div>
      )}
    </div>
  );
}
```

---

## Part 3: Deployment — Docker Compose

(Unchanged from previous version - see Dockerfiles and docker-compose.yml)

---

## Implementation Roadmap (PRs)

Each PR is independently reviewable and testable. PRs build on each other but can be merged incrementally.

---

### PR 1: Audit Trace Models & Alignment Tests
**Branch:** `feat/audit-trace-models`
**Reviewable independently:** Yes (no runtime changes)

**Changes:**
- `app/models.py`:
  - Add `PolicyViolationCode` enum
  - Add `PolicyViolation` model
  - Add `SpecSnapshot` model with `from_spec()` factory
  - Add `OutcomeSnapshot` model with `from_outcome()` factory
  - Add `AuditTraceEntry` model
  - Add `AuditTrace` model with `target_source` field
  - Update `ProgramResponse`: `audit_trace: Optional[AuditTrace] = None`
    - **NOTE:** Optional in PR1 to avoid breaking existing tests/callers
    - Will be made required (or "not None in practice") in PR3

- `tests/test_model_alignment.py` (new):
  - `TestOutcomeSnapshotAlignment`
  - `TestSpecSnapshotAlignment`
  - Field subset assertions
  - **Test that factories are the only construction path** (no raw `__init__`)

**Key constraint:** `SpecSnapshot` and `OutcomeSnapshot` constructors should be "private" by convention:
```python
class OutcomeSnapshot(BaseModel):
    # Fields are public, but direct construction is discouraged
    # Use from_outcome() factory to ensure alignment

    @classmethod
    def from_outcome(cls, outcome: "ProgramOutcome") -> "OutcomeSnapshot":
        """ONLY way to create OutcomeSnapshot. Do not call __init__ directly."""
        ...
```

**Acceptance:**
- [ ] All new models pass Pydantic validation
- [ ] Factory methods work correctly
- [ ] Alignment tests pass
- [ ] `ProgramResponse.audit_trace` is Optional (default None)
- [ ] Existing tests still pass (no runtime changes)

**Size:** ~200 lines

---

### PR 2: Structured Policy Violations
**Branch:** `feat/structured-policy-violations`
**Depends on:** PR 1
**Reviewable independently:** Yes

**Scope Control:**
- This PR ONLY changes the violations type and adds `floor_fraction`
- Does NOT change orchestrator calls yet (that's PR3)
- Existing callers continue to work with the default `floor_fraction`

**Changes:**
- `app/revision_policy.py`:
  - Change `PolicyResult.violations` from `list[str]` to `list[PolicyViolation]`
  - Add `floor_fraction` parameter with **default value** for backward compatibility:
    ```python
    def enforce_revision_policy(
        immutable_hard: HardConstraints,
        original_target: float,
        prev_spec: SelectorSpec,
        new_spec: SelectorSpec,
        floor_fraction: float = DEFAULT_REVISION_POLICY_CONFIG.global_target_floor_fraction,  # default = 0.75
        config: RevisionPolicyConfig = DEFAULT_REVISION_POLICY_CONFIG,
    ) -> PolicyResult:
    ```
  - Return structured `PolicyViolation` objects with codes

- `tests/test_revision_policy.py`:
  - Update existing tests to check `PolicyViolation` objects (not strings)
  - Add helper to extract violation codes: `violation_codes = [v.code for v in result.violations]`
  - Add tests for structured violation codes
  - **NOTE:** Hard constraint immutability tests added in PR3 (with orchestrator integration)

**Backward Compatibility:**
- Existing callers that don't pass `floor_fraction` get the default 0.75 behavior
- Callers checking `if result.violations:` still work (list truthiness)
- Callers iterating violations need minor updates (`.code`, `.detail` instead of string)

**Migration path for callers:**
```python
# Before (PR1)
if "cannot increase" in result.violations[0]:
    ...

# After (PR2)
if result.violations[0].code == PolicyViolationCode.TARGET_INCREASED:
    ...

# Or check by code:
if any(v.code == PolicyViolationCode.TARGET_INCREASED for v in result.violations):
    ...
```

**Acceptance:**
- [ ] `enforce_revision_policy()` returns `list[PolicyViolation]`
- [ ] All violation codes are from `PolicyViolationCode` enum
- [ ] Default `floor_fraction=0.75` maintains backward compatibility
- [ ] `floor_fraction=1.0` enforces sacred override behavior
- [ ] Existing revision policy tests pass (updated for new type)

**Size:** ~150 lines changed

---

### PR 3: Orchestrator Audit Trace Integration
**Branch:** `feat/orchestrator-audit-trace`
**Depends on:** PR 1, PR 2
**Reviewable independently:** Yes

**Changes:**
- `app/orchestrator.py`:
  - Determine `target_source` based on `request.target_amount_override`
  - Calculate `floor_fraction` (1.0 for override, 0.75 for LLM)
  - Build `AuditTrace` with entries per iteration
  - Use `SpecSnapshot.from_spec()` and `OutcomeSnapshot.from_outcome()`
  - Pass `floor_fraction` to `enforce_revision_policy()`
  - Include `audit_trace` in `ProgramResponse`

- `tests/test_audit_trace.py` (new):
  - `test_audit_trace_present_in_response`
  - `test_user_override_floor_is_100_percent`
  - `test_llm_extraction_floor_is_75_percent`
  - `test_audit_trace_last_entry_matches_outcome`

- `tests/test_hard_constraint_immutability.py` (new):
  - **Multi-iteration tests that verify constraints stay locked**
  ```python
  def test_leverage_stays_locked_across_iterations():
      """
      Run a scenario where LLM tries to relax leverage each iteration.
      Verify that after N iterations, leverage is still at original value.
      """
      # Use MockLLMClient that progressively tries to relax constraints
      mock_llm = MockLLMClientThatRelaxes(
          target_reductions=[0.95, 0.90, 0.85],  # Try reducing target each iteration
          leverage_attempts=[4.5, 5.0, 5.5],      # Try relaxing leverage each iteration
      )

      response = run_program(request, mock_llm)

      # Final spec should have original leverage, not any attempted relaxation
      assert response.selector_spec.hard_constraints.max_net_leverage == 4.0

      # All iterations should show policy violations for leverage attempts
      for entry in response.audit_trace.entries[1:]:  # Skip initial
          assert any(
              v.code == PolicyViolationCode.LEVERAGE_RELAXED
              for v in entry.policy_violations
          )

  def test_all_four_constraints_immutable():
      """Verify all 4 hard constraints cannot be relaxed."""
      # Test max_net_leverage, min_interest_coverage,
      # min_fixed_charge_coverage, max_critical_fraction
      ...
  ```

**Edge Case: `target_source` Determination**
```python
# Clear rule: target_source is based ONLY on whether override was provided
# NOT on whether LLM extraction succeeded or matched the override

if request.target_amount_override is not None:
    target_source = "user_override"
    floor_fraction = 1.0
    # Use the override value (already clamped in step 3b)
else:
    target_source = "llm_extraction"
    floor_fraction = 0.75
    # Use whatever LLM extracted (already in initial_spec.target_amount)
```

**Note:** If LLM extraction fails, that's an infrastructure error (PR4 handles via 503).
There's no partial state where we have a spec but don't know the target source.

**Acceptance:**
- [ ] `run_program()` returns `audit_trace` (not None)
- [ ] `target_source` is "user_override" when `target_amount_override` provided
- [ ] `target_source` is "llm_extraction" when no override provided
- [ ] `floor_fraction` is 1.0 for overrides, 0.75 otherwise
- [ ] Last entry's `outcome_snapshot` matches `response.outcome`
- [ ] Hard constraints are immutable across all iterations (new test)
- [ ] All existing orchestrator tests pass

**Size:** ~100 lines added to orchestrator, ~200 lines tests

---

### PR 4: Runs API & In-Memory Store
**Branch:** `feat/runs-api`
**Depends on:** PR 3
**Reviewable independently:** Yes

**Changes:**
- `app/run_store.py` (new):
  - `RunRecord` class
  - `RunStore` with thread-safe in-memory dict
  - `run_store` singleton

- `app/exceptions.py` (new):
  - `InfrastructureError` exception class
  - **Note:** This is a new module to avoid circular imports between api.py and llm/

- `app/api.py`:
  - `POST /api/runs` (201 Created)
  - `GET /api/runs/{run_id}` (200 / 404)
  - `GET /api/runs` (200, list)
  - Use `uuid.uuid4()` for run_id generation (not sequential)
  - Proper error handling (503 for infra errors)

- `app/llm/openai_client.py`:
  - Import `InfrastructureError` from `app.exceptions`
  - Catch connection errors → raise `InfrastructureError`

- `tests/test_runs_api.py` (new):
  - Test create/get/list endpoints
  - Test 201 vs 503 semantics
  - Test 404 for unknown run

**Error Classification:**

| Error Type | HTTP Status | Response Shape | When |
|------------|-------------|----------------|------|
| **Validation Error** | `201 Created` | `{ run_id, status: "failed", error: "..." }` | Invalid input that fails domain validation (e.g., empty assets, invalid corporate state) |
| **Engine INFEASIBLE** | `201 Created` | `{ run_id, status: "completed", response: { outcome: { status: "infeasible" } } }` | Valid input, but no feasible selection exists |
| **Engine NUMERIC_ERROR** | `201 Created` | `{ run_id, status: "completed", response: { outcome: { status: "numeric_error" } } }` | Valid input, but numeric computation failed |
| **Infrastructure Error** | `503 Service Unavailable` | `{ error, detail, code }` | LLM unreachable, parse failure, etc. |
| **Invalid JSON** | `422 Unprocessable Entity` | FastAPI validation | Malformed request body |

**Key distinction:**
- "Validation errors" are business rule violations (our `ValidationError` class)
- These are **not** the same as FastAPI 422 errors (schema violations)
- A run with `status: "failed"` is still a persisted record (we tried, it failed)
- Only infrastructure errors (503) don't persist a run record

**Run ID Generation:**
```python
from uuid import uuid4

run_id = str(uuid4())  # e.g., "550e8400-e29b-41d4-a716-446655440000"
```

**Exception Handling Pattern:**
```python
# app/exceptions.py
class InfrastructureError(Exception):
    """External service unreachable or failed unexpectedly."""
    pass

# app/api.py
from app.exceptions import InfrastructureError  # NOT from app.api

@app.post("/api/runs", status_code=status.HTTP_201_CREATED)
async def create_run(request: ProgramRequest, fund_id: Optional[str] = None):
    run_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    try:
        response = run_program(request, get_llm_client())
        # ... success path ...

    except ValidationError as e:
        # Domain validation failed - still create run record
        run_store.create(RunRecord(..., error=str(e)))
        return {"run_id": run_id, "status": "failed", "error": str(e)}

    except InfrastructureError as e:
        # Don't create run record - infrastructure failure
        raise HTTPException(status_code=503, detail={"error": "Infrastructure error", ...})

    except Exception as e:
        # Unexpected - log and treat as infrastructure failure
        logger.exception("Unexpected error in create_run")
        raise HTTPException(status_code=503, detail={"error": "Internal error", ...})
```

**Acceptance:**
- [ ] `POST /api/runs` returns 201 with `run_id` (uuid4 format)
- [ ] `GET /api/runs/{id}` returns full run with audit_trace
- [ ] `GET /api/runs` returns list (empty initially)
- [ ] `ValidationError` returns 201 with `status: "failed"`
- [ ] `InfrastructureError` returns 503 (no run record created)
- [ ] 404 for unknown run_id
- [ ] No circular imports between api.py and llm/

**Size:** ~250 lines

---

### PR 5: Frontend Setup & TypeScript Types
**Branch:** `feat/frontend-setup`
**Depends on:** PR 4 (for API contract)
**Reviewable independently:** Yes

**Changes:**
- `frontend/` directory:
  - Initialize Vite + React + TypeScript
  - `package.json`, `tsconfig.json`, `vite.config.ts`

- `frontend/src/types/index.ts`:
  - All TypeScript types matching backend exactly
  - `PolicyViolationCode`, `PolicyViolation`
  - `SpecSnapshot`, `OutcomeSnapshot`, `AuditTraceEntry`, `AuditTrace`
  - `ProgramOutcome`, `ProgramResponse`
  - `RunRecord`, `RunListItem`

- `frontend/src/api/runs.ts`:
  - `createRun()`, `getRun()`, `listRuns()`

- `frontend/src/App.tsx`:
  - Basic routing setup

- `frontend/src/index.css`:
  - Base styles, three-column layout

**Type Drift Prevention Strategy:**

1. **Compile-time subset check** (in `types/index.ts`):
   ```typescript
   // Compile error if OutcomeSnapshot has fields not in ProgramOutcome
   type AssertSubset<T, U> = keyof T extends keyof U ? true : never;
   type _CheckOutcomeSnapshot = AssertSubset<
     Omit<OutcomeSnapshot, 'status' | 'violations'>,
     ProgramOutcome
   >;
   ```

2. **Shared schema (future consideration):**
   - Could generate TS types from Python models using `pydantic-to-typescript`
   - For MVP, manual sync with compile-time checks is sufficient

3. **PR review checklist item:**
   - Every PR that touches `app/models.py` must update `frontend/src/types/index.ts`
   - Alignment tests in `test_model_alignment.py` catch Python-side drift

**Vite Dev Proxy Configuration:**

```typescript
// frontend/vite.config.ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',  // FastAPI backend
        changeOrigin: true,
      },
    },
  },
})
```

**Local Development Flow:**
```bash
# Terminal 1: Backend
cd /path/to/auquan-agent
uvicorn app.api:app --reload --port 8000

# Terminal 2: Frontend
cd frontend
npm run dev  # Runs on :5173, proxies /api to :8000
```

**Acceptance:**
- [ ] `npm run build` succeeds
- [ ] TypeScript types compile without errors
- [ ] Type subset assertion compiles (no drift)
- [ ] API client functions are typed correctly
- [ ] Basic app renders
- [ ] Dev proxy works (frontend :5173 → backend :8000)

**Size:** ~450 lines

---

### PR 6: Frontend Components
**Branch:** `feat/frontend-components`
**Depends on:** PR 5
**Reviewable independently:** Yes

**Changes:**
- `frontend/src/components/`:
  - `AuditTraceTimeline.tsx` - iteration timeline with violations
  - `NumericInvariantsCard.tsx` - target/floor/source display
  - `MetricsCard.tsx` - leverage/coverage metrics
  - `AssetTable.tsx` - selected assets table
  - `ScenarioForm.tsx` - input form
  - `RunList.tsx` - run list sidebar
  - `ErrorBoundary.tsx` - catch rendering errors
  - `LoadingSpinner.tsx` - loading state indicator

- `frontend/src/pages/`:
  - `ScenarioPlannerPage.tsx` - three-column layout assembly

- `frontend/src/tests/`:
  - Component smoke tests (optional for MVP)

**UX Edge Cases:**

1. **Loading States:**
   ```typescript
   // ScenarioPlannerPage.tsx
   const [isSubmitting, setIsSubmitting] = useState(false);
   const [isLoadingRun, setIsLoadingRun] = useState(false);

   // Show spinner during API calls
   {isSubmitting && <LoadingSpinner message="Creating run..." />}
   {isLoadingRun && <LoadingSpinner message="Loading run details..." />}
   ```

2. **Empty States:**
   ```typescript
   // RunList.tsx
   {runs.length === 0 && (
     <div className="empty-state">
       No runs yet. Create one using the form.
     </div>
   )}

   // AuditTraceTimeline.tsx
   {entries.length === 0 && (
     <div className="empty-state">
       No iterations recorded.
     </div>
   )}
   ```

3. **Error States:**
   ```typescript
   // ScenarioPlannerPage.tsx
   const [error, setError] = useState<string | null>(null);

   // API error handling
   try {
     const result = await createRun(request);
     if (result.status === "failed") {
       setError(`Run failed: ${result.error}`);
     }
   } catch (e) {
     if (e instanceof Response && e.status === 503) {
       setError("Service temporarily unavailable. Please try again.");
     } else {
       setError("An unexpected error occurred.");
     }
   }

   // Display error
   {error && <div className="error-banner">{error}</div>}
   ```

4. **Error Boundary:**
   ```typescript
   // ErrorBoundary.tsx
   class ErrorBoundary extends React.Component<Props, State> {
     state = { hasError: false, error: null };

     static getDerivedStateFromError(error: Error) {
       return { hasError: true, error };
     }

     render() {
       if (this.state.hasError) {
         return (
           <div className="error-fallback">
             <h2>Something went wrong</h2>
             <pre>{this.state.error?.message}</pre>
             <button onClick={() => window.location.reload()}>Reload</button>
           </div>
         );
       }
       return this.props.children;
     }
   }

   // Usage in App.tsx
   <ErrorBoundary>
     <ScenarioPlannerPage />
   </ErrorBoundary>
   ```

5. **Null/Optional Value Display:**
   ```typescript
   // MetricsCard.tsx
   function formatMetric(value: number | null, suffix: string = ""): string {
     if (value === null) return "N/A";
     return `${value.toFixed(2)}${suffix}`;
   }

   // Usage
   <div>Interest Coverage: {formatMetric(outcome.interest_coverage_after, "x")}</div>
   ```

6. **Stale Data Handling:**
   ```typescript
   // After creating a run, refresh the run list
   const handleSubmit = async () => {
     const result = await createRun(request);
     await refreshRunList();  // Don't rely on optimistic update
     setSelectedRunId(result.run_id);
   };
   ```

**Acceptance:**
- [ ] ScenarioPlannerPage renders three-column layout
- [ ] Form submits and creates run
- [ ] Run list shows created runs
- [ ] Selecting run shows audit trace timeline
- [ ] Numeric invariants display correctly
- [ ] `target_source` indicator shows "Sacred" vs "LLM Extracted"
- [ ] Loading spinner shown during API calls
- [ ] Empty states shown for no runs / no iterations
- [ ] Error banner shown for API failures
- [ ] Error boundary catches rendering errors
- [ ] Null metrics displayed as "N/A"

**Size:** ~700 lines

---

### PR 7: Docker & Deployment
**Branch:** `feat/docker-deployment`
**Depends on:** PR 4, PR 6
**Reviewable independently:** Yes

**Changes:**
- `Dockerfile` (backend):
  - Python 3.11-slim base
  - Install requirements, run uvicorn

- `frontend/Dockerfile`:
  - Multi-stage: node build → nginx serve

- `frontend/nginx.conf`:
  - `/api/` proxy to backend
  - SPA fallback

- `docker-compose.yml`:
  - `backend` service (port 3001)
  - `frontend` service (port 8080)

- `README.md` updates:
  - Docker setup instructions

**Acceptance:**
- [ ] `docker compose up --build` starts both services
- [ ] http://localhost:8080 loads frontend
- [ ] Frontend can create runs via API proxy
- [ ] Audit trace displays correctly end-to-end

**Size:** ~100 lines

---

## PR Dependency Graph

```
PR1 (models) ──┬──> PR2 (policy violations)
               │           │
               │           v
               └──> PR3 (orchestrator) ──> PR4 (runs API) ──> PR5 (frontend setup)
                                                    │                │
                                                    │                v
                                                    │         PR6 (components)
                                                    │                │
                                                    └────────────────┴──> PR7 (docker)
```

---

## Global Unknowns & Edge Cases

Issues that span multiple PRs and need consistent handling.

### LLM Mid-Loop Failure

**Scenario:** LLM is reachable for initial spec generation, but fails during a revision call (iteration 2+).

**Current Behavior:** The `run_program()` function would raise an exception mid-loop, leaving no audit trace recorded.

**Options:**

| Option | Behavior | Pros | Cons |
|--------|----------|------|------|
| A: Fail entire run | Raise `InfrastructureError`, return 503 | Simple, consistent | Loses partial progress |
| B: Return partial audit trace | Return last successful iteration's outcome with `status: "incomplete"` | Preserves work done | More complex, new status |
| C: Retry then fail | Retry LLM call once, then option A | More resilient | Adds latency |

**Recommendation:** Start with **Option A** (fail entire run). Add retry logic in LLM client if flakiness is observed.

**Implementation (PR4):**
```python
# app/llm/openai_client.py
def revise_selector_spec(self, ...):
    try:
        response = self.client.chat.completions.create(...)
        return self._parse_spec(response)
    except openai.APIConnectionError as e:
        raise InfrastructureError(f"LLM unreachable during revision: {e}")
    except openai.APIStatusError as e:
        if e.status_code >= 500:
            raise InfrastructureError(f"LLM error during revision: {e}")
        raise  # 4xx is likely our fault
```

The orchestrator doesn't need special handling - the exception bubbles up and `create_run` catches it as `InfrastructureError`.

---

### Logging & Observability

**Current State:** No structured logging in the codebase.

**Recommendation:** Add minimal structured logging in PR4, expand in later PRs.

**Logging Points:**

| Location | Log Level | What | Why |
|----------|-----------|------|-----|
| `create_run` entry | INFO | `run_id`, `fund_id`, request summary | Track run starts |
| `create_run` success | INFO | `run_id`, `status`, `outcome.status` | Track completions |
| `create_run` failure | ERROR | `run_id`, exception details | Debug failures |
| `run_program` iteration | DEBUG | iteration number, spec changes | Debug revision loop |
| LLM calls | DEBUG | model, token counts | Monitor costs |

**Implementation (PR4):**
```python
import logging

logger = logging.getLogger(__name__)

@app.post("/api/runs", status_code=status.HTTP_201_CREATED)
async def create_run(request: ProgramRequest, fund_id: Optional[str] = None):
    run_id = str(uuid4())
    logger.info(f"Creating run {run_id} for fund={fund_id}")

    try:
        response = run_program(request, get_llm_client())
        logger.info(f"Run {run_id} completed: outcome.status={response.outcome.status}")
        ...
    except InfrastructureError as e:
        logger.error(f"Run {run_id} failed (infrastructure): {e}")
        raise HTTPException(...)
    except Exception as e:
        logger.exception(f"Run {run_id} failed (unexpected)")
        raise HTTPException(...)
```

**Future Enhancements (not in MVP):**
- Structured JSON logging for log aggregation
- Request/response tracing with correlation IDs
- Prometheus metrics for run duration, status distribution
- OpenTelemetry spans for LLM call timing

---

### Concurrent Run Safety

**Scenario:** Multiple `POST /api/runs` requests arrive simultaneously.

**Current Design:** In-memory `RunStore` with `threading.Lock` is thread-safe for concurrent access.

**Risks:**
- Memory exhaustion if many runs created (mitigated by `limit` param on list)
- No persistence across restarts (documented as "out of scope")

**Not a concern:** Each run is independent. No shared state between runs except the store itself.

---

## Review Checklist Per PR

Each PR review should verify:
- [ ] Types align between Python and TypeScript (if applicable)
- [ ] Factory methods used (no manual field copying)
- [ ] Tests cover happy path + edge cases
- [ ] No breaking changes to existing API
- [ ] Existing tests pass

---

## Determinism Boundaries Summary

| Aspect | Deterministic? | Enforced By |
|--------|----------------|-------------|
| Floor = 100% when user override | Yes | `floor_fraction = 1.0` |
| Floor = 75% when LLM extraction | Yes | `RevisionPolicyConfig` |
| Per-iteration max drop = 20% | Yes | `max_per_iteration_target_drop_fraction` |
| Hard constraints immutable | Yes | `enforce_revision_policy` clamps |
| Filter relaxation ±0.1/iter | Yes | `enforce_revision_policy` clamps |
| Initial target extraction | No | LLM decision |
| Revision strategy | No | LLM decision |
| Explanation text | No | LLM decision |

---

## Validation Checklist

- [ ] `run_program()` returns `audit_trace` (required, not optional)
- [ ] `audit_trace.target_source` is "user_override" when override provided
- [ ] `audit_trace.floor_fraction` is 1.0 for user overrides
- [ ] `OutcomeSnapshot.from_outcome()` factory used in orchestrator
- [ ] `SpecSnapshot.from_spec()` factory used in orchestrator
- [ ] `test_model_alignment.py` passes (prevents field drift)
- [ ] Policy violations are structured `PolicyViolation` objects
- [ ] TypeScript types include all coverage fields
- [ ] Hard constraint immutability tests pass (constraints cannot be relaxed)
- [ ] Multi-iteration tests verify constraints stay locked
- [ ] Frontend renders `target_source` indicator
- [ ] `docker compose up` works end-to-end
