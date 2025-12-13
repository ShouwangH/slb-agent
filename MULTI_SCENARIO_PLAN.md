# Multi-Scenario Orchestrator Implementation Plan

> **Status**: Draft - Pending Approval
> **Date**: 2025-12-12
> **Author**: Claude (Implementation Planning)

---

## 1. Current State Overview

### 1.1 Files and Symbols Involved in Single-Scenario Runs

| File | Key Symbols | Purpose |
|------|-------------|---------|
| `app/models.py` | `ProgramRequest`, `ProgramResponse`, `SelectorSpec`, `ProgramOutcome`, `AuditTrace`, `AuditTraceEntry`, `SpecSnapshot`, `OutcomeSnapshot`, `PolicyViolation` | All data models |
| `app/orchestrator.py` | `run_program()`, `summarize_assets()` | Main entry point + agentic loop |
| `app/run_store.py` | `RunRecord`, `RunStore`, `run_store` | In-memory persistence layer |
| `app/api.py` | `create_program()`, `create_run()`, `get_run()`, `list_runs()` | HTTP endpoints |
| `app/engine/selector.py` | `select_assets()` | Core deterministic engine |
| `app/revision_policy.py` | `enforce_revision_policy()`, `PolicyResult` | Revision bounds enforcement |
| `app/llm/interface.py` | `LLMClient` protocol | LLM abstraction |
| `app/config.py` | `EngineConfig`, `RevisionPolicyConfig` | Configuration |
| `frontend/src/types/index.ts` | All TypeScript interfaces | Frontend type definitions |

### 1.2 Data Flow Summary

```
ProgramRequest (HTTP POST)
    │
    ▼
api.create_run()
    │
    ├── run_id = uuid4()
    │
    ▼
orchestrator.run_program()
    │
    ├── llm.generate_selector_spec() → SelectorSpec
    │
    ├── [Agentic Loop: max_iterations]
    │   ├── select_assets() → ProgramOutcome
    │   ├── If INFEASIBLE: llm.revise_selector_spec()
    │   └── enforce_revision_policy()
    │
    ├── generate_explanation_nodes() + llm.generate_explanation_summary()
    │
    ▼
ProgramResponse (with AuditTrace)
    │
    ▼
run_store.create(RunRecord)
    │
    ▼
HTTP 201 { run_id, status, response }
```

### 1.3 Key Execution Path

The "one true path" from request to response:

1. **Entry**: `api.create_run()` calls `orchestrator.run_program()`
2. **Spec Generation**: LLM generates `SelectorSpec` from natural language
3. **Engine Execution**: `select_assets()` performs greedy selection → `ProgramOutcome`
4. **Revision Loop**: If infeasible, LLM revises spec (bounded by `enforce_revision_policy`)
5. **Explanation**: Engine generates nodes, LLM generates summary
6. **Storage**: `RunRecord` created with `ProgramResponse`

---

## 2. Invariants We Must Not Break

### 2.1 Core Engine Interface

| Invariant | Location | Rationale |
|-----------|----------|-----------|
| `select_assets(assets, corporate_state, spec, config) → ProgramOutcome` | `engine/selector.py:118` | Single entry point for deterministic selection |
| `ProgramRequest` JSON shape | `models.py:590-612` | API contract with frontend |
| `ProgramResponse` JSON shape | `models.py:615-623` | API contract with frontend |
| `ProgramOutcome` fields | `models.py:357-401` | Frontend reads these directly |
| `AuditTrace` + `AuditTraceEntry` structure | `models.py:528-582` | Frontend renders audit timeline |

### 2.2 RunRecord Structure

| Field | Type | Must Preserve |
|-------|------|---------------|
| `run_id` | `str` | Yes - UUID4 primary key |
| `fund_id` | `Optional[str]` | Yes - filtering mechanism |
| `program_description` | `str` | Yes - UI display |
| `response` | `Optional[ProgramResponse]` | Yes - full response payload |
| `error` | `Optional[str]` | Yes - error capture |
| `created_at` | `str` | Yes - ISO timestamp |

### 2.3 HTTP API Contracts

| Endpoint | Contract |
|----------|----------|
| `POST /api/runs` | Returns `{ run_id, status, response?, error? }` |
| `GET /api/runs/{run_id}` | Returns full `RunRecord` shape |
| `GET /api/runs` | Returns list of `{ run_id, fund_id, program_description, status, created_at }` |

### 2.4 Revision Policy Invariants

| Rule | Location | Behavior |
|------|----------|----------|
| `program_type` immutable | `revision_policy.py:93-105` | Change → invalid |
| `hard_constraints` cannot relax | `revision_policy.py:111-245` | Clamp to original |
| `target_amount` monotonically decreasing | `revision_policy.py:252-298` | Max 20%/iter, floor enforced |
| Filter relaxation bounded | `revision_policy.py:304-402` | ±0.1/iter with ceiling/floor |

### 2.5 Frontend Assumptions

Based on `frontend/src/types/index.ts`:

- **RunRecord**: Frontend expects exactly `{ run_id, fund_id, program_description, status, response, error, created_at }`
- **RunListItem**: Expects `{ run_id, fund_id, program_description, status, created_at }` (no `response`)
- **Status**: Only `"completed" | "failed"` - no other values
- **AuditTrace**: Frontend assumes one trace per run

**Critical**: The frontend currently assumes **one run per request**. Multi-scenario will introduce N runs per user action.

---

## 3. Proposed New Types and Fields

### 3.1 ScenarioKind Enum

**File**: `app/models.py` (add after existing enums, ~line 87)

```python
class ScenarioKind(str, Enum):
    """
    Classification of scenario variants in a scenario set.

    These represent strategic alternatives generated from a single brief.
    """
    BASE = "base"              # Direct interpretation of the brief
    RISK_OFF = "risk_off"      # Conservative: tighter constraints, lower target
    AGGRESSIVE = "aggressive"  # Higher target, relaxed constraints
    GEO_FOCUS = "geo_focus"    # Geographic concentration variant
    CUSTOM = "custom"          # LLM-defined variant with custom rationale
```

**Rationale**: Closed enum provides type safety and UI-friendly labels. `CUSTOM` allows LLM flexibility.

### 3.2 Scenario Metadata on RunRecord

**File**: `app/run_store.py` (extend `RunRecord` dataclass)

```python
@dataclass
class RunRecord:
    run_id: str
    fund_id: Optional[str]
    program_description: str
    response: Optional[ProgramResponse]
    error: Optional[str]
    created_at: str

    # NEW: Scenario metadata (Optional for backwards compatibility)
    scenario_set_id: Optional[str] = None  # Links runs in the same set
    scenario_label: str = "Single Scenario"  # Human-readable label
    scenario_kind: Optional[str] = None  # ScenarioKind value or None
    scenario_brief_fragment: Optional[str] = None  # LLM rationale snippet
```

**Backwards Compatibility**: All new fields have defaults. Existing runs continue to work with `scenario_set_id=None`.

### 3.3 ScenarioSetSummary Model

**File**: `app/models.py` (add after API Models section, ~line 631)

```python
class ScenarioSetSummary(BaseModel):
    """
    Summary of a scenario set containing multiple program runs.

    Created when a user submits a brief that generates multiple scenario variants.
    """
    id: str = Field(..., description="Unique scenario set ID (UUID4)")
    brief: str = Field(..., description="Original natural language brief")
    base_spec_hash: Optional[str] = Field(
        None, description="Hash of the base ProgramRequest for deduplication"
    )
    created_at: str = Field(..., description="ISO timestamp")

    # Scenario tracking
    num_scenarios: int = Field(..., ge=1, description="Number of scenarios in set")
    run_ids: list[str] = Field(..., description="Ordered list of run IDs")
    scenario_labels: list[str] = Field(..., description="Labels for each scenario")
    scenario_kinds: list[str] = Field(..., description="ScenarioKind values")

    # Status rollup
    completed_count: int = Field(0, ge=0, description="Runs with status=completed")
    failed_count: int = Field(0, ge=0, description="Runs with status=failed")
```

### 3.4 ScenarioDefinition Model (Internal)

**File**: `app/models.py` (internal model, not exposed in API response)

```python
class ScenarioDefinition(BaseModel):
    """
    Definition of a single scenario variant, generated by LLM.

    Used internally to describe how to modify the base spec.
    This is the LLM output contract for scenario generation.
    """
    label: str = Field(..., description="Short human-readable label (e.g., 'Risk-Off')")
    kind: ScenarioKind = Field(..., description="Scenario classification")
    brief_fragment: str = Field(
        ..., max_length=200,
        description="1-2 sentence rationale for this variant"
    )

    # Spec modifications (all Optional - None means "use base value")
    target_amount_override: Optional[float] = Field(
        None, gt=0, description="Override target amount"
    )
    floor_override: Optional[float] = Field(
        None, gt=0, description="Override floor (minimum acceptable target)"
    )
    max_leverage_override: Optional[float] = Field(
        None, gt=0, lt=10, description="Override max net leverage"
    )
    min_coverage_override: Optional[float] = Field(
        None, gt=0, lt=20, description="Override min fixed charge coverage"
    )

    # Asset filter overrides
    max_criticality_override: Optional[float] = Field(
        None, ge=0, le=1, description="Override max criticality filter"
    )
    min_leaseability_override: Optional[float] = Field(
        None, ge=0, le=1, description="Override min leaseability filter"
    )
    exclude_markets_override: Optional[list[str]] = Field(
        None, description="Markets to exclude in this scenario"
    )
```

### 3.5 TypeScript Additions

**File**: `frontend/src/types/index.ts` (add after existing types)

```typescript
// =============================================================================
// Scenario Types (NEW)
// =============================================================================

export type ScenarioKind =
  | "base"
  | "risk_off"
  | "aggressive"
  | "geo_focus"
  | "custom";

export interface ScenarioSetSummary {
  id: string;
  brief: string;
  base_spec_hash: string | null;
  created_at: string;
  num_scenarios: number;
  run_ids: string[];
  scenario_labels: string[];
  scenario_kinds: string[];
  completed_count: number;
  failed_count: number;
}

// Extended RunRecord with optional scenario fields
export interface RunRecord {
  run_id: string;
  fund_id: string | null;
  program_description: string;
  status: "completed" | "failed";
  response: ProgramResponse | null;
  error: string | null;
  created_at: string;

  // NEW: Scenario metadata (optional for backwards compatibility)
  scenario_set_id?: string | null;
  scenario_label?: string;
  scenario_kind?: ScenarioKind | null;
  scenario_brief_fragment?: string | null;
}

// Extended RunListItem
export interface RunListItem {
  run_id: string;
  fund_id: string | null;
  program_description: string;
  status: "completed" | "failed";
  created_at: string;

  // NEW: Optional scenario metadata
  scenario_set_id?: string | null;
  scenario_label?: string;
  scenario_kind?: ScenarioKind | null;
}
```

---

## 4. Orchestrator Design

### 4.1 Module Structure

```
app/
├── orchestrator.py          # Existing single-scenario orchestrator (unchanged)
├── scenario_orchestrator.py # NEW: Multi-scenario coordination
├── scenario_store.py        # NEW: ScenarioSet storage
└── llm/
    ├── interface.py         # Existing (extend with new method)
    └── prompts/
        └── scenario_gen.py  # NEW: Prompt for scenario generation
```

### 4.2 Core Function Signatures

**File**: `app/scenario_orchestrator.py`

```python
from typing import Optional
from app.models import (
    ProgramRequest, ScenarioDefinition, ScenarioSetSummary, ScenarioKind
)
from app.run_store import RunRecord
from app.llm.interface import LLMClient
from app.config import EngineConfig

def generate_scenarios_from_brief(
    brief: str,
    base_request: ProgramRequest,
    llm: LLMClient,
    num_scenarios: int = 3,
    kinds: Optional[list[ScenarioKind]] = None,
) -> list[ScenarioDefinition]:
    """
    Generate N scenario definitions from a natural language brief.

    Args:
        brief: Natural language program description
        base_request: The base ProgramRequest to derive scenarios from
        llm: LLM client for generation
        num_scenarios: Number of scenarios to generate (default 3)
        kinds: Specific scenario kinds to generate (None = LLM chooses)

    Returns:
        List of ScenarioDefinition objects describing each variant

    Note:
        The first scenario should always be BASE (direct interpretation).
        LLM may generate fewer scenarios if brief doesn't support variety.
    """
    ...

def apply_scenario_patch(
    base_request: ProgramRequest,
    scenario: ScenarioDefinition,
) -> ProgramRequest:
    """
    Apply scenario modifications to create a concrete ProgramRequest.

    Args:
        base_request: The original ProgramRequest
        scenario: ScenarioDefinition with overrides

    Returns:
        New ProgramRequest with scenario modifications applied

    Invariant Enforcement:
        - floor_override cannot exceed target_amount
        - max_leverage_override bounded to [0, 10]
        - coverage overrides bounded to [0, 20]
        - filter values bounded to [0, 1]
    """
    ...

def run_scenario_set(
    brief: str,
    base_request: ProgramRequest,
    llm: LLMClient,
    config: EngineConfig,
    num_scenarios: int = 3,
    fund_id: Optional[str] = None,
) -> ScenarioSetSummary:
    """
    Execute a complete multi-scenario run.

    This is the main entry point for multi-scenario execution.

    Args:
        brief: Natural language program description
        base_request: Base ProgramRequest with assets and corporate state
        llm: LLM client
        config: Engine configuration
        num_scenarios: Target number of scenarios
        fund_id: Optional fund identifier

    Returns:
        ScenarioSetSummary with all run IDs and metadata

    Flow:
        1. Generate ScenarioDefinitions from brief
        2. For each scenario:
           a. Apply patch to create ProgramRequest
           b. Call run_program() (existing orchestrator)
           c. Store RunRecord with scenario metadata
        3. Create and store ScenarioSetSummary
        4. Return summary
    """
    ...
```

### 4.3 LLM Interface Extension

**File**: `app/llm/interface.py` (add to Protocol)

```python
def generate_scenario_definitions(
    self,
    brief: str,
    asset_summary: str,
    num_scenarios: int,
    existing_kinds: Optional[list[str]] = None,
) -> list[dict]:
    """
    Generate scenario variant definitions from a brief.

    Args:
        brief: Natural language program description
        asset_summary: Summary of available assets
        num_scenarios: Target number of scenarios
        existing_kinds: Kinds already generated (for diversity)

    Returns:
        List of dicts matching ScenarioDefinition schema

    Contract:
        - First scenario MUST be kind="base"
        - Each scenario has unique label
        - brief_fragment explains the variant's rationale
        - Overrides are relative to a "reasonable base interpretation"
    """
    ...
```

### 4.4 LLM Contract (JSON Schema)

```json
{
  "scenarios": [
    {
      "label": "Base Case",
      "kind": "base",
      "brief_fragment": "Direct interpretation: $10M SLB targeting distribution centers",
      "target_amount_override": null,
      "floor_override": null,
      "max_leverage_override": null,
      "min_coverage_override": null,
      "max_criticality_override": null,
      "min_leaseability_override": null,
      "exclude_markets_override": null
    },
    {
      "label": "Conservative",
      "kind": "risk_off",
      "brief_fragment": "Tighter leverage covenant (3.0x vs 4.0x) with lower target",
      "target_amount_override": 8000000,
      "floor_override": 7000000,
      "max_leverage_override": 3.0,
      "min_coverage_override": 3.5,
      "max_criticality_override": 0.5,
      "min_leaseability_override": null,
      "exclude_markets_override": null
    },
    {
      "label": "Aggressive",
      "kind": "aggressive",
      "brief_fragment": "Maximize proceeds with relaxed criticality threshold",
      "target_amount_override": 15000000,
      "floor_override": 12000000,
      "max_leverage_override": null,
      "min_coverage_override": null,
      "max_criticality_override": 0.8,
      "min_leaseability_override": 0.3,
      "exclude_markets_override": null
    }
  ]
}
```

### 4.5 Spec Patching Strategy

The `apply_scenario_patch()` function applies overrides with these rules:

| Field | Override Logic | Invariant Check |
|-------|----------------|-----------------|
| `program_description` | Append scenario label | None |
| `floor_override` | Use scenario value if set | Must be ≤ target_amount |
| `max_leverage_override` | Use scenario value if set | Bounded [0, 10] |
| `min_coverage_override` | Use scenario value if set | Bounded [0, 20] |
| Asset filters | Merge with base filters | Values bounded [0, 1] |

**Key Invariant**: The base request's `assets` and `corporate_state` are NEVER modified. Only `program_description` and override fields change.

### 4.6 Scenario Store

**File**: `app/scenario_store.py`

```python
from dataclasses import dataclass
from typing import Optional
import threading

@dataclass
class ScenarioSetRecord:
    """Storage record for a scenario set."""
    summary: ScenarioSetSummary

class ScenarioStore:
    """Thread-safe in-memory store for scenario sets."""

    def __init__(self) -> None:
        self._sets: dict[str, ScenarioSetRecord] = {}
        self._lock = threading.Lock()

    def create(self, summary: ScenarioSetSummary) -> None:
        with self._lock:
            self._sets[summary.id] = ScenarioSetRecord(summary=summary)

    def get(self, set_id: str) -> Optional[ScenarioSetSummary]:
        with self._lock:
            record = self._sets.get(set_id)
            return record.summary if record else None

    def list_sets(
        self,
        fund_id: Optional[str] = None,
        limit: int = 10,
    ) -> list[ScenarioSetSummary]:
        # Filter by fund_id if provided (requires joining with runs)
        ...

scenario_store = ScenarioStore()
```

---

## 5. API Changes (Additive)

### 5.1 New Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/scenario_sets` | Create and execute multi-scenario run |
| `GET` | `/api/scenario_sets` | List scenario sets |
| `GET` | `/api/scenario_sets/{id}` | Get scenario set with runs |

### 5.2 POST /api/scenario_sets

**Request**:
```json
{
  "brief": "We need $10M in SLB proceeds targeting distribution centers...",
  "base_request": {
    "assets": [...],
    "corporate_state": {...},
    "program_type": "slb",
    "program_description": "...",
    "floor_override": null,
    "max_leverage_override": null,
    "min_coverage_override": null
  },
  "num_scenarios": 3,
  "fund_id": "optional-fund-id"
}
```

**Response** (201 Created):
```json
{
  "scenario_set": {
    "id": "uuid-here",
    "brief": "We need $10M...",
    "created_at": "2025-12-12T10:00:00Z",
    "num_scenarios": 3,
    "run_ids": ["run-1", "run-2", "run-3"],
    "scenario_labels": ["Base Case", "Conservative", "Aggressive"],
    "scenario_kinds": ["base", "risk_off", "aggressive"],
    "completed_count": 3,
    "failed_count": 0
  },
  "runs": [
    { "run_id": "run-1", "status": "completed", ... },
    { "run_id": "run-2", "status": "completed", ... },
    { "run_id": "run-3", "status": "completed", ... }
  ]
}
```

### 5.3 GET /api/scenario_sets

**Query Parameters**:
- `fund_id`: Optional filter
- `limit`: Max results (default 10, max 100)

**Response** (200 OK):
```json
[
  {
    "id": "uuid-1",
    "brief": "We need $10M...",
    "created_at": "...",
    "num_scenarios": 3,
    "run_ids": [...],
    "scenario_labels": [...],
    "scenario_kinds": [...],
    "completed_count": 3,
    "failed_count": 0
  }
]
```

### 5.4 GET /api/scenario_sets/{id}

**Response** (200 OK):
```json
{
  "scenario_set": {
    "id": "uuid",
    "brief": "...",
    ...
  },
  "runs": [
    // Full RunRecord objects with responses
  ]
}
```

### 5.5 Impact on Existing Routes

| Endpoint | Change |
|----------|--------|
| `GET /api/runs` | **Additive**: Returns new optional fields `scenario_set_id`, `scenario_label`, `scenario_kind` |
| `GET /api/runs/{id}` | **Additive**: Returns new optional fields |
| `POST /api/runs` | **No change**: Still creates single-scenario runs |

**Backwards Compatibility**: Existing consumers can ignore new fields (they're optional). The `status` field remains `"completed" | "failed"`.

---

## 6. Frontend Impact Analysis

### 6.1 Components That Read Runs

| Component | File | Impact |
|-----------|------|--------|
| `RunList` | `components/RunList.tsx` | Low - can ignore scenario fields |
| `RunsPanel` | `components/RunsPanel.tsx` | Low - displays run list |
| `ScenarioPlannerPage` | `pages/ScenarioPlannerPage.tsx` | **Medium** - needs scenario set support |

### 6.2 Current Assumption: One Run Per Request

The frontend currently calls `POST /api/runs` and expects one run back. Multi-scenario changes this to:

1. **New flow**: Call `POST /api/scenario_sets` → get N runs back
2. **Display change**: Show scenario tabs/cards instead of single result
3. **Comparison view**: Side-by-side metrics for scenarios

### 6.3 Minimal UI Adaptation Path

**Phase 1** (No UI changes):
- Frontend continues using `POST /api/runs` for single scenarios
- New scenario fields are ignored
- Existing functionality unchanged

**Phase 2** (Optional scenario display):
- If `scenario_set_id` is present on a run, show scenario label
- Add "View Scenario Set" link
- Runs list can filter by scenario set

**Phase 3** (Full scenario support):
- New "Run Scenario Analysis" button
- Scenario set creation form
- Comparison view for scenarios
- Scenario-aware run list

### 6.4 Type Changes Required

```typescript
// RunListItem now has optional scenario fields - safe to ignore
interface RunListItem {
  // ... existing fields ...
  scenario_set_id?: string | null;  // NEW - can ignore
  scenario_label?: string;           // NEW - can ignore
}
```

---

## 7. Migration Plan

### Step 1: Add New Models and Fields (No Behavior Change)

**Files to modify**:
- `app/models.py`: Add `ScenarioKind`, `ScenarioDefinition`, `ScenarioSetSummary`
- `app/run_store.py`: Add optional scenario fields to `RunRecord`
- `frontend/src/types/index.ts`: Add TypeScript equivalents

**Tests**:
- All existing tests pass (no behavior change)
- New unit tests for model validation

**Verification**:
```bash
pytest tests/ -v
# All existing tests should pass
```

### Step 2: Wire Scenario Metadata into Existing Runs

**Files to modify**:
- `app/api.py`: Pass scenario metadata through `create_run()`
- `app/run_store.py`: Handle new fields in storage

**Default values for existing runs**:
```python
RunRecord(
    # ... existing fields ...
    scenario_set_id=None,        # Not part of a set
    scenario_label="Single Scenario",
    scenario_kind=None,          # Not classified
    scenario_brief_fragment=None,
)
```

**Tests**:
- Existing endpoint tests pass
- New tests for metadata round-trip

### Step 3: Add Scenario Store and Orchestrator

**New files**:
- `app/scenario_store.py`: ScenarioStore class
- `app/scenario_orchestrator.py`: Multi-scenario coordination

**Tests**:
- Unit tests for `apply_scenario_patch()` invariant enforcement
- Integration test for `run_scenario_set()` with mock LLM

### Step 4: Add LLM Scenario Generation

**Files to modify**:
- `app/llm/interface.py`: Add `generate_scenario_definitions()` to protocol
- `app/llm/mock.py`: Add mock implementation
- `app/llm/openai_client.py`: Add OpenAI implementation

**Tests**:
- Mock LLM returns valid scenario definitions
- Schema validation on LLM output

### Step 5: Add New API Endpoints

**Files to modify**:
- `app/api.py`: Add `/api/scenario_sets` routes

**Tests**:
- Endpoint integration tests
- Error handling tests

### Step 6: Frontend Updates (Optional)

**Files to modify**:
- `frontend/src/types/index.ts`: Already done in Step 1
- Components: Only if scenario UI is needed

---

## 8. Open Questions / Risks

### 8.1 Open Questions

| Question | Impact | Recommendation |
|----------|--------|----------------|
| Should scenario sets have their own `fund_id`? | API design | No - inherit from runs for simplicity |
| Should we support partial execution (some scenarios fail)? | UX | Yes - return completed + failed in same response |
| How to handle LLM timeout during scenario generation? | Reliability | Return partial results + error for remaining |
| Should scenarios share the same `AuditTrace` or have separate ones? | Data model | **Separate** - each run has its own trace |

### 8.2 Potential Breaking Edges

| Risk | Mitigation |
|------|------------|
| Frontend crashes on new fields | All new fields are optional with defaults |
| Existing `/api/runs` consumers break | Response shape only gains fields, never loses them |
| ScenarioSetSummary serialization | Use Pydantic's `model_dump()` for JSON |
| Race conditions in stores | Both stores use `threading.Lock()` |

### 8.3 Performance Considerations

| Concern | Mitigation |
|---------|------------|
| N LLM calls per scenario set | Batch scenario generation in single LLM call |
| N engine runs sequentially | Consider parallel execution (asyncio) in future |
| Large response payloads | Scenario set list endpoint returns summaries only |

### 8.4 Fields That CANNOT Change

These would break the frontend or existing integrations:

- `RunRecord.run_id` (primary key)
- `RunRecord.response` (full ProgramResponse)
- `ProgramResponse.outcome.status` values
- `AuditTrace` structure
- `/api/runs/{id}` response shape (only additive changes)

### 8.5 Deferred Decisions

| Decision | Why Deferred |
|----------|--------------|
| Async scenario execution | Adds complexity; sequential is fine for MVP |
| Scenario set deletion | Not needed yet; can add later |
| Scenario comparison metrics | UI-dependent; design with frontend team |
| Persistent storage | In-memory is fine for MVP; migrate to DB later |

---

## 9. Summary

This plan introduces multi-scenario orchestration as an **additive layer** on top of the existing single-scenario system:

1. **New models**: `ScenarioKind`, `ScenarioDefinition`, `ScenarioSetSummary`
2. **Extended models**: `RunRecord` gains optional scenario metadata
3. **New orchestrator**: `run_scenario_set()` coordinates N single-scenario runs
4. **New API**: `/api/scenario_sets` endpoints (additive)
5. **Existing API**: `/api/runs` unchanged except for new optional fields

**Key principle**: Every existing test should pass after each step. The single-scenario path remains the default and unchanged.
