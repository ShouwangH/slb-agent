# Multi-Scenario Orchestrator Implementation Plan

> **Status**: Draft v2 - Revised per reviewer feedback
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

## 2. Invariants

### 2.1 Protocol Invariants (MUST preserve)

These are API contracts that external consumers depend on. Breaking these breaks the frontend.

| Invariant | Location | Why It's Sacred |
|-----------|----------|-----------------|
| `ProgramRequest` JSON shape | `models.py:590-612` | Frontend serializes this |
| `ProgramResponse` JSON shape | `models.py:615-623` | Frontend deserializes this |
| `ProgramOutcome.status` enum values | `models.py:52-57` | `"ok" \| "infeasible" \| "numeric_error"` |
| `RunRecord` core fields | `run_store.py:19-37` | Frontend expects exact shape |
| `AuditTrace` structure | `models.py:559-582` | Frontend renders timeline |
| `/api/runs` response shape | `api.py:259-320` | Only additive changes allowed |

### 2.2 Engine Invariants (MUST preserve)

The deterministic engine must remain a pure function.

| Invariant | Location | Behavior |
|-----------|----------|----------|
| `select_assets()` signature | `engine/selector.py:118` | `(assets, corporate_state, spec, config) → ProgramOutcome` |
| Greedy selection is deterministic | `engine/selector.py:177-209` | Same inputs → same outputs |
| Constraint checking is pure | `engine/metrics.py` | No side effects |

### 2.3 Behavioral Invariants (Within single-run revision loop ONLY)

These apply to the **revision loop within a single run**. They do NOT apply to multi-scenario generation.

| Rule | Scope | Behavior |
|------|-------|----------|
| `program_type` immutable | Single-run revision | Change → invalid |
| `hard_constraints` cannot relax | Single-run revision | Clamp to original |
| `target_amount` monotonically decreasing | Single-run revision | Max 20%/iter, floor enforced |
| Filter relaxation bounded | Single-run revision | ±0.1/iter with ceiling/floor |

> **IMPORTANT**: The monotonic target rule is a revision-loop constraint, not a scenario-generation constraint. Multi-scenario explicitly allows different target amounts per scenario.

### 2.4 Frontend Assumptions

- **RunRecord**: Expects `{ run_id, fund_id, program_description, status, response, error, created_at }`
- **RunListItem**: Expects `{ run_id, fund_id, program_description, status, created_at }`
- **Status**: Only `"completed" | "failed"`
- **AuditTrace**: One trace per run

**Critical**: Frontend currently assumes **one run per request**. Multi-scenario introduces N runs per user action.

---

## 3. Product Decision: Proceeds Story

> **Decision**: Scenarios represent **different capital asks**.
>
> Each scenario may have a different target amount (proceeds goal). This is intentional—the point of multi-scenario is to explore "what if we needed $8M vs $12M?" alongside constraint variations.
>
> **Implications**:
> - `target_amount` is a scenario parameter, not shared
> - LLM can vary target per scenario
> - The monotonic target rule in `revision_policy.py` applies only to the **within-run revision loop**, not to scenario generation

---

## 4. Proposed New Types and Fields

### 4.1 ScenarioKind Enum

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

### 4.2 Scenario Metadata on RunRecord

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

    # NEW: Scenario metadata
    # None = single-scenario run (not part of any set)
    scenario_set_id: Optional[str] = None

    # Optional[ScenarioKind] - None means "single-scenario, no classification"
    scenario_kind: Optional[ScenarioKind] = None

    # Human-readable label (only meaningful when scenario_set_id is not None)
    scenario_label: Optional[str] = None
```

**Type semantics**:
- `scenario_set_id = None` → This is a standalone single-scenario run
- `scenario_set_id = "uuid"` → This run belongs to a scenario set
- `scenario_kind = None` → Not classified (single-scenario or legacy run)
- `scenario_kind = ScenarioKind.BASE` → Base scenario in a set

### 4.3 ScenarioSetSummary Model (Simplified)

**File**: `app/models.py` (add after API Models section, ~line 631)

```python
class ScenarioSetSummary(BaseModel):
    """
    Summary of a scenario set. Metadata lives on RunRecords, not here.

    This is a lightweight index—fetch runs by ID for full details.
    """
    id: str = Field(..., description="Unique scenario set ID (UUID4)")
    brief: str = Field(..., description="Original natural language brief")
    created_at: str = Field(..., description="ISO timestamp")
    run_ids: list[str] = Field(..., description="Ordered list of run IDs")
```

**Design note**: Scenario labels, kinds, and status are derived from `RunRecord`s. No duplicate storage.

### 4.4 ScenarioDefinition Model (v1 - Minimal LLM Surface)

**File**: `app/models.py` (internal model for LLM output)

```python
class ScenarioDefinition(BaseModel):
    """
    Definition of a single scenario variant, generated by LLM.

    v1: Minimal control surface. Only target and two key constraints.
    """
    label: str = Field(..., min_length=1, max_length=50,
        description="Short label (e.g., 'Conservative')")
    kind: ScenarioKind = Field(..., description="Scenario classification")
    rationale: str = Field(..., min_length=1, max_length=200,
        description="1-2 sentence rationale for this variant")

    # v1: Only three numeric knobs
    target_amount: float = Field(..., gt=0,
        description="Target proceeds for this scenario")
    max_leverage: Optional[float] = Field(None, gt=0, lt=10,
        description="Max net leverage (None = use default)")
    min_coverage: Optional[float] = Field(None, gt=0, lt=20,
        description="Min fixed charge coverage (None = use default)")
```

**v1 constraints**:
- No filter overrides (max_criticality, min_leaseability) - use defaults
- No market exclusions - use base request
- Keep LLM contract small and strict

### 4.5 TypeScript Additions

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
  created_at: string;
  run_ids: string[];
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

  // Scenario metadata (null = single-scenario run, not part of any set)
  scenario_set_id: string | null;
  scenario_kind: ScenarioKind | null;
  scenario_label: string | null;
}

// Extended RunListItem
export interface RunListItem {
  run_id: string;
  fund_id: string | null;
  program_description: string;
  status: "completed" | "failed";
  created_at: string;

  // Scenario metadata (null = single-scenario run)
  scenario_set_id: string | null;
  scenario_kind: ScenarioKind | null;
  scenario_label: string | null;
}
```

---

## 5. Orchestrator Design

### 5.1 Module Structure

```
app/
├── orchestrator.py          # Existing single-scenario orchestrator (unchanged)
├── scenario_orchestrator.py # NEW: Multi-scenario coordination
└── run_store.py             # EXTENDED: Add scenario set tracking
```

**Design decision**: No separate `ScenarioStore`. ScenarioSetSummary is lightweight (just `id`, `brief`, `created_at`, `run_ids`) and can be stored in `RunStore` alongside runs. This avoids cross-store consistency issues.

### 5.2 RunStore Extension

**File**: `app/run_store.py` (extend existing class)

```python
class RunStore:
    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}
        self._scenario_sets: dict[str, ScenarioSetSummary] = {}  # NEW
        self._lock = threading.Lock()

    # ... existing methods ...

    # NEW: Scenario set methods
    def create_scenario_set(self, summary: ScenarioSetSummary) -> None:
        with self._lock:
            self._scenario_sets[summary.id] = summary

    def get_scenario_set(self, set_id: str) -> Optional[ScenarioSetSummary]:
        with self._lock:
            return self._scenario_sets.get(set_id)

    def list_scenario_sets(self, limit: int = 10) -> list[ScenarioSetSummary]:
        with self._lock:
            sets = list(self._scenario_sets.values())
        sets.reverse()  # Most recent first
        return sets[:limit]

    def get_runs_for_set(self, set_id: str) -> list[RunRecord]:
        """Get all runs belonging to a scenario set."""
        with self._lock:
            return [r for r in self._runs.values() if r.scenario_set_id == set_id]
```

### 5.3 Core Function Signatures

**File**: `app/scenario_orchestrator.py`

```python
from typing import Optional
from app.models import ProgramRequest, ScenarioDefinition, ScenarioSetSummary
from app.run_store import run_store
from app.llm.interface import LLMClient
from app.config import EngineConfig

def build_scenario_request(
    base_request: ProgramRequest,
    scenario: ScenarioDefinition,
) -> ProgramRequest:
    """
    Build a concrete ProgramRequest from base + scenario definition.

    Args:
        base_request: The original ProgramRequest (assets, corporate_state)
        scenario: ScenarioDefinition with target and constraints

    Returns:
        New ProgramRequest ready for run_program()

    Invariants:
        - assets and corporate_state are NEVER modified
        - program_description gets scenario label appended
        - max_leverage bounded to (0, 10)
        - min_coverage bounded to (0, 20)
    """
    ...

def run_scenario_set(
    brief: str,
    base_request: ProgramRequest,
    llm: LLMClient,
    config: EngineConfig,
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
        1. scenario_set_id = uuid4()
        2. scenarios = llm.generate_scenario_definitions(brief, num_scenarios)
        3. For each scenario:
           a. request = build_scenario_request(base_request, scenario)
           b. response = run_program(request, llm, config)
           c. Store RunRecord with scenario_set_id, scenario_kind, scenario_label
        4. Store ScenarioSetSummary
        5. Return (summary, run_results)
    """
    ...
```

### 5.4 LLM Interface Extension

**File**: `app/llm/interface.py` (add to Protocol)

```python
def generate_scenario_definitions(
    self,
    brief: str,
    asset_summary: str,
    num_scenarios: int,
) -> list[ScenarioDefinition]:
    """
    Generate scenario definitions from a brief.

    Args:
        brief: Natural language program description
        asset_summary: Summary of available assets (from summarize_assets)
        num_scenarios: Number of scenarios to generate (1-5)

    Returns:
        List of ScenarioDefinition objects

    Contract (v1):
        - First scenario MUST be kind=BASE
        - Each scenario has unique label
        - target_amount is required for each scenario
        - max_leverage and min_coverage are optional
    """
    ...
```

### 5.5 LLM Contract (v1 - Strict and Minimal)

```json
{
  "scenarios": [
    {
      "label": "Base Case",
      "kind": "base",
      "rationale": "Direct interpretation: $10M SLB targeting distribution centers",
      "target_amount": 10000000,
      "max_leverage": null,
      "min_coverage": null
    },
    {
      "label": "Conservative",
      "kind": "risk_off",
      "rationale": "Tighter leverage covenant (3.0x) with lower $8M target",
      "target_amount": 8000000,
      "max_leverage": 3.0,
      "min_coverage": 3.5
    },
    {
      "label": "Aggressive",
      "kind": "aggressive",
      "rationale": "Maximize proceeds at $15M with standard constraints",
      "target_amount": 15000000,
      "max_leverage": null,
      "min_coverage": null
    }
  ]
}
```

**v1 constraints**:
- Only 3 numeric knobs: `target_amount` (required), `max_leverage` (optional), `min_coverage` (optional)
- No filter overrides
- No market exclusions
- Rationale capped at 200 chars

### 5.6 Revision Strategy: Base vs Variant Scenarios

**Key design decision**: Base scenario runs deterministically; variant scenarios use the agentic loop.

| Scenario Kind | Revision Loop? | Floor | Rationale |
|---------------|----------------|-------|-----------|
| `BASE` | **No** | `floor = target` (sacred) | User asked for exactly this—show if it's achievable |
| `RISK_OFF`, `AGGRESSIVE`, etc. | **Yes** | `floor = target * 0.9` | Explore what's actually possible |

**Why?**
- Base scenario answers: "Can I get exactly $10M?" → Show feasible or infeasible, no revision.
- Variant scenarios answer: "What if I tried $15M?" → Let the agent find what's achievable within bounds.

### 5.7 Request Building Strategy

The `build_scenario_request()` function:

```python
def build_scenario_request(
    base_request: ProgramRequest,
    scenario: ScenarioDefinition,
) -> ProgramRequest:
    # Base scenario: no revision flexibility (floor = target)
    # Variant scenarios: allow 10% reduction during revision
    if scenario.kind == ScenarioKind.BASE:
        floor = scenario.target_amount  # Sacred - no revision
    else:
        floor = scenario.target_amount * 0.9  # Allow agentic exploration

    return ProgramRequest(
        # Unchanged from base
        assets=base_request.assets,
        corporate_state=base_request.corporate_state,
        program_type=base_request.program_type,

        # Updated
        program_description=f"{base_request.program_description} [{scenario.label}]",

        # Scenario overrides
        floor_override=floor,
        max_leverage_override=scenario.max_leverage,  # None = use LLM inference
        min_coverage_override=scenario.min_coverage,  # None = use LLM inference
    )
```

**Result**:
- Base scenario with `floor_override = target_amount` → `floor_fraction = 1.0` → revision loop can't reduce target → runs once, returns feasible/infeasible
- Variant scenarios with `floor_override = target * 0.9` → revision loop can explore down to 90% of target

**Key invariant**: `assets` and `corporate_state` are NEVER modified. Each scenario is a different "capital ask" on the same portfolio.

---

## 6. API Changes (Additive)

### 6.1 New Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/scenario_sets` | Create and execute multi-scenario run |
| `GET` | `/api/scenario_sets` | List scenario sets |
| `GET` | `/api/scenario_sets/{id}` | Get scenario set with runs |

### 6.2 POST /api/scenario_sets

**Request**:
```json
{
  "brief": "We need $10M in SLB proceeds targeting distribution centers...",
  "base_request": {
    "assets": [...],
    "corporate_state": {...},
    "program_type": "slb",
    "program_description": "..."
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
    "run_ids": ["run-1", "run-2", "run-3"]
  },
  "runs": [
    {
      "run_id": "run-1",
      "status": "completed",
      "scenario_set_id": "uuid-here",
      "scenario_kind": "base",
      "scenario_label": "Base Case",
      "response": { ... }
    },
    {
      "run_id": "run-2",
      "status": "completed",
      "scenario_set_id": "uuid-here",
      "scenario_kind": "risk_off",
      "scenario_label": "Conservative",
      "response": { ... }
    },
    {
      "run_id": "run-3",
      "status": "completed",
      "scenario_set_id": "uuid-here",
      "scenario_kind": "aggressive",
      "scenario_label": "Aggressive",
      "response": { ... }
    }
  ]
}
```

**Note**: Scenario metadata (labels, kinds, status) lives on runs, not on the set summary.

### 6.3 GET /api/scenario_sets

**Query Parameters**:
- `limit`: Max results (default 10, max 100)

**Response** (200 OK):
```json
[
  {
    "id": "uuid-1",
    "brief": "We need $10M...",
    "created_at": "2025-12-12T10:00:00Z",
    "run_ids": ["run-1", "run-2", "run-3"]
  }
]
```

### 6.4 GET /api/scenario_sets/{id}

**Response** (200 OK):
```json
{
  "scenario_set": {
    "id": "uuid",
    "brief": "...",
    "created_at": "...",
    "run_ids": [...]
  },
  "runs": [
    // Full RunRecord objects with scenario metadata and responses
  ]
}
```

### 6.5 Impact on Existing Routes

| Endpoint | Change |
|----------|--------|
| `GET /api/runs` | **Additive**: Returns `scenario_set_id`, `scenario_kind`, `scenario_label` (all nullable) |
| `GET /api/runs/{id}` | **Additive**: Returns same new fields |
| `POST /api/runs` | **No change**: Creates single-scenario runs with `scenario_*` fields = null |

**Backwards Compatibility**:
- New fields are always present but nullable
- `null` means "single-scenario run, not part of any set"
- `status` field unchanged: `"completed" | "failed"`

---

## 7. Frontend Impact Analysis

### 7.1 Components That Read Runs

| Component | File | Impact |
|-----------|------|--------|
| `RunList` | `components/RunList.tsx` | Low - scenario fields are nullable |
| `RunsPanel` | `components/RunsPanel.tsx` | Low - displays run list |
| `ScenarioPlannerPage` | `pages/ScenarioPlannerPage.tsx` | **Medium** - needs scenario set support |

### 7.2 Current Assumption: One Run Per Request

The frontend currently calls `POST /api/runs` and expects one run back. Multi-scenario changes this to:

1. **New flow**: Call `POST /api/scenario_sets` → get N runs back
2. **Display change**: Show scenario tabs/cards instead of single result
3. **Comparison view**: Side-by-side metrics for scenarios

### 7.3 Minimal UI Adaptation Path

**Phase 1** (No UI changes):
- Frontend continues using `POST /api/runs` for single scenarios
- New scenario fields are null, frontend ignores them
- Existing functionality unchanged

**Phase 2** (Optional scenario display):
- If `scenario_set_id !== null`, show scenario label badge
- Add "View Scenario Set" link
- Runs list can group by scenario set

**Phase 3** (Full scenario support):
- New "Run Scenario Analysis" button
- Scenario set creation form
- Comparison view for scenarios
- Scenario-aware run list

### 7.4 Type Changes

Types are non-breaking. New fields are nullable:

```typescript
// Before: fields didn't exist
// After: fields exist but are null for single-scenario runs
interface RunListItem {
  // ... existing fields ...
  scenario_set_id: string | null;   // null = single scenario
  scenario_kind: ScenarioKind | null;
  scenario_label: string | null;
}
```

---

## 8. PR Roadmap

Each PR is independently reviewable and mergeable. Invariants are verified at each step.

---

### PR 1: Add Scenario Models (No Behavior Change)

**Scope**: Add new types only. No runtime behavior changes.

**Files**:
| File | Changes |
|------|---------|
| `app/models.py` | Add `ScenarioKind`, `ScenarioDefinition`, `ScenarioSetSummary` |
| `frontend/src/types/index.ts` | Add TypeScript equivalents |

**Invariant Checks**:
- [ ] `ProgramRequest` unchanged
- [ ] `ProgramResponse` unchanged
- [ ] `ProgramOutcome` unchanged
- [ ] `AuditTrace` unchanged
- [ ] All existing tests pass: `pytest tests/ -v`

**Review Focus**:
- Type definitions match between Python and TypeScript
- No imports of new types in existing code yet

---

### PR 2: Extend RunRecord with Scenario Fields

**Scope**: Add optional fields to `RunRecord`. All defaults = `None`.

**Files**:
| File | Changes |
|------|---------|
| `app/run_store.py` | Add `scenario_set_id`, `scenario_kind`, `scenario_label` to `RunRecord` |

**Code**:
```python
@dataclass
class RunRecord:
    # ... existing required fields (unchanged) ...
    run_id: str
    fund_id: Optional[str]
    program_description: str
    response: Optional[ProgramResponse]
    error: Optional[str]
    created_at: str

    # NEW: Optional scenario metadata (defaults = None)
    scenario_set_id: Optional[str] = None
    scenario_kind: Optional[ScenarioKind] = None
    scenario_label: Optional[str] = None
```

**Invariant Checks**:
- [ ] Existing `RunRecord` construction sites unchanged (defaults handle it)
- [ ] `run_store.create()` works with old-style records
- [ ] `run_store.get()` returns records with new fields = None
- [ ] All existing tests pass: `pytest tests/ -v`

**Review Focus**:
- Dataclass field ordering (defaults must come after non-defaults)
- No changes to existing `RunRecord` instantiation sites

---

### PR 3: Expose Scenario Fields in API Responses

**Scope**: Return scenario fields in `/api/runs` endpoints (always present, nullable).

**Files**:
| File | Changes |
|------|---------|
| `app/api.py` | Update `get_run()` and `list_runs()` response dicts |
| `frontend/src/types/index.ts` | Update `RunRecord` and `RunListItem` interfaces |

**API Changes**:
```python
# GET /api/runs/{id} - add to response
return {
    # ... existing fields ...
    "scenario_set_id": record.scenario_set_id,   # null for existing runs
    "scenario_kind": record.scenario_kind.value if record.scenario_kind else None,
    "scenario_label": record.scenario_label,     # null for existing runs
}
```

**Invariant Checks**:
- [ ] Existing fields unchanged (run_id, fund_id, status, etc.)
- [ ] New fields are always present (not omitted)
- [ ] New fields are `null` for all existing runs
- [ ] Frontend TypeScript compiles
- [ ] All existing tests pass

**Review Focus**:
- Response shape is additive only
- `scenario_kind` serialized as string value, not enum object

---

### PR 4: Add Scenario Set Storage to RunStore

**Scope**: Extend `RunStore` with scenario set methods. No API exposure yet.

**Files**:
| File | Changes |
|------|---------|
| `app/run_store.py` | Add `_scenario_sets` dict and CRUD methods |
| `tests/test_run_store.py` | Add tests for new methods |

**New Methods**:
```python
def create_scenario_set(self, summary: ScenarioSetSummary) -> None
def get_scenario_set(self, set_id: str) -> Optional[ScenarioSetSummary]
def list_scenario_sets(self, limit: int = 10) -> list[ScenarioSetSummary]
def get_runs_for_set(self, set_id: str) -> list[RunRecord]
def clear(self) -> None  # Update to also clear _scenario_sets
```

**Invariant Checks**:
- [ ] Existing `RunStore` methods unchanged
- [ ] Thread safety preserved (single lock covers both dicts)
- [ ] `clear()` clears both runs and scenario sets
- [ ] All existing tests pass

**Review Focus**:
- Lock acquisition order (single lock, no deadlocks)
- `get_runs_for_set()` filters correctly by `scenario_set_id`

---

### PR 5: Add LLM Scenario Generation Interface

**Scope**: Extend `LLMClient` protocol. Add mock implementation.

**Files**:
| File | Changes |
|------|---------|
| `app/llm/interface.py` | Add `generate_scenario_definitions()` to Protocol |
| `app/llm/mock.py` | Add mock implementation |
| `tests/test_llm_mock.py` | Add tests for new method |

**Protocol Addition**:
```python
def generate_scenario_definitions(
    self,
    brief: str,
    asset_summary: str,
    num_scenarios: int,
) -> list[ScenarioDefinition]:
    """Generate scenario definitions from a brief."""
    ...
```

**Mock Implementation Contract**:
- Always returns exactly `num_scenarios` definitions
- First scenario is always `kind=BASE`
- Returns deterministic results for testing

**Invariant Checks**:
- [ ] Existing `LLMClient` methods unchanged
- [ ] `MockLLMClient` satisfies full protocol
- [ ] All existing tests pass

**Review Focus**:
- Mock returns valid `ScenarioDefinition` objects
- First scenario is always BASE

---

### PR 6: Add OpenAI Scenario Generation

**Scope**: Implement `generate_scenario_definitions()` for OpenAI client.

**Files**:
| File | Changes |
|------|---------|
| `app/llm/openai_client.py` | Add `generate_scenario_definitions()` |
| `app/llm/prompts/scenario_gen.py` | NEW: Prompt template |

**Prompt Contract**:
- Input: brief, asset_summary, num_scenarios
- Output: JSON array matching `ScenarioDefinition` schema
- First scenario MUST be `kind=base`
- All scenarios must have unique labels

**Invariant Checks**:
- [ ] Existing OpenAI methods unchanged
- [ ] Output validated against `ScenarioDefinition` schema
- [ ] Graceful error handling for malformed LLM output

**Review Focus**:
- Prompt clarity and constraints
- JSON parsing and validation
- Error handling for invalid LLM responses

---

### PR 7: Add Scenario Orchestrator (Core Logic)

**Scope**: New `scenario_orchestrator.py` with `build_scenario_request()` and `run_scenario_set()`.

**Files**:
| File | Changes |
|------|---------|
| `app/scenario_orchestrator.py` | NEW: Core orchestration logic |
| `tests/test_scenario_orchestrator.py` | NEW: Unit and integration tests |

**Key Functions**:
```python
def build_scenario_request(base_request, scenario) -> ProgramRequest
def run_scenario_set(brief, base_request, llm, config, num_scenarios, fund_id) -> (ScenarioSetSummary, list[dict])
```

**Critical Invariants**:
- [ ] `build_scenario_request()` NEVER modifies `assets` or `corporate_state`
- [ ] BASE scenario gets `floor_override = target_amount` (no revision)
- [ ] Variant scenarios get `floor_override = target * 0.9` (agentic)
- [ ] Each scenario calls existing `run_program()` unchanged
- [ ] `RunRecord`s stored with correct `scenario_set_id`, `scenario_kind`, `scenario_label`

**Determinism Checks**:
- [ ] BASE scenario: `floor_fraction = 1.0` → revision loop exits after 1 iteration
- [ ] Same inputs → same outputs (via existing engine determinism)

**Review Focus**:
- `build_scenario_request()` preserves base request immutably
- BASE vs variant floor logic is correct
- Error handling for individual scenario failures

---

### PR 8: Add Scenario Set API Endpoints

**Scope**: Wire up HTTP endpoints for scenario sets.

**Files**:
| File | Changes |
|------|---------|
| `app/api.py` | Add `POST/GET /api/scenario_sets`, `GET /api/scenario_sets/{id}` |
| `tests/test_api.py` | Add endpoint tests |

**Endpoints**:
| Method | Path | Handler |
|--------|------|---------|
| POST | `/api/scenario_sets` | `create_scenario_set()` |
| GET | `/api/scenario_sets` | `list_scenario_sets()` |
| GET | `/api/scenario_sets/{id}` | `get_scenario_set()` |

**Invariant Checks**:
- [ ] Existing `/api/runs` endpoints unchanged
- [ ] `POST /api/runs` still works for single scenarios
- [ ] Response shapes match spec in Section 6

**Review Focus**:
- Error handling (404 for missing set, 503 for LLM failure)
- Response includes both `scenario_set` and `runs`

---

### PR 9: Frontend Type Updates (Optional)

**Scope**: Update frontend to consume new fields. No UI changes yet.

**Files**:
| File | Changes |
|------|---------|
| `frontend/src/types/index.ts` | Ensure scenario types are complete |
| `frontend/src/components/RunList.tsx` | Handle nullable scenario fields |

**Invariant Checks**:
- [ ] Existing UI renders correctly
- [ ] No runtime errors from new nullable fields
- [ ] TypeScript compiles without errors

---

## 9. PR Dependency Graph

```
PR1 (models)
  │
  ▼
PR2 (RunRecord fields)
  │
  ├──────────────────┐
  ▼                  ▼
PR3 (API response)  PR4 (RunStore methods)
  │                  │
  │                  ▼
  │                PR5 (LLM interface + mock)
  │                  │
  │                  ▼
  │                PR6 (OpenAI implementation)
  │                  │
  └──────────────────┤
                     ▼
                   PR7 (scenario_orchestrator.py)
                     │
                     ▼
                   PR8 (API endpoints)
                     │
                     ▼
                   PR9 (Frontend - optional)
```

**Parallel Work**:
- PR3 and PR4 can proceed in parallel after PR2
- PR5 and PR6 can proceed after PR4
- PR7 requires PR3, PR4, PR5/PR6

---

## 10. Verification Checklist (Per PR)

Run these checks before merging each PR:

```bash
# 1. All existing tests pass
pytest tests/ -v

# 2. Type checking passes
mypy app/

# 3. No regressions in API responses
# (manual or automated API tests)

# 4. Frontend compiles (if types changed)
cd frontend && npm run build
```

**Interface Consistency Checks**:
- [ ] Python `ScenarioKind` values match TypeScript `ScenarioKind` type
- [ ] Python `ScenarioSetSummary` fields match TypeScript interface
- [ ] API response JSON keys match TypeScript interface properties
- [ ] Nullable fields are `T | null` in TypeScript, `Optional[T]` in Python

---

## 9. Open Questions / Risks

### 9.1 Resolved Questions

| Question | Decision |
|----------|----------|
| Proceeds story? | **Different capital asks** - target varies per scenario |
| Separate ScenarioStore? | **No** - fold into RunStore |
| Duplicate metadata on ScenarioSetSummary? | **No** - derive from runs |
| LLM control surface? | **Minimal v1** - only target, max_leverage, min_coverage |

### 9.2 Remaining Open Questions

| Question | Impact | Recommendation |
|----------|--------|----------------|
| Partial execution (some scenarios fail)? | UX | Yes - return completed + failed |
| LLM timeout during generation? | Reliability | Fail entire set, don't return partial |

### 9.3 Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Frontend breaks on new fields | Fields are nullable, always present |
| Type confusion on scenario_kind | `Optional[ScenarioKind]` with explicit null semantics |
| Store consistency | Single `RunStore` with lock, no cross-store issues |

### 9.4 Deferred Decisions

| Decision | Why Deferred |
|----------|--------------|
| Async scenario execution | Sequential is fine for v1 |
| Filter overrides (criticality, leaseability) | Keep LLM surface small for v1 |
| Market exclusions per scenario | Adds complexity, defer to v2 |
| Persistent storage | In-memory is fine for MVP |

---

## 10. Summary

This plan introduces multi-scenario orchestration as an **additive layer**:

| Component | Change |
|-----------|--------|
| `models.py` | Add `ScenarioKind`, `ScenarioDefinition`, `ScenarioSetSummary` |
| `run_store.py` | Add scenario fields to `RunRecord`, add scenario set storage |
| `scenario_orchestrator.py` | NEW: `build_scenario_request()`, `run_scenario_set()` |
| `llm/interface.py` | Add `generate_scenario_definitions()` to protocol |
| `api.py` | Add `/api/scenario_sets` endpoints |
| `/api/runs` | Additive nullable fields only |

**Key principles**:
1. Every existing test passes after each step
2. Single-scenario path unchanged (scenario fields = null)
3. Metadata lives on runs, not duplicated on sets
4. LLM contract is minimal for v1 (3 numeric knobs)
5. Monotonic target rule applies to revision loop only, not scenario generation
6. **Base scenario is deterministic** (no revision) - shows exactly what user asked for
7. **Variant scenarios are agentic** - explore what's achievable within bounds
