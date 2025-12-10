# Implementation Roadmap

**Goal:** Implement DESIGN.md in small, reviewable PRs that maintain architectural consistency and testability.

**Principles:**
- Each PR should be independently reviewable (< 500 lines ideally)
- Each PR should leave the codebase in a working state
- Tests accompany the code they test (not in a separate PR)
- No forward references to unimplemented code
- Dependencies flow downward: API → Orchestrator → Engine → Models/Config

---

## Phase 1: Foundation (PRs 1-3)

### PR 1: Project Setup & Models
**Scope:** Project structure, dependencies, Pydantic models, enums

**Files:**
```
slb-agent/
├── pyproject.toml
├── app/
│   ├── __init__.py
│   └── models.py
└── tests/
    ├── __init__.py
    └── test_models.py
```

**What to implement:**
- `pyproject.toml` with dependencies: `pydantic>=2.0`, `fastapi`, `openai`, `pytest`, `pytest-asyncio`, `hypothesis`
- All enums: `AssetType`, `MarketTier`, `ProgramType`, `Objective`, `SelectionStatus`
- Input models: `Asset`, `CorporateState`
- Spec models: `HardConstraints`, `SoftPreferences`, `AssetFilters`, `SelectorSpec`
- Output models: `ConstraintViolation`, `AssetSelection`, `AssetSLBMetrics`
- Explanation models: `ExplanationNode`, `Explanation`
- API models: `ProgramRequest`, `ProgramResponse`
- Do NOT implement `ProgramOutcome` yet (depends on metrics structure from PR 2)

**Tests:**
- Pydantic validation works (required fields, bounds)
- Enums serialize correctly
- Invalid values rejected

**Review focus:**
- Field naming consistency
- Validation rules match DESIGN.md Section 4
- No business logic in models

---

### PR 2: Engine Config & Metrics Models
**Scope:** Configuration object, metrics dataclasses, remaining models

**Files:**
```
app/
├── config.py
└── models.py (update)
```

**What to implement:**
- `EngineConfig` with all parameters (cap rates, thresholds, multipliers)
- `DEFAULT_CAP_RATE_CURVE` constant
- `DEFAULT_ENGINE_CONFIG` instance
- Internal metrics models: `BaselineMetrics`, `PostMetrics`, `PortfolioMetrics`
- `ProgramOutcome` model (now that metrics structure is defined)

**Tests:**
- Config defaults match DESIGN.md Section 5.1
- Config validation (epsilon > 0, rates in valid ranges)
- Metrics models accept None for optional fields

**Review focus:**
- All magic numbers in config, not scattered
- Clear separation: config (parameters) vs models (data structures)

---

### PR 3: Input Validation
**Scope:** Validation functions for assets, corporate state, spec

**Files:**
```
app/
└── validation.py
tests/
└── test_validation.py
```

**What to implement:**
- `validate_asset(asset: Asset) -> List[str]`
- `validate_assets(assets: List[Asset]) -> List[str]`
- `validate_corporate_state(state: CorporateState) -> List[str]`
- `validate_spec(spec: SelectorSpec) -> List[str]`
- Custom exception: `ValidationError`

**Tests:**
- Each validation rule from DESIGN.md Section 7.1-7.2
- Edge cases: empty list, duplicate IDs, boundary values
- Valid inputs return empty list

**Review focus:**
- Error messages are actionable
- Validation is pure (no side effects)
- All invariants from DESIGN.md enforced

---

## Phase 2: Pure Core Engine (PRs 4-5)

### PR 4: Metrics Engine
**Scope:** All metric computation (asset-level + fund-level + constraints)

These are tightly coupled: you compute asset metrics to get proceeds/rent, then fund metrics to get leverage/coverage, then check constraints. Splitting them artificially fragments the math.

**Files:**
```
app/
└── engine/
    ├── __init__.py
    └── metrics.py
tests/
└── test_metrics.py
```

**What to implement:**
- `compute_asset_slb_metrics(asset: Asset, config: EngineConfig) -> AssetSLBMetrics`
- `compute_baseline_metrics(state: CorporateState, config: EngineConfig) -> BaselineMetrics`
- `compute_post_transaction_metrics(state, selected, config) -> Tuple[PostMetrics, List[str]]`
- `compute_critical_fraction(selected, config) -> float`
- `check_constraints(selected, state, hard_constraints, target, config) -> Tuple[PortfolioMetrics, List[ConstraintViolation], List[str]]`
- All numeric guardrails: None for undefined, clamping with warnings

**Tests:**
- Asset metrics: known inputs → expected outputs, different types/tiers
- Baseline metrics: normal inputs, ebitda ≈ 0 → None
- Post-transaction: various proceeds, over-repayment warning, interest clamping
- Idempotence: empty selection → before == after
- Constraints: each type passing/failing, None handling, multiple violations

**Review focus:**
- Pure functions throughout
- Warnings returned, not logged
- Math matches DESIGN.md Sections 5.2-5.3, 6.3

---

### PR 5: Selection Algorithm
**Scope:** Scoring, filtering, greedy selection - all in one cohesive module

**Files:**
```
app/engine/
└── selector.py
tests/
└── test_selector.py
```

**What to implement:**
- `compute_score(asset: Asset, preferences: SoftPreferences) -> float`
- `apply_filters(assets: List[Asset], filters: AssetFilters) -> List[Asset]`
- `select_assets(assets, state, spec, config) -> ProgramOutcome`
- Greedy loop: sort by score, add if constraints pass, stop at target

**Tests:**
- Scoring: low criticality preferred, high leaseability preferred
- Filtering: each filter type individually and combined
- Selection: feasible scenario → OK
- Selection: infeasible target → TARGET_NOT_MET
- Selection: constraint violation → skips violating assets
- Selection: no eligible assets → NO_ELIGIBLE_ASSETS

**Review focus:**
- Single entry point: `select_assets`
- Calls metrics.py for all computation (no duplicated math)
- Status determination logic correct

---

## Phase 3: Orchestration (PRs 6-9)

### PR 6: Explanation Generation (Engine Side)
**Scope:** Structured explanation node creation

**Files:**
```
app/engine/
└── explanations.py
tests/
└── test_explanations.py
```

**What to implement:**
- `generate_explanation_nodes(spec, outcome, state, config) -> List[ExplanationNode]`
- Node generation for: binding constraints, selection drivers, risks
- Severity assignment logic

**Tests:**
- Binding constraint nodes have correct metrics/thresholds
- Driver nodes reference selection criteria
- Risk nodes identify concentration issues
- Node IDs are unique

**Review focus:**
- Pure function, no LLM calls
- Nodes are machine-readable (structured data)
- Consistent with DESIGN.md Section 4.6

---

### PR 7: LLM Interface & Mock
**Scope:** LLM interface with mock implementation for testing

**Files:**
```
app/
├── llm/
│   ├── __init__.py
│   ├── interface.py
│   └── mock.py
tests/
└── test_llm_mock.py
```

**What to implement:**
- Abstract interface: `LLMClient` (Protocol) with methods:
  - `generate_selector_spec(program_type, description, summary) -> SelectorSpec`
  - `revise_selector_spec(description, prev_spec, outcome) -> SelectorSpec`
  - `generate_explanation_summary(nodes) -> str`
- Mock implementation: `MockLLMClient` with deterministic responses
- Configuration to select mock vs real client

**Tests:**
- Mock client returns valid structured data
- Interface contract documented

**Review focus:**
- Clean abstraction (no OpenAI-specific code in interface)
- Mock is deterministic and controllable for testing
- Prompt templates NOT in this PR (just interface)

---

### PR 8: Revision Policy
**Scope:** Revision policy as standalone, testable module (NOT in orchestrator)

Extracting this prevents the orchestrator from becoming a god object. The policy is pure logic with no dependencies on LLM or engine.

**Files:**
```
app/
└── revision_policy.py
tests/
└── test_revision_policy.py
```

**What to implement:**
- `PolicyResult` dataclass: `valid: bool`, `spec: Optional[SelectorSpec]`, `violations: List[str]`
- `enforce_revision_policy(immutable_hard, original_target, prev_spec, new_spec) -> PolicyResult`
- Immutable hard constraint checks (leverage, coverage cannot relax)
- Target monotonicity (can only decrease)
- Per-iteration bounds (max 20% reduction)
- Global floor (50% of original)
- Filter relaxation bounds (criticality +0.1, leaseability -0.1)

**Tests:**
- Relaxing max_net_leverage → rejected
- Relaxing min_fixed_charge_coverage → rejected
- Target increase → rejected
- Target drop > 20% → clamped to 20%
- Target below 50% floor → rejected (invalid)
- Filter relaxation within bounds → accepted
- Filter relaxation beyond bounds → clamped

**Review focus:**
- Pure function, no dependencies on engine or LLM
- All rules from DESIGN.md Section 8.2-8.3
- Policy violations are descriptive strings

---

### PR 9: Orchestrator (Thin Coordinator)
**Scope:** Thin orchestration loop that delegates to other modules

The orchestrator's job is coordination only: call LLM, call engine, call policy, repeat.

**Files:**
```
app/
└── orchestrator.py
tests/
└── test_orchestrator.py
```

**What to implement:**
- `run_program(request: ProgramRequest, llm: LLMClient, config: EngineConfig) -> ProgramResponse`
- Loop structure: generate spec → validate → run selector → check status → (revise if needed) → generate explanation
- Delegates to:
  - `llm.generate_selector_spec()` / `llm.revise_selector_spec()`
  - `selector.select_assets()`
  - `revision_policy.enforce_revision_policy()`
  - `explanations.generate_explanation_nodes()`
  - `llm.generate_explanation_summary()`

**Tests (using MockLLMClient):**
- Happy path: first spec works → returns OK
- Revision loop: infeasible → revise → OK
- Max iterations: stops after limit
- Policy violation in revision → loop terminates
- Explanation nodes passed to LLM for summary

**Review focus:**
- Orchestrator does NOT contain policy logic (delegates to revision_policy.py)
- Orchestrator does NOT recompute metrics (uses outcome from selector)
- Each step is a single function call to another module
- Easy to trace: "what happens when" is linear

---

## Phase 4: API Layer (PRs 10-11)

### PR 10: FastAPI Application
**Scope:** HTTP API with single endpoint

**Files:**
```
app/
└── api.py
tests/
└── test_api.py
```

**What to implement:**
- FastAPI app instance
- `POST /program` endpoint
- Request validation via Pydantic
- Error response model
- HTTP status code mapping (200, 400, 422, 500)

**Tests:**
- Valid request → 200 with ProgramResponse
- Invalid program_type → 400
- Validation error → 400
- Pydantic error → 422
- Internal error → 500

**Review focus:**
- Thin layer: validation and routing only
- No business logic in API layer
- Error responses are structured

---

### PR 11: OpenAI LLM Client
**Scope:** Real LLM implementation with prompts

**Files:**
```
app/llm/
├── openai_client.py
└── prompts.py
tests/
└── test_prompts.py
```

**What to implement:**
- `OpenAILLMClient(LLMClient)` using `openai` SDK
- `call_llm_structured[T](system, user, response_model) -> T`
- Prompt templates from DESIGN.md Section 9
- Asset summary generation

**Tests:**
- Prompt templates produce valid strings
- Structured output parsing works (mock OpenAI response)
- Error handling for API failures

**Review focus:**
- Prompts match DESIGN.md exactly
- No business logic in prompts
- Proper error handling and retries

---

## Phase 5: Testing & Polish (PRs 12-13)

### PR 12: Golden Baseline & Invariant Tests
**Scope:** Canonical test portfolio + property-based tests (combined since both are "quality gates")

**Files:**
```
tests/
├── golden/
│   ├── __init__.py
│   └── canonical_portfolio.py
├── test_golden.py
└── test_invariants.py
```

**What to implement:**
- `GOLDEN_ASSETS` (10 assets, hand-picked)
- `GOLDEN_CORPORATE_STATE`
- `GOLDEN_SCENARIOS` with expected outcomes
- Golden tests that fail if economics change unexpectedly
- Property tests for metric computation (no NaN/Inf)
- Transaction invariants (leverage never increases)
- Guardrail invariants (interest never negative)

**Tests:**
- Golden: baseline scenario with specific expected values
- Golden: infeasible scenarios with correct violations
- Invariant: random valid inputs → no crashes
- Invariant: leverage_after <= leverage_before
- Invariant: interest_after >= 0

**Review focus:**
- Expected values are hand-verified
- Hypothesis strategies generate realistic inputs
- Maintenance instructions in comments

---

### PR 13: Documentation & Entry Point
**Scope:** README, CLI entry, final polish

**Files:**
```
├── README.md
├── app/
│   └── __main__.py
└── scripts/
    └── run_demo.py
```

**What to implement:**
- README with setup, usage, architecture overview
- `__main__.py` for `python -m app` invocation
- Demo script with sample portfolio
- Environment variable configuration (API keys)

**Review focus:**
- README is accurate and complete
- Demo script works out of the box
- No leftover TODOs or debug code

---

## PR Dependency Graph

```
PR 1 (Models)
    │
    ├── PR 2 (Config & Metrics Models)
    │       │
    │       └── PR 3 (Validation)
    │               │
    │               └── PR 4 (Metrics Engine)
    │                       │
    │                       └── PR 5 (Selector)
    │                               │
    │                               ├── PR 6 (Explanations)
    │                               │       │
    │                               │       └───────────────┐
    │                               │                       │
    │                               └── PR 7 (LLM Interface)│
    │                                       │               │
    │                                       └── PR 8 (Revision Policy)
    │                                               │
    │                                               └── PR 9 (Orchestrator)
    │                                                       │
    │                                                       ├── PR 10 (API)
    │                                                       │       │
    │                                                       │       └── PR 11 (OpenAI Client)
    │                                                       │
    │                                                       └── PR 12 (Golden & Invariant Tests)
    │                                                               │
    │                                                               └── PR 13 (Docs)
```

---

## Review Checklist (for every PR)

### Code Quality
- [ ] No magic numbers (all in config)
- [ ] Pure functions where possible
- [ ] Explicit dependencies (no globals except DEFAULT_ENGINE_CONFIG)
- [ ] Type hints on all public functions
- [ ] Docstrings on public functions

### Architecture
- [ ] Respects layer boundaries (API → Orchestrator → Engine → Models)
- [ ] No circular imports
- [ ] Single responsibility per module
- [ ] Config passed explicitly, not accessed globally in engine

### Testing
- [ ] Tests accompany new code
- [ ] Edge cases covered
- [ ] Mocks used appropriately (not testing implementation details)
- [ ] Tests are deterministic

### Consistency
- [ ] Naming matches DESIGN.md
- [ ] Error codes match DESIGN.md
- [ ] Field names match across layers

---

## Estimated Timeline

| Phase | PRs | Effort |
|-------|-----|--------|
| Foundation | 1-3 | ~1 day |
| Core Engine | 4-5 | ~1 day |
| Orchestration | 6-9 | ~1.5 days |
| API Layer | 10-11 | ~0.5 days |
| Testing & Polish | 12-13 | ~0.5 days |
| **Total** | **13 PRs** | **~4.5 days** |

---

## Notes

### What's NOT in v1 (future PRs)
- Multi-bucket debt model (revolver + term)
- EBITDA reduction from SLB rent
- OR-Tools optimization
- Document ingestion
- Production infra (k8s, observability)

### Risk Areas
- **LLM reliability:** PR 11 should have retry logic and fallbacks
- **Numeric edge cases:** PR 4 has the most complexity; review carefully
- **Revision policy:** PR 8 has subtle invariants; test thoroughly

### Shortcuts for Demo
If time-constrained, these can be simplified:
- PR 6: Skip risk node generation, just do constraints + drivers
- PR 12: Skip hypothesis, use hand-written edge case tests only
- PR 11: Use gpt-4o-mini initially, upgrade later
