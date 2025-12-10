# DESIGN.md — Real Estate Funding Workflow Agent (SLB Template)

**Version:** 1.0
**Status:** Implementation-Ready
**Estimated Implementation:** 1–2 days

---

## Table of Contents

1. [Overview](#1-overview)
2. [Scope and Non-Goals](#2-scope-and-non-goals)
3. [Tech Stack and Repository Layout](#3-tech-stack-and-repository-layout)
4. [Data Models](#4-data-models)
5. [SLB Economics Model](#5-slb-economics-model)
6. [Selection Algorithm (Engine)](#6-selection-algorithm-engine)
7. [Validation and Defaults](#7-validation-and-defaults)
8. [Agentic Loop and Revision Policy](#8-agentic-loop-and-revision-policy)
9. [LLM Prompting Strategy](#9-llm-prompting-strategy)
10. [API Design](#10-api-design)
11. [Minimal UI Sketch](#11-minimal-ui-sketch)
12. [Testing Strategy](#12-testing-strategy)
13. [Future Extensions](#13-future-extensions)

---

## 1. Overview

### 1.1 Story

A corporate real estate team needs to raise capital via Sale-Leaseback (SLB) transactions. They have:
- A portfolio of real estate assets with varying characteristics
- Corporate financial metrics (debt, EBITDA, interest expense)
- A natural-language program brief describing their goals and constraints

This agent translates their intent into an executable funding program, selects optimal assets, validates financial constraints, and produces investment-committee-ready explanations.

### 1.2 High-Level Flow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Program Brief  │────▶│  LLM: Generate  │────▶│  SelectorSpec   │
│  (Natural Lang) │     │  SelectorSpec   │     │  (Structured)   │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Asset[]        │────▶│  Deterministic  │────▶│  ProgramOutcome │
│  CorporateState │     │  Engine         │     │  (Metrics)      │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                              ┌───────────────────┬──────┴──────┐
                              ▼                   ▼             ▼
                         [status=OK]        [INFEASIBLE]   [NUMERIC_ERROR]
                              │                   │             │
                              │                   ▼             ▼
                              │            ┌─────────────┐    Error
                              │            │ LLM: Revise │
                              │            │ SelectorSpec│
                              │            └──────┬──────┘
                              │                   │
                              │          (loop ≤ max_iterations)
                              │                   │
                              ▼                   ▼
                        ┌─────────────────────────────┐
                        │  LLM: Generate Explanation  │
                        └──────────────┬──────────────┘
                                       │
                                       ▼
                        ┌─────────────────────────────┐
                        │       ProgramResponse       │
                        │  (spec, outcome, explanation)│
                        └─────────────────────────────┘
```

### 1.3 Key Design Principles

| Principle | Implementation |
|-----------|----------------|
| **LLM as Planner** | Translates intent → structure; never performs calculations |
| **Engine as Truth** | All numeric metrics computed deterministically |
| **Explicit Agentic Loop** | Bounded iterations with policy enforcement |
| **Domain Knowledge** | CRE/SLB economics baked into engine and config |

---

## 2. Scope and Non-Goals

### 2.1 In Scope

| Area | Details |
|------|---------|
| **Program Type** | Single: Sale-Leaseback (SLB) for retailer-like portfolios |
| **Backend** | Python service using FastAPI, Pydantic, OpenAI API |
| **LLM Calls** | (1) Initial spec generation, (2) Spec revision on infeasibility, (3) Explanation generation |
| **Selection Engine** | Simplified SLB economics, constraint enforcement, greedy heuristic |
| **API** | Single `POST /program` endpoint |
| **UI** | Sketch only (data contracts, wireframe) |
| **Testing** | Unit tests for engine; contract tests for orchestrator |

### 2.2 Out of Scope

| Area | Rationale |
|------|-----------|
| **Document ingestion** | Data room parsing, rent rolls, Argus → future v2 |
| **RAG / Embeddings** | Multi-source data fusion not needed for demo |
| **Other program types** | Revolver, CMBS, closures → future modular extension |
| **Global optimization** | MILP/ILP → greedy heuristic sufficient for demo |
| **Production infra** | k8s, CI/CD, observability → noted but not implemented |

### 2.3 Why This Slice?

This demo is **not** an end-to-end Auquan clone. It is a focused vertical slice chosen to demonstrate:

1. **Domain expertise** in CRE funding structures (SLB economics, leverage covenants)
2. **Agentic reasoning** with explicit revision loops and policy constraints
3. **Clean architecture** separating LLM intent-parsing from deterministic computation
4. **Practical constraints** (bounded iterations, immutable hard constraints, monotonic revision)

---

## 3. Tech Stack and Repository Layout

### 3.1 Technology Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Language** | Python 3.11+ | Type hints, async support |
| **Web Framework** | FastAPI | Async, auto-docs, Pydantic native |
| **Validation** | Pydantic v2 | Runtime validation, JSON schema |
| **HTTP Client** | `openai` SDK | Native structured outputs |
| **Testing** | pytest | Standard, async support |
| **Orchestration** | Hand-rolled | No LangChain/LangGraph — explicit control |

### 3.2 Directory Structure

```
slb-agent/
├── app/
│   ├── __init__.py
│   ├── models.py          # Pydantic models & enums (Section 4)
│   ├── engine.py          # SLB economics + greedy selector (Section 6)
│   ├── llm.py             # Wrappers: generate_spec, revise_spec, generate_explanation
│   ├── orchestrator.py    # run_program + agentic loop (Section 8)
│   ├── api.py             # FastAPI app, routes
│   └── config.py          # Constants, cap rate curve, defaults
├── tests/
│   ├── test_engine.py     # Unit tests for selection engine
│   └── test_orchestrator.py # Integration tests with mocked LLM
├── README.md              # Setup and usage instructions
└── DESIGN.md              # This document
```

### 3.3 Key Files and Responsibilities

| File | Primary Responsibility |
|------|------------------------|
| `models.py` | All Pydantic models, enums, type aliases |
| `engine.py` | `compute_slb_metrics()`, `select_assets()`, constraint checking |
| `llm.py` | OpenAI API wrappers, prompt templates, JSON parsing |
| `orchestrator.py` | `run_program()`, revision policy enforcement |
| `api.py` | FastAPI app instance, endpoint definitions |
| `config.py` | `EngineConfig`, `DEFAULT_ENGINE_CONFIG`, cap rate curve |

### 3.4 Architecture Layers

The system is structured into three cleanly separated layers to prevent metric drift and ensure consistency.

```
┌─────────────────────────────────────────────────────────────────┐
│                     UI / LLM Layer                              │
│  • Renders charts, tables, explanations                         │
│  • Feeds LLM prompts with pre-computed numbers                 │
│  • NEVER recomputes metrics — uses values from ProgramOutcome  │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ ProgramResponse (JSON DTOs)
                              │
┌─────────────────────────────────────────────────────────────────┐
│                   API / Orchestration Layer                     │
│  • Maps HTTP input → engine input                              │
│  • Maps engine output → JSON DTOs                              │
│  • Applies scenario presets                                    │
│  • Runs agentic loop, enforces revision policy                 │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ ProgramOutcome, ExplanationNode[]
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      Pure Core Engine                           │
│  • SINGLE source of truth for all metrics                      │
│  • Given: portfolio state, transaction params, config          │
│  • Returns: baseline metrics, post metrics, constraint evals,  │
│             structured explanations (machine-readable)         │
│  • Deterministic, stateless, side-effect-free                  │
└─────────────────────────────────────────────────────────────────┘
```

**Critical Invariant:** Metrics are computed exactly once, in the pure core engine. If the UI, LLM, or API layer needs a metric, it reads from `ProgramOutcome`. It does NOT recompute.

**Why this matters:**
- Prevents metric drift (different values in different places)
- Makes debugging straightforward (one place to look)
- Enables confident testing (mock engine, test orchestration separately)
- Allows LLM to summarize pre-computed results without risk of calculation errors

### 3.5 Asset vs Fund Separation

Within the engine, computation is further separated into two stages:

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: Asset-Level Computation                               │
│  • For each asset: compute market_value, proceeds, slb_rent    │
│  • Output: List[AssetSLBMetrics]                               │
│  • Pure function of (asset, config)                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2: Fund-Level Computation                                │
│  • Aggregate asset transactions                                │
│  • Update: net_debt, interest, lease_expense                   │
│  • Compute: leverage, coverage metrics                         │
│  • Pure function of (fund_state, asset_transactions, config)   │
└─────────────────────────────────────────────────────────────────┘
```

**Benefits:**
- Clear invariants ("asset math" vs "fund math")
- Flexibility to add other transaction types (partial sales, debt refinancing) with same fund engine
- Easier unit testing (test asset metrics independently from fund aggregation)

---

## 4. Data Models

All models defined in `app/models.py`. Implementation should be mechanical from these specifications.

### 4.1 Enums

```python
class AssetType(str, Enum):
    STORE = "store"
    DISTRIBUTION_CENTER = "distribution_center"
    OFFICE = "office"
    MIXED_USE = "mixed_use"
    OTHER = "other"

class MarketTier(int, Enum):
    TIER_1 = 1  # Primary markets (NYC, LA, Chicago)
    TIER_2 = 2  # Secondary markets (Austin, Nashville)
    TIER_3 = 3  # Tertiary markets

class ProgramType(str, Enum):
    SLB = "slb"
    # Future: REVOLVER = "revolver", CMBS = "cmbs", etc.

class Objective(str, Enum):
    MAXIMIZE_PROCEEDS = "maximize_proceeds"
    MINIMIZE_RISK = "minimize_risk"
    BALANCED = "balanced"

class SelectionStatus(str, Enum):
    OK = "ok"
    INFEASIBLE = "infeasible"
    NUMERIC_ERROR = "numeric_error"
```

### 4.2 Asset

Represents a single real estate property in the portfolio.

| Field | Type | Required | V1 Used | Description |
|-------|------|----------|---------|-------------|
| `asset_id` | `str` | Yes | Yes | Unique identifier |
| `name` | `Optional[str]` | No | No | Human-readable name |
| `asset_type` | `AssetType` | Yes | Yes | Property classification |
| `market` | `str` | Yes | Yes | Location (e.g., "Dallas, TX") |
| `country` | `Optional[str]` | No | No | Country code |
| `market_tier` | `Optional[MarketTier]` | No | Yes | Market classification |
| `noi` | `float` | Yes | Yes | Annual Net Operating Income ($) |
| `book_value` | `float` | Yes | Yes | Accounting book value ($) |
| `criticality` | `float` | Yes | Yes | Mission-criticality score [0, 1] |
| `leaseability_score` | `float` | Yes | Yes | Re-lease/repurpose ease [0, 1] |
| `tenant_name` | `Optional[str]` | No | No | Primary tenant |
| `tenant_credit_score` | `Optional[float]` | No | No | Tenant credit [0, 1] |
| `wault_years` | `Optional[float]` | No | No | Weighted avg unexpired lease term |
| `demographic_index` | `Optional[float]` | No | No | Future: location demographics |
| `esg_risk_score` | `Optional[float]` | No | No | Future: ESG risk |
| `current_ltv` | `Optional[float]` | No | No | Future: existing leverage |
| `existing_debt_amount` | `Optional[float]` | No | No | Future: encumbered debt |
| `encumbrance_type` | `Optional[str]` | No | No | Future: mortgage, ground lease |

**V1 Active Fields:** `asset_id`, `asset_type`, `market`, `market_tier`, `noi`, `book_value`, `criticality`, `leaseability_score`

**Validation Rules:**
- `noi > 0`
- `book_value > 0`
- `0 <= criticality <= 1`
- `0 <= leaseability_score <= 1`

### 4.3 CorporateState

Pre-transaction corporate financial position.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `net_debt` | `float` | Yes | Total debt minus cash ($) |
| `ebitda` | `float` | Yes | Trailing 12-month EBITDA ($) |
| `interest_expense` | `float` | Yes | Annual interest expense ($) |
| `lease_expense` | `Optional[float]` | No | Pre-SLB operating lease expense (default: 0) |

**Validation Invariants:**
- `net_debt >= 0`
- `ebitda` may be negative (but leverage/coverage become undefined or inverted)
- `interest_expense >= 0`
- `lease_expense >= 0` if provided

**Derived Metrics (pre-transaction):**
- `leverage_before = net_debt / ebitda` (undefined if `ebitda ≈ 0`)
- `interest_coverage_before = ebitda / interest_expense` (undefined if `interest_expense ≈ 0`)
- `fixed_charge_coverage_before = ebitda / (interest_expense + lease_expense)` (undefined if denominator `≈ 0`)

### 4.4 SelectorSpec

Structured representation of program parameters, generated/revised by LLM.

#### 4.4.1 HardConstraints

Immutable constraints that the engine must satisfy. Once set from initial spec, the LLM cannot relax these.

| Field | Type | Default | Semantics |
|-------|------|---------|-----------|
| `max_net_leverage` | `Optional[float]` | 4.0 | `net_debt_after / ebitda <= max_net_leverage` |
| `min_interest_coverage` | `Optional[float]` | None | `ebitda / interest_after >= min_interest_coverage` |
| `min_fixed_charge_coverage` | `Optional[float]` | 3.0 | `ebitda / (interest_after + total_lease_expense) >= min_fixed_charge_coverage` |
| `max_critical_fraction` | `Optional[float]` | None | `sum(noi_selected where criticality > 0.7) / sum(noi_selected) <= max_critical_fraction` |
| `max_asset_type_share` | `Optional[Dict[AssetType, float]]` | None | V2: per-type concentration limits |

**Coverage Metric Selection:**
- `min_interest_coverage` tests pure interest coverage (EBITDA / Interest)
- `min_fixed_charge_coverage` tests fixed-charge coverage (EBITDA / (Interest + Leases))
- Both can be specified simultaneously; both must pass
- For SLB transactions, `min_fixed_charge_coverage` is typically the binding constraint since SLB adds lease expense
- **Important:** EBITDA is held constant in this model (see Section 5.4). This is a demo simplification.

#### 4.4.2 SoftPreferences

Adjustable preferences that influence scoring. LLM may relax these during revision.

| Field | Type | Default | Semantics |
|-------|------|---------|-----------|
| `prefer_low_criticality` | `bool` | True | Favor non-critical assets |
| `prefer_high_leaseability` | `bool` | True | Favor easily re-leased assets |
| `weight_criticality` | `float` | 1.0 | Weight in scoring function |
| `weight_leaseability` | `float` | 1.0 | Weight in scoring function |

#### 4.4.3 AssetFilters

Pre-selection filters applied before scoring. LLM may relax these during revision.

| Field | Type | Default | Semantics |
|-------|------|---------|-----------|
| `include_asset_types` | `Optional[List[AssetType]]` | None | Whitelist (None = all) |
| `exclude_asset_types` | `Optional[List[AssetType]]` | None | Blacklist |
| `exclude_markets` | `Optional[List[str]]` | None | Markets to exclude |
| `min_leaseability_score` | `Optional[float]` | None | Floor for eligibility |
| `max_criticality` | `Optional[float]` | None | Ceiling for eligibility |

#### 4.4.4 Full SelectorSpec

| Field | Type | Required |
|-------|------|----------|
| `program_type` | `ProgramType` | Yes |
| `objective` | `Objective` | Yes |
| `target_amount` | `float` | Yes |
| `hard_constraints` | `HardConstraints` | Yes |
| `soft_preferences` | `SoftPreferences` | Yes |
| `asset_filters` | `AssetFilters` | Yes |
| `time_horizon_years` | `Optional[int]` | No |
| `max_iterations` | `int` | Yes (default: 3) |

**Semantic Notes:**
- `target_amount` is the target SLB proceeds in dollars
- Target is met when `proceeds >= target_amount * (1 - TOLERANCE)` where `TOLERANCE = 0.02`
- `max_iterations` bounds the agentic revision loop

### 4.5 Outcome Models

#### 4.5.1 ConstraintViolation

| Field | Type | Description |
|-------|------|-------------|
| `code` | `str` | Constraint identifier (e.g., "MAX_NET_LEVERAGE") |
| `detail` | `str` | Human-readable explanation |
| `actual` | `float` | Computed value |
| `limit` | `float` | Constraint threshold |

**Standard Codes:**
- `MAX_NET_LEVERAGE` — Post-SLB leverage exceeds limit
- `MIN_COVERAGE` — Post-SLB coverage below minimum
- `MAX_CRITICAL_FRACTION` — Critical asset concentration too high
- `TARGET_NOT_MET` — Insufficient proceeds even with all eligible assets

#### 4.5.2 AssetSelection

| Field | Type | Description |
|-------|------|-------------|
| `asset` | `Asset` | The selected asset |
| `proceeds` | `float` | SLB proceeds from this asset |
| `slb_rent` | `float` | Annual leaseback rent obligation |

#### 4.5.3 ProgramOutcome

| Field | Type | Description |
|-------|------|-------------|
| `status` | `SelectionStatus` | ok / infeasible / numeric_error |
| `selected_assets` | `List[AssetSelection]` | Assets in the SLB pool |
| `proceeds` | `float` | Total SLB proceeds |
| `leverage_before` | `float` | Pre-transaction net leverage |
| `leverage_after` | `float` | Post-transaction net leverage |
| `interest_coverage_before` | `Optional[float]` | Pre-transaction interest coverage (EBITDA / Interest) |
| `interest_coverage_after` | `Optional[float]` | Post-transaction interest coverage |
| `fixed_charge_coverage_before` | `Optional[float]` | Pre-transaction fixed charge coverage (EBITDA / (Interest + Leases)) |
| `fixed_charge_coverage_after` | `Optional[float]` | Post-transaction fixed charge coverage |
| `critical_fraction` | `float` | Critical NOI / Total selected NOI |
| `violations` | `List[ConstraintViolation]` | Constraint violations (empty if OK) |
| `warnings` | `List[str]` | Non-fatal warnings (e.g., "surplus proceeds ignored") |

**Nullable Metrics:**
Coverage metrics are `Optional[float]` because they become undefined when the denominator is near zero. When `None`, the UI/LLM layer should display "N/M" (not meaningful) rather than a numeric value.

### 4.6 Explanation

Structured explanation data returned by the engine. The LLM/UI layer renders this into human-readable text.

#### 4.6.1 ExplanationNode

A single explanation item with machine-readable metadata.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique identifier for this node |
| `label` | `str` | Human-readable label |
| `severity` | `Literal["info", "warning", "error"]` | Severity level |
| `category` | `Literal["constraint", "driver", "risk", "alternative"]` | Node category |
| `metric` | `Optional[str]` | Related metric name (e.g., "fixed_charge_coverage") |
| `baseline_value` | `Optional[float]` | Pre-transaction value |
| `post_value` | `Optional[float]` | Post-transaction value |
| `threshold` | `Optional[float]` | Constraint threshold if applicable |
| `asset_ids` | `Optional[List[str]]` | Related asset IDs |
| `detail` | `Optional[str]` | Additional context |

#### 4.6.2 Explanation

| Field | Type | Description |
|-------|------|-------------|
| `summary` | `str` | 2-3 sentence executive summary (generated by LLM from nodes) |
| `nodes` | `List[ExplanationNode]` | Structured explanation items |

**Node Categories:**
- `constraint`: Binding constraints that limited the solution
- `driver`: Key factors in asset selection
- `risk`: Risk factors in the selected pool
- `alternative`: Alternatives considered or rejected

**Usage Pattern:**
1. Engine computes `ExplanationNode` list from metrics and selection logic
2. LLM generates `summary` from the structured nodes
3. UI can render nodes as bullet lists, tooltips, or expandable sections
4. Nodes remain stable when math evolves (only labels/thresholds change)

### 4.7 API Models

#### ProgramRequest

| Field | Type | Required |
|-------|------|----------|
| `assets` | `List[Asset]` | Yes |
| `corporate_state` | `CorporateState` | Yes |
| `program_type` | `ProgramType` | Yes |
| `program_description` | `str` | Yes |

#### ProgramResponse

| Field | Type |
|-------|------|
| `selector_spec` | `SelectorSpec` |
| `outcome` | `ProgramOutcome` |
| `explanation` | `Explanation` |

### 4.8 EngineConfig

Configuration object for all economic parameters and thresholds. Passed explicitly to engine functions rather than hardcoded.

```python
class EngineConfig(BaseModel):
    """
    All configurable parameters for the economics engine.
    Loaded from JSON/YAML or passed via API for scenario testing.
    """

    # Cap rate curve by asset type and market tier
    cap_rate_curve: Dict[AssetType, Dict[MarketTier, float]]

    # Default market tier when not specified on asset
    default_market_tier: MarketTier = MarketTier.TIER_2

    # Transaction costs
    transaction_haircut: float = 0.025  # 2.5% of market value

    # Debt assumptions
    avg_cost_of_debt: float = 0.06  # 6% blended cost of debt

    # SLB rent modeling
    slb_rent_multiplier: float = 1.0  # slb_rent = noi * multiplier (0.9-1.2 typical)

    # Selection tolerance
    target_tolerance: float = 0.02  # 2% under-target is acceptable

    # Criticality threshold for "critical asset" classification
    criticality_threshold: float = 0.7

    # Numeric guardrails
    epsilon: float = 1e-9  # Denominators below this are treated as zero

    # Default hard constraints (used when not specified in SelectorSpec)
    default_max_net_leverage: float = 4.0
    default_min_fixed_charge_coverage: float = 3.0
    default_min_interest_coverage: Optional[float] = None

    # Iteration defaults
    default_max_iterations: int = 3
```

**Benefits:**
- Scenarios can vary parameters without code changes (e.g., "what if avg_cost_of_debt = 7%?")
- Unit tests can inject different configs to test edge cases
- Clients with different covenant regimes can customize thresholds
- `slb_rent_multiplier` allows modeling aggressive (< 1.0) vs conservative (> 1.0) SLB terms

---

## 5. SLB Economics Model

All economics logic lives in `engine.py`. These are **demo simplifications**, not GAAP-accurate.

### 5.1 Default Configuration (`config.py`)

All parameters are now defined in `EngineConfig` (Section 4.8). The `config.py` module provides a default instance:

```python
DEFAULT_CAP_RATE_CURVE: Dict[AssetType, Dict[MarketTier, float]] = {
    AssetType.STORE: {
        MarketTier.TIER_1: 0.055,
        MarketTier.TIER_2: 0.065,
        MarketTier.TIER_3: 0.075,
    },
    AssetType.DISTRIBUTION_CENTER: {
        MarketTier.TIER_1: 0.045,
        MarketTier.TIER_2: 0.055,
        MarketTier.TIER_3: 0.065,
    },
    AssetType.OFFICE: {
        MarketTier.TIER_1: 0.060,
        MarketTier.TIER_2: 0.070,
        MarketTier.TIER_3: 0.080,
    },
    AssetType.MIXED_USE: {
        MarketTier.TIER_1: 0.055,
        MarketTier.TIER_2: 0.065,
        MarketTier.TIER_3: 0.075,
    },
    AssetType.OTHER: {
        MarketTier.TIER_1: 0.070,
        MarketTier.TIER_2: 0.080,
        MarketTier.TIER_3: 0.090,
    },
}

DEFAULT_ENGINE_CONFIG = EngineConfig(
    cap_rate_curve=DEFAULT_CAP_RATE_CURVE,
    default_market_tier=MarketTier.TIER_2,
    transaction_haircut=0.025,
    avg_cost_of_debt=0.06,
    slb_rent_multiplier=1.0,
    target_tolerance=0.02,
    criticality_threshold=0.7,
    epsilon=1e-9,
    default_max_net_leverage=4.0,
    default_min_fixed_charge_coverage=3.0,
    default_min_interest_coverage=None,
    default_max_iterations=3,
)
```

**Note:** All engine functions accept `config: EngineConfig` as an explicit parameter. This enables:
- Scenario analysis with varied assumptions
- Unit testing with controlled parameters
- Client-specific covenant configurations

### 5.2 Per-Asset SLB Metrics

```python
def compute_asset_slb_metrics(
    asset: Asset,
    config: EngineConfig,
) -> AssetSLBMetrics:
    tier = asset.market_tier or config.default_market_tier
    cap_rate = config.cap_rate_curve[asset.asset_type][tier]

    market_value = asset.noi / cap_rate
    proceeds = market_value * (1 - config.transaction_haircut)

    # SLB rent uses multiplier to allow aggressive/conservative modeling
    slb_rent = asset.noi * config.slb_rent_multiplier

    return AssetSLBMetrics(
        market_value=market_value,
        proceeds=proceeds,
        slb_rent=slb_rent,
        cap_rate=cap_rate,
    )
```

**SLB Rent Modeling:**
- `slb_rent = noi * slb_rent_multiplier`
- With `multiplier = 1.0`: rent equals current NOI (baseline assumption)
- With `multiplier < 1.0`: aggressive SLB terms (rent below NOI, favorable to seller)
- With `multiplier > 1.0`: conservative SLB terms (rent above NOI, e.g., escalations priced in)

**Why not `rent = market_value × cap_rate`?**
That formula produces `rent = noi` by construction, which conflates "current NOI," "market rent," and "negotiated lease payment." The multiplier approach:
1. Preserves the baseline (multiplier = 1.0 gives rent ≈ NOI)
2. Provides a lever for scenario analysis
3. Makes the assumption explicit rather than hiding it in circular math

### 5.3 Portfolio-Level Metrics

#### Before Transaction (from CorporateState)

```python
def compute_baseline_metrics(
    state: CorporateState,
    config: EngineConfig,
) -> BaselineMetrics:
    """Compute pre-transaction metrics with numeric guardrails."""

    # Leverage
    if abs(state.ebitda) < config.epsilon:
        leverage_before = None  # Not meaningful
    else:
        leverage_before = state.net_debt / state.ebitda

    # Interest coverage
    if abs(state.interest_expense) < config.epsilon:
        interest_coverage_before = None  # Not meaningful (no interest)
    else:
        interest_coverage_before = state.ebitda / state.interest_expense

    # Fixed charge coverage
    total_fixed_charges = state.interest_expense + (state.lease_expense or 0)
    if abs(total_fixed_charges) < config.epsilon:
        fixed_charge_coverage_before = None  # Not meaningful
    else:
        fixed_charge_coverage_before = state.ebitda / total_fixed_charges

    return BaselineMetrics(
        leverage=leverage_before,
        interest_coverage=interest_coverage_before,
        fixed_charge_coverage=fixed_charge_coverage_before,
    )
```

#### After Transaction

Given selected assets with total proceeds and total slb_rent:

```python
def compute_post_transaction_metrics(
    state: CorporateState,
    selected: List[AssetSelection],
    config: EngineConfig,
) -> Tuple[PostMetrics, List[str]]:
    """
    Compute post-transaction metrics.
    Returns (metrics, warnings) where warnings captures any guardrail triggers.
    """
    warnings: List[str] = []

    total_proceeds = sum(a.proceeds for a in selected)
    total_slb_rent = sum(a.slb_rent for a in selected)

    # Debt repayment with guardrails
    debt_repaid = total_proceeds  # assume 100% to debt paydown

    # Guard: cannot repay more debt than exists
    if debt_repaid > state.net_debt:
        warnings.append(
            f"Proceeds (${debt_repaid:,.0f}) exceed net_debt (${state.net_debt:,.0f}); "
            f"surplus ${debt_repaid - state.net_debt:,.0f} ignored for debt paydown"
        )
        debt_repaid = state.net_debt

    net_debt_after = state.net_debt - debt_repaid

    # Interest reduction with clamping
    interest_reduction = debt_repaid * config.avg_cost_of_debt
    interest_after = max(0, state.interest_expense - interest_reduction)

    if state.interest_expense - interest_reduction < 0:
        warnings.append(
            f"Interest reduction (${interest_reduction:,.0f}) exceeds interest_expense "
            f"(${state.interest_expense:,.0f}); clamped to zero"
        )

    # EBITDA unchanged (demo simplification)
    ebitda = state.ebitda

    # Lease expense
    lease_expense_after = (state.lease_expense or 0) + total_slb_rent

    # Compute metrics with guardrails
    if abs(ebitda) < config.epsilon:
        leverage_after = None
        interest_coverage_after = None
        fixed_charge_coverage_after = None
    else:
        leverage_after = net_debt_after / ebitda

        if abs(interest_after) < config.epsilon:
            interest_coverage_after = None  # No interest remaining
        else:
            interest_coverage_after = ebitda / interest_after

        total_fixed_charges = interest_after + lease_expense_after
        if abs(total_fixed_charges) < config.epsilon:
            fixed_charge_coverage_after = None
        else:
            fixed_charge_coverage_after = ebitda / total_fixed_charges

    return PostMetrics(
        net_debt_after=net_debt_after,
        interest_after=interest_after,
        lease_expense_after=lease_expense_after,
        leverage=leverage_after,
        interest_coverage=interest_coverage_after,
        fixed_charge_coverage=fixed_charge_coverage_after,
        proceeds=total_proceeds,
    ), warnings
```

#### Critical Fraction

```python
def compute_critical_fraction(
    selected: List[AssetSelection],
    config: EngineConfig,
) -> float:
    critical_noi = sum(
        a.asset.noi for a in selected
        if a.asset.criticality > config.criticality_threshold
    )
    total_noi = sum(a.asset.noi for a in selected)

    if abs(total_noi) < config.epsilon:
        return 0.0  # No assets selected or all zero NOI
    return critical_noi / total_noi
```

#### Numeric Guardrails Summary

| Condition | Handling |
|-----------|----------|
| `ebitda ≈ 0` | Leverage and coverage metrics return `None` |
| `interest_expense ≈ 0` | Interest coverage returns `None` |
| `interest_after + lease_expense ≈ 0` | Fixed charge coverage returns `None` |
| `debt_repaid > net_debt` | Cap at `net_debt`, emit warning |
| `interest_reduction > interest_expense` | Clamp to zero, emit warning |
| `total_noi ≈ 0` | Critical fraction returns `0.0` |

**Note:** `None` metrics should be displayed as "N/M" (not meaningful) in the UI and excluded from constraint checks.

### 5.4 Documented Simplifications

| Simplification | Reality | Demo Assumption |
|----------------|---------|-----------------|
| EBITDA unchanged | SLB removes asset OpEx; rent hits P&L | EBITDA constant (see note below) |
| Full debt paydown | May have other uses | 100% to debt (surplus capped) |
| Uniform cost of debt | Different tranches (revolver, term, bonds) | Single blended rate |
| SLB rent via multiplier | Negotiated rates, escalations | `noi × multiplier` (default 1.0) |
| No tax effects | Complex tax treatment | Ignored |

**EBITDA Treatment Note:**

This model holds EBITDA constant pre/post transaction. This is **not GAAP-accurate**:

- In reality, SLB rent would appear as an operating expense, reducing EBITDA
- Some lenders use EBITDAR (EBITDA before rent) for SLB-heavy companies
- Our fixed-charge coverage metric partially captures this by adding rent to the denominator

**Implications:**
- Coverage metrics in this model are "demo coverage," not credit-agreement coverage
- The model may overstate pain (rent in denominator without EBITDA offset) or understate it (no EBITDA reduction)
- For production use, decide whether to:
  1. Reduce EBITDA by incremental rent: `ebitda_after = ebitda - slb_rent`
  2. Use EBITDAR-based metrics explicitly
  3. Keep the current approach but label metrics clearly (e.g., "Pro Forma Coverage (Demo)")

**Current choice:** Option 3 — keep EBITDA constant, document the limitation, label metrics appropriately.

---

## 6. Selection Algorithm (Engine)

Implemented in `engine.py`. Greedy heuristic — not globally optimal.

### 6.1 Scoring Function

```python
def compute_score(
    asset: Asset,
    preferences: SoftPreferences,
) -> float:
    """Higher score = more preferred for selection."""
    score = 0.0

    # Criticality: prefer low (score decreases with criticality)
    if preferences.prefer_low_criticality:
        score -= preferences.weight_criticality * asset.criticality

    # Leaseability: prefer high (score increases with leaseability)
    if preferences.prefer_high_leaseability:
        score += preferences.weight_leaseability * asset.leaseability_score

    return score
```

### 6.2 Filter Application

```python
def apply_filters(
    assets: List[Asset],
    filters: AssetFilters,
) -> List[Asset]:
    eligible = assets

    if filters.include_asset_types:
        eligible = [a for a in eligible if a.asset_type in filters.include_asset_types]

    if filters.exclude_asset_types:
        eligible = [a for a in eligible if a.asset_type not in filters.exclude_asset_types]

    if filters.exclude_markets:
        eligible = [a for a in eligible if a.market not in filters.exclude_markets]

    if filters.min_leaseability_score is not None:
        eligible = [a for a in eligible if a.leaseability_score >= filters.min_leaseability_score]

    if filters.max_criticality is not None:
        eligible = [a for a in eligible if a.criticality <= filters.max_criticality]

    return eligible
```

### 6.3 Constraint Checking

```python
def check_constraints(
    selected: List[AssetSelection],
    corporate_state: CorporateState,
    hard_constraints: HardConstraints,
    target_amount: float,
    config: EngineConfig,
) -> Tuple[PortfolioMetrics, List[ConstraintViolation], List[str]]:
    """
    Compute metrics and check all hard constraints.
    Returns (metrics, violations, warnings).
    """
    baseline = compute_baseline_metrics(corporate_state, config)
    post_metrics, warnings = compute_post_transaction_metrics(
        corporate_state, selected, config
    )
    critical_fraction = compute_critical_fraction(selected, config)

    violations = []

    # Check leverage (skip if metric is None / not meaningful)
    if hard_constraints.max_net_leverage is not None:
        if post_metrics.leverage is None:
            violations.append(ConstraintViolation(
                code="MAX_NET_LEVERAGE",
                detail="Post-SLB leverage undefined (EBITDA ≈ 0)",
                actual=float('nan'),
                limit=hard_constraints.max_net_leverage,
            ))
        elif post_metrics.leverage > hard_constraints.max_net_leverage:
            violations.append(ConstraintViolation(
                code="MAX_NET_LEVERAGE",
                detail=f"Post-SLB leverage {post_metrics.leverage:.2f}x exceeds limit",
                actual=post_metrics.leverage,
                limit=hard_constraints.max_net_leverage,
            ))

    # Check interest coverage
    if hard_constraints.min_interest_coverage is not None:
        if post_metrics.interest_coverage is None:
            # Interest ≈ 0 means infinite coverage; not a violation
            pass
        elif post_metrics.interest_coverage < hard_constraints.min_interest_coverage:
            violations.append(ConstraintViolation(
                code="MIN_INTEREST_COVERAGE",
                detail=f"Post-SLB interest coverage {post_metrics.interest_coverage:.2f}x below minimum",
                actual=post_metrics.interest_coverage,
                limit=hard_constraints.min_interest_coverage,
            ))

    # Check fixed charge coverage
    if hard_constraints.min_fixed_charge_coverage is not None:
        if post_metrics.fixed_charge_coverage is None:
            violations.append(ConstraintViolation(
                code="MIN_FIXED_CHARGE_COVERAGE",
                detail="Post-SLB fixed charge coverage undefined (denominator ≈ 0)",
                actual=float('nan'),
                limit=hard_constraints.min_fixed_charge_coverage,
            ))
        elif post_metrics.fixed_charge_coverage < hard_constraints.min_fixed_charge_coverage:
            violations.append(ConstraintViolation(
                code="MIN_FIXED_CHARGE_COVERAGE",
                detail=f"Post-SLB fixed charge coverage {post_metrics.fixed_charge_coverage:.2f}x below minimum",
                actual=post_metrics.fixed_charge_coverage,
                limit=hard_constraints.min_fixed_charge_coverage,
            ))

    # Check critical fraction
    if hard_constraints.max_critical_fraction is not None:
        if critical_fraction > hard_constraints.max_critical_fraction:
            violations.append(ConstraintViolation(
                code="MAX_CRITICAL_FRACTION",
                detail=f"Critical asset concentration {critical_fraction:.1%} exceeds limit",
                actual=critical_fraction,
                limit=hard_constraints.max_critical_fraction,
            ))

    # Check target met
    if post_metrics.proceeds < target_amount * (1 - config.target_tolerance):
        violations.append(ConstraintViolation(
            code="TARGET_NOT_MET",
            detail=f"Proceeds ${post_metrics.proceeds:,.0f} below target ${target_amount:,.0f}",
            actual=post_metrics.proceeds,
            limit=target_amount * (1 - config.target_tolerance),
        ))

    # Assemble full metrics object
    metrics = PortfolioMetrics(
        leverage_before=baseline.leverage,
        leverage_after=post_metrics.leverage,
        interest_coverage_before=baseline.interest_coverage,
        interest_coverage_after=post_metrics.interest_coverage,
        fixed_charge_coverage_before=baseline.fixed_charge_coverage,
        fixed_charge_coverage_after=post_metrics.fixed_charge_coverage,
        critical_fraction=critical_fraction,
        proceeds=post_metrics.proceeds,
    )

    return metrics, violations, warnings
```

**Constraint Codes:**
- `MAX_NET_LEVERAGE` — Post-SLB leverage exceeds limit
- `MIN_INTEREST_COVERAGE` — Post-SLB interest coverage below minimum
- `MIN_FIXED_CHARGE_COVERAGE` — Post-SLB fixed charge coverage below minimum
- `MAX_CRITICAL_FRACTION` — Critical asset concentration too high
- `TARGET_NOT_MET` — Insufficient proceeds even with all eligible assets
- `NO_ELIGIBLE_ASSETS` — No assets pass the filter criteria

### 6.4 Selection Algorithm

```python
def select_assets(
    assets: List[Asset],
    corporate_state: CorporateState,
    spec: SelectorSpec,
    config: EngineConfig,
) -> ProgramOutcome:
    """
    Greedy selection: add assets in score order until target met,
    skipping any that would violate hard constraints.

    This is the ONLY entry point for metric computation. API/UI/LLM layers
    must not recompute metrics — they use the values from ProgramOutcome.
    """

    # Step 1: Compute baseline metrics (needed for "no eligible assets" case)
    baseline = compute_baseline_metrics(corporate_state, config)

    # Step 2: Apply filters
    eligible = apply_filters(assets, spec.asset_filters)

    if not eligible:
        return ProgramOutcome(
            status=SelectionStatus.INFEASIBLE,
            selected_assets=[],
            proceeds=0,
            leverage_before=baseline.leverage,
            leverage_after=baseline.leverage,
            interest_coverage_before=baseline.interest_coverage,
            interest_coverage_after=baseline.interest_coverage,
            fixed_charge_coverage_before=baseline.fixed_charge_coverage,
            fixed_charge_coverage_after=baseline.fixed_charge_coverage,
            critical_fraction=0,
            violations=[ConstraintViolation(
                code="NO_ELIGIBLE_ASSETS",
                detail="No assets pass the filter criteria",
                actual=0,
                limit=1,
            )],
            warnings=[],
        )

    # Step 3: Compute SLB metrics and scores for eligible assets
    candidates = []
    for asset in eligible:
        slb_metrics = compute_asset_slb_metrics(asset, config)
        score = compute_score(asset, spec.soft_preferences)
        candidates.append((asset, slb_metrics, score))

    # Step 4: Sort by score descending
    candidates.sort(key=lambda x: x[2], reverse=True)

    # Step 5: Greedy selection
    selected: List[AssetSelection] = []
    all_warnings: List[str] = []

    for asset, slb_metrics, score in candidates:
        # Tentatively add
        tentative = selected + [AssetSelection(
            asset=asset,
            proceeds=slb_metrics.proceeds,
            slb_rent=slb_metrics.slb_rent,
        )]

        # Check constraints (except target, which we're trying to reach)
        metrics, violations, warnings = check_constraints(
            tentative,
            corporate_state,
            spec.hard_constraints,
            target_amount=0,  # Don't check target yet
            config=config,
        )

        # Skip if adding this asset violates leverage/coverage/critical
        non_target_violations = [v for v in violations if v.code != "TARGET_NOT_MET"]
        if non_target_violations:
            continue  # Skip this asset

        # Add asset
        selected = tentative
        all_warnings = warnings  # Keep latest warnings

        # Check if target met
        if metrics.proceeds >= spec.target_amount * (1 - config.target_tolerance):
            break

    # Step 6: Final constraint check with target
    final_metrics, final_violations, final_warnings = check_constraints(
        selected,
        corporate_state,
        spec.hard_constraints,
        spec.target_amount,
        config=config,
    )
    all_warnings = final_warnings

    # Step 7: Determine status
    # NUMERIC_ERROR only for truly unexpected NaN/Inf in proceeds
    # None metrics are expected and handled gracefully
    if final_metrics.proceeds is not None and (
        math.isnan(final_metrics.proceeds) or math.isinf(final_metrics.proceeds)
    ):
        status = SelectionStatus.NUMERIC_ERROR
    elif final_violations:
        status = SelectionStatus.INFEASIBLE
    else:
        status = SelectionStatus.OK

    return ProgramOutcome(
        status=status,
        selected_assets=selected,
        proceeds=final_metrics.proceeds,
        leverage_before=final_metrics.leverage_before,
        leverage_after=final_metrics.leverage_after,
        interest_coverage_before=final_metrics.interest_coverage_before,
        interest_coverage_after=final_metrics.interest_coverage_after,
        fixed_charge_coverage_before=final_metrics.fixed_charge_coverage_before,
        fixed_charge_coverage_after=final_metrics.fixed_charge_coverage_after,
        critical_fraction=final_metrics.critical_fraction,
        violations=final_violations,
        warnings=all_warnings,
    )
```

### 6.5 Algorithm Notes

| Aspect | V1 Approach | V2 Consideration |
|--------|-------------|------------------|
| Optimality | Greedy heuristic | MILP/OR-Tools for global optimum |
| Tie-breaking | Arbitrary (list order) | Secondary sort by proceeds/asset |
| Backtracking | None | Consider if first choice blocks better solutions |
| Multi-objective | Single score | Pareto frontier exploration |

---

## 7. Validation and Defaults

### 7.1 Input Invariant Validation

Validate inputs at the system boundary to catch garbage-in before it produces garbage-out.

#### 7.1.1 Asset Validation

```python
def validate_asset(asset: Asset) -> List[str]:
    """Validate a single asset. Returns list of errors."""
    errors = []

    if asset.noi < 0:
        errors.append(f"Asset {asset.asset_id}: noi must be >= 0 (got {asset.noi})")

    if asset.book_value <= 0:
        errors.append(f"Asset {asset.asset_id}: book_value must be > 0 (got {asset.book_value})")

    if not (0 <= asset.criticality <= 1):
        errors.append(f"Asset {asset.asset_id}: criticality must be in [0, 1] (got {asset.criticality})")

    if not (0 <= asset.leaseability_score <= 1):
        errors.append(f"Asset {asset.asset_id}: leaseability_score must be in [0, 1] (got {asset.leaseability_score})")

    return errors


def validate_assets(assets: List[Asset]) -> List[str]:
    """Validate asset list. Returns list of errors."""
    errors = []

    if not assets:
        errors.append("Asset list cannot be empty")
        return errors

    # Check for duplicate IDs
    ids = [a.asset_id for a in assets]
    if len(ids) != len(set(ids)):
        errors.append("Asset IDs must be unique")

    # Validate each asset
    for asset in assets:
        errors.extend(validate_asset(asset))

    return errors
```

#### 7.1.2 CorporateState Validation

```python
def validate_corporate_state(state: CorporateState) -> List[str]:
    """Validate corporate state. Returns list of errors."""
    errors = []

    if state.net_debt < 0:
        errors.append(f"net_debt must be >= 0 (got {state.net_debt})")

    # EBITDA can be negative (distressed company), but warn
    # No hard error, but downstream metrics will be None

    if state.interest_expense < 0:
        errors.append(f"interest_expense must be >= 0 (got {state.interest_expense})")

    if state.lease_expense is not None and state.lease_expense < 0:
        errors.append(f"lease_expense must be >= 0 (got {state.lease_expense})")

    return errors
```

### 7.2 Spec Validation (`validate_spec`)

```python
def validate_spec(spec: SelectorSpec) -> List[str]:
    """
    Returns list of validation errors. Empty list = valid.
    """
    errors = []

    # Target amount
    if spec.target_amount <= 0:
        errors.append("target_amount must be positive")

    # Hard constraints ranges
    hc = spec.hard_constraints
    if hc.max_net_leverage is not None:
        if not (0 < hc.max_net_leverage < 10):
            errors.append("max_net_leverage must be in (0, 10)")

    if hc.min_interest_coverage is not None:
        if not (0 < hc.min_interest_coverage < 50):
            errors.append("min_interest_coverage must be in (0, 50)")

    if hc.min_fixed_charge_coverage is not None:
        if not (0 < hc.min_fixed_charge_coverage < 20):
            errors.append("min_fixed_charge_coverage must be in (0, 20)")

    if hc.max_critical_fraction is not None:
        if not (0 < hc.max_critical_fraction <= 1):
            errors.append("max_critical_fraction must be in (0, 1]")

    # Soft preferences weights
    sp = spec.soft_preferences
    if sp.weight_criticality < 0:
        errors.append("weight_criticality must be non-negative")
    if sp.weight_leaseability < 0:
        errors.append("weight_leaseability must be non-negative")

    # Filter bounds
    af = spec.asset_filters
    if af.min_leaseability_score is not None:
        if not (0 <= af.min_leaseability_score <= 1):
            errors.append("min_leaseability_score must be in [0, 1]")

    if af.max_criticality is not None:
        if not (0 <= af.max_criticality <= 1):
            errors.append("max_criticality must be in [0, 1]")

    # Max iterations
    if spec.max_iterations < 1:
        errors.append("max_iterations must be at least 1")
    if spec.max_iterations > 10:
        errors.append("max_iterations should not exceed 10")

    return errors
```

### 7.3 Engine Validation

The engine now handles edge cases gracefully rather than throwing errors:

| Condition | Previous Behavior | New Behavior |
|-----------|-------------------|--------------|
| `ebitda ≈ 0` | `NUMERIC_ERROR` | Metric returns `None`, continues |
| `interest ≈ 0` | Division error | Interest coverage returns `None` |
| `proceeds = NaN/Inf` | `NUMERIC_ERROR` | Still `NUMERIC_ERROR` (unexpected) |
| `debt_repaid > net_debt` | Negative net_debt | Cap and emit warning |
| `interest_reduction > interest` | Negative interest | Clamp to zero, emit warning |

**NUMERIC_ERROR** is now reserved for truly unexpected conditions (e.g., NaN proceeds), not for expected edge cases like zero denominators.

### 7.4 Default Values

All defaults are now defined in `EngineConfig` (Section 4.8). For quick reference:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `default_max_net_leverage` | 4.0 | Maximum post-SLB leverage |
| `default_min_fixed_charge_coverage` | 3.0 | Minimum fixed charge coverage |
| `default_min_interest_coverage` | None | Not enforced by default |
| `target_tolerance` | 0.02 | 2% under-target acceptable |
| `slb_rent_multiplier` | 1.0 | Rent = NOI × multiplier |
| `avg_cost_of_debt` | 0.06 | 6% blended rate |
| `criticality_threshold` | 0.7 | Threshold for "critical" assets |
| `default_max_iterations` | 3 | Max revision loop iterations |

### 7.5 Asset Summary for LLM

```python
def summarize_assets(assets: List[Asset]) -> str:
    """
    Create a summary for LLM context. No raw data dump.
    """
    total_noi = sum(a.noi for a in assets)
    avg_cap_rate = 0.065  # approximate
    est_market_value = total_noi / avg_cap_rate

    by_type = Counter(a.asset_type for a in assets)
    avg_criticality = sum(a.criticality for a in assets) / len(assets)
    avg_leaseability = sum(a.leaseability_score for a in assets) / len(assets)

    return f"""Portfolio Summary:
- {len(assets)} assets
- Total NOI: ${total_noi:,.0f}
- Estimated market value: ${est_market_value:,.0f}
- Type breakdown: {dict(by_type)}
- Average criticality: {avg_criticality:.2f}
- Average leaseability: {avg_leaseability:.2f}
"""
```

---

## 8. Agentic Loop and Revision Policy

Implemented in `orchestrator.py`.

### 8.1 Main Orchestration Function

```python
async def run_program(request: ProgramRequest) -> ProgramResponse:
    """
    Main entry point. Orchestrates spec generation, engine runs,
    revision loop, and explanation generation.
    """

    # 1. Validate program type
    if request.program_type != ProgramType.SLB:
        raise ValueError(f"program_type '{request.program_type}' not supported in v1")

    # 2. Summarize assets for LLM context
    asset_summary = summarize_assets(request.assets)

    # 3. Generate initial SelectorSpec via LLM
    initial_spec = await generate_selector_spec(
        program_type=request.program_type,
        program_description=request.program_description,
        asset_summary=asset_summary,
    )

    # 4. Validate spec
    validation_errors = validate_spec(initial_spec)
    if validation_errors:
        raise SpecValidationError(validation_errors)

    # 5. Capture immutable hard constraints
    immutable_hard = initial_spec.hard_constraints.model_copy(deep=True)
    original_target = initial_spec.target_amount

    # 6. Agentic loop
    current_spec = initial_spec
    outcome: Optional[ProgramOutcome] = None

    for attempt in range(current_spec.max_iterations):
        # Run engine
        outcome = select_assets(
            assets=request.assets,
            corporate_state=request.corporate_state,
            spec=current_spec,
        )

        # Check result
        if outcome.status == SelectionStatus.OK:
            break

        if outcome.status == SelectionStatus.NUMERIC_ERROR:
            raise EngineError("Numeric error in selection engine")

        # INFEASIBLE: attempt revision
        if attempt < current_spec.max_iterations - 1:
            revised_spec = await revise_selector_spec(
                original_description=request.program_description,
                previous_spec=current_spec,
                outcome=outcome,
            )

            # Enforce revision policy
            policy_result = enforce_revision_policy(
                immutable_hard=immutable_hard,
                original_target=original_target,
                prev_spec=current_spec,
                new_spec=revised_spec,
            )

            if not policy_result.valid:
                # Policy violation: stop loop, return last infeasible
                break

            current_spec = policy_result.spec

    # 7. Generate explanation
    explanation = await generate_explanation(
        spec=current_spec,
        outcome=outcome,
        original_description=request.program_description,
    )

    # 8. Return response
    return ProgramResponse(
        selector_spec=current_spec,
        outcome=outcome,
        explanation=explanation,
    )
```

### 8.2 Revision Policy

```python
@dataclass
class PolicyResult:
    valid: bool
    spec: Optional[SelectorSpec]
    violations: List[str]


def enforce_revision_policy(
    immutable_hard: HardConstraints,
    original_target: float,
    prev_spec: SelectorSpec,
    new_spec: SelectorSpec,
) -> PolicyResult:
    """
    Enforce constraints on what the LLM may change during revision.
    Returns a PolicyResult with potentially adjusted spec.
    """
    violations = []
    adjusted = new_spec.model_copy(deep=True)

    # === IMMUTABLE CHECKS ===

    # Program type must not change
    if new_spec.program_type != prev_spec.program_type:
        violations.append("Cannot change program_type")
        return PolicyResult(valid=False, spec=None, violations=violations)

    # Hard constraints: cannot relax beyond immutable
    if new_spec.hard_constraints.max_net_leverage is not None:
        if immutable_hard.max_net_leverage is not None:
            if new_spec.hard_constraints.max_net_leverage > immutable_hard.max_net_leverage:
                violations.append(
                    f"Cannot increase max_net_leverage beyond {immutable_hard.max_net_leverage}"
                )
                adjusted.hard_constraints.max_net_leverage = immutable_hard.max_net_leverage

    if new_spec.hard_constraints.min_fixed_charge_coverage is not None:
        if immutable_hard.min_fixed_charge_coverage is not None:
            if new_spec.hard_constraints.min_fixed_charge_coverage < immutable_hard.min_fixed_charge_coverage:
                violations.append(
                    f"Cannot decrease min_fixed_charge_coverage below {immutable_hard.min_fixed_charge_coverage}"
                )
                adjusted.hard_constraints.min_fixed_charge_coverage = immutable_hard.min_fixed_charge_coverage

    # === MONOTONICITY CHECKS ===

    # Target amount: must decrease or stay same
    if new_spec.target_amount > prev_spec.target_amount:
        violations.append("target_amount cannot increase")
        adjusted.target_amount = prev_spec.target_amount

    # Target amount: bounded per-iteration reduction (max 20% drop per iteration)
    min_allowed_target = prev_spec.target_amount * 0.80
    if new_spec.target_amount < min_allowed_target:
        violations.append(f"target_amount cannot drop more than 20% per iteration")
        adjusted.target_amount = min_allowed_target

    # Target amount: global floor (50% of original)
    global_floor = original_target * 0.50
    if adjusted.target_amount < global_floor:
        violations.append(f"target_amount cannot go below 50% of original (${global_floor:,.0f})")
        return PolicyResult(valid=False, spec=None, violations=violations)

    # === BOUNDED FILTER RELAXATION ===

    # max_criticality: can increase by at most 0.1 per iteration, never above 0.8
    if new_spec.asset_filters.max_criticality is not None:
        prev_crit = prev_spec.asset_filters.max_criticality or 0.5
        if new_spec.asset_filters.max_criticality > prev_crit + 0.1:
            adjusted.asset_filters.max_criticality = min(prev_crit + 0.1, 0.8)
        if adjusted.asset_filters.max_criticality > 0.8:
            adjusted.asset_filters.max_criticality = 0.8

    # min_leaseability_score: can decrease by at most 0.1 per iteration, never below 0.2
    if new_spec.asset_filters.min_leaseability_score is not None:
        prev_lease = prev_spec.asset_filters.min_leaseability_score or 0.5
        if new_spec.asset_filters.min_leaseability_score < prev_lease - 0.1:
            adjusted.asset_filters.min_leaseability_score = max(prev_lease - 0.1, 0.2)
        if adjusted.asset_filters.min_leaseability_score < 0.2:
            adjusted.asset_filters.min_leaseability_score = 0.2

    # If we made adjustments, spec is still valid but modified
    if violations:
        return PolicyResult(valid=True, spec=adjusted, violations=violations)

    return PolicyResult(valid=True, spec=new_spec, violations=[])
```

### 8.3 Revision Policy Summary Table

| Field | Allowed Change | Bounds |
|-------|----------------|--------|
| `program_type` | Never | Immutable |
| `hard_constraints.max_net_leverage` | Never relax | ≤ original |
| `hard_constraints.min_fixed_charge_coverage` | Never relax | ≥ original |
| `target_amount` | Decrease only | ≥ 80% prev, ≥ 50% original |
| `asset_filters.max_criticality` | Increase | +0.1/iter, max 0.8 |
| `asset_filters.min_leaseability_score` | Decrease | -0.1/iter, min 0.2 |
| `soft_preferences.*` | Any | Unrestricted |

---

## 9. LLM Prompting Strategy

Implemented in `llm.py`. Uses OpenAI SDK with structured outputs.

### 9.1 Model Configuration

```python
# config.py
LLM_MODEL = "gpt-4o-mini"  # or "gpt-4o" for production
LLM_TEMPERATURE = 0.2  # Low temperature for structured output
LLM_MAX_TOKENS = 2000
```

### 9.2 generate_selector_spec

**Purpose:** Translate natural-language brief into structured SelectorSpec.

**Inputs:**
- `program_type: ProgramType`
- `program_description: str`
- `asset_summary: str`

**Output:** `SelectorSpec`

**Prompt Pattern:**

```
System:
You are a commercial real estate funding analyst. Your task is to interpret a
program brief and produce a structured SelectorSpec for a {program_type} program.

IMPORTANT RULES:
1. Never perform calculations. You do not know exact portfolio metrics.
2. Use the asset_summary for context, but do not reference specific values.
3. Apply default values when the brief is vague.
4. Output valid JSON matching the SelectorSpec schema.

DEFAULTS (use when not specified):
- max_net_leverage: 4.0
- min_fixed_charge_coverage: 3.0
- target_amount: 20% of estimated portfolio market value (from summary)
- max_iterations: 3

User:
Program Type: {program_type}

Program Brief:
{program_description}

Asset Summary:
{asset_summary}

Generate a SelectorSpec in JSON format.
```

**Structured Output:**
Use OpenAI's `response_format` with JSON schema matching `SelectorSpec`.

### 9.3 revise_selector_spec

**Purpose:** Adjust spec to address infeasibility while respecting revision policy.

**Inputs:**
- `original_description: str`
- `previous_spec: SelectorSpec`
- `outcome: ProgramOutcome`

**Output:** `SelectorSpec`

**Prompt Pattern:**

```
System:
You are a commercial real estate funding analyst. The previous SelectorSpec
resulted in an infeasible outcome. Your task is to revise the spec to find
a feasible solution.

REVISION RULES:
1. You may REDUCE target_amount (never increase, max 20% reduction)
2. You may RELAX asset filters (increase max_criticality, decrease min_leaseability)
3. You may ADJUST soft preference weights
4. You may NOT change hard_constraints (leverage, coverage limits are immutable)
5. You may NOT change program_type

CONSTRAINT VIOLATIONS FROM PREVIOUS RUN:
{format_violations(outcome.violations)}

STRATEGY:
- If TARGET_NOT_MET: consider reducing target or relaxing filters
- If MAX_NET_LEVERAGE: cannot fix via spec (hard constraint)
- If MAX_CRITICAL_FRACTION: reduce max_criticality filter or reduce target

User:
Original Brief: {original_description}

Previous Spec:
{previous_spec.model_dump_json(indent=2)}

Previous Outcome:
- Status: {outcome.status}
- Proceeds: ${outcome.proceeds:,.0f}
- Leverage After: {outcome.leverage_after:.2f}x
- Interest Coverage After: {outcome.interest_coverage_after:.2f}x
- Fixed Charge Coverage After: {outcome.fixed_charge_coverage_after:.2f}x
- Violations: {outcome.violations}

Generate a revised SelectorSpec in JSON format.
```

### 9.4 generate_explanation

**Purpose:** Produce IC-style narrative explaining the outcome.

**Inputs:**
- `spec: SelectorSpec`
- `outcome: ProgramOutcome`
- `original_description: str`

**Output:** `Explanation`

**Prompt Pattern:**

```
System:
You are a commercial real estate investment committee analyst. Generate an
executive summary explaining the funding program outcome.

RULES:
1. All numeric values must come from the provided outcome. Do not calculate.
2. Be concise: summary should be 2-3 sentences.
3. Identify binding constraints (what limited the solution)
4. Explain key drivers (why these assets were selected)
5. Note risks in the selected pool
6. Mention alternatives considered if the outcome is infeasible

User:
Original Brief: {original_description}

Final Spec:
{spec.model_dump_json(indent=2)}

Outcome:
- Status: {outcome.status}
- Selected Assets: {len(outcome.selected_assets)}
- Total Proceeds: ${outcome.proceeds:,.0f}
- Leverage: {outcome.leverage_before:.2f}x → {outcome.leverage_after:.2f}x
- Interest Coverage: {outcome.interest_coverage_before:.2f}x → {outcome.interest_coverage_after:.2f}x
- Fixed Charge Coverage: {outcome.fixed_charge_coverage_before:.2f}x → {outcome.fixed_charge_coverage_after:.2f}x
- Critical Fraction: {outcome.critical_fraction:.1%}
- Violations: {outcome.violations}

Generate an Explanation in JSON format.
```

### 9.5 LLM Wrapper Implementation

```python
from openai import AsyncOpenAI
from pydantic import BaseModel

client = AsyncOpenAI()

async def call_llm_structured[T: BaseModel](
    system_prompt: str,
    user_prompt: str,
    response_model: type[T],
) -> T:
    """Generic wrapper for structured LLM calls."""

    response = await client.beta.chat.completions.parse(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=response_model,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )

    return response.choices[0].message.parsed
```

---

## 10. API Design

Implemented in `app/api.py`.

### 10.1 Endpoints

#### POST /program

Main endpoint to run a funding program.

**Request Body:** `ProgramRequest`

```json
{
  "assets": [
    {
      "asset_id": "A001",
      "asset_type": "store",
      "market": "Dallas, TX",
      "market_tier": 2,
      "noi": 500000,
      "book_value": 4000000,
      "criticality": 0.3,
      "leaseability_score": 0.8
    }
  ],
  "corporate_state": {
    "net_debt": 2000000000,
    "ebitda": 500000000,
    "interest_expense": 100000000
  },
  "program_type": "slb",
  "program_description": "Raise ~$500M via SLB, keep leverage below 4x and coverage above 3x. Avoid mission-critical sites."
}
```

**Response Body:** `ProgramResponse`

```json
{
  "selector_spec": { ... },
  "outcome": {
    "status": "ok",
    "selected_assets": [ ... ],
    "proceeds": 485000000,
    "leverage_before": 4.0,
    "leverage_after": 3.03,
    "interest_coverage_before": 5.0,
    "interest_coverage_after": 4.2,
    "fixed_charge_coverage_before": 5.0,
    "fixed_charge_coverage_after": 3.1,
    "critical_fraction": 0.15,
    "violations": []
  },
  "explanation": {
    "summary": "Successfully structured a $485M SLB program...",
    "binding_constraints": ["min_fixed_charge_coverage"],
    "key_drivers": ["Selected 45 store assets with low criticality..."],
    "risks": ["Concentration in Texas markets (35% of proceeds)"],
    "alternatives_considered": []
  }
}
```

**Status Codes:**

| Code | Condition |
|------|-----------|
| 200 | Success (includes infeasible outcomes) |
| 400 | Invalid request (bad program_type, validation failure) |
| 422 | Pydantic validation error |
| 500 | Internal error (LLM failure, numeric error) |

### 10.2 Error Responses

```python
class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str]
    code: str  # e.g., "UNSUPPORTED_PROGRAM_TYPE", "SPEC_VALIDATION_FAILED"
```

### 10.3 FastAPI Application

```python
from fastapi import FastAPI, HTTPException
from app.models import ProgramRequest, ProgramResponse, ErrorResponse
from app.orchestrator import run_program

app = FastAPI(
    title="SLB Agent API",
    version="1.0.0",
    description="Real Estate Funding Workflow Agent (SLB Template)",
)

@app.post(
    "/program",
    response_model=ProgramResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def create_program(request: ProgramRequest) -> ProgramResponse:
    if request.program_type != ProgramType.SLB:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error="Unsupported program type",
                detail=f"program_type '{request.program_type}' not supported in v1",
                code="UNSUPPORTED_PROGRAM_TYPE",
            ).model_dump(),
        )

    try:
        return await run_program(request)
    except SpecValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error="Spec validation failed",
                detail=str(e.errors),
                code="SPEC_VALIDATION_FAILED",
            ).model_dump(),
        )
    except EngineError as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="Engine error",
                detail=str(e),
                code="ENGINE_ERROR",
            ).model_dump(),
        )
```

---

## 11. Minimal UI Sketch

**Note:** No implementation required. This section defines the data contract and wireframe.

### 11.1 Page Layout

```
┌────────────────────────────────────────────────────────────────┐
│  SLB Program Builder                                    [Run] │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Program Description:                                          │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ Raise ~$500M via SLB, keep leverage below 4x and        │ │
│  │ coverage above 3x. Avoid mission-critical sites.        │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  Target Override (optional): [________] ← numeric input        │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─ Summary Card ─────────────────────────────────────────┐   │
│  │                                                         │   │
│  │  Status: [OK ✓]  /  [INFEASIBLE ✗]                     │   │
│  │                                                         │   │
│  │  Target:    $500,000,000                               │   │
│  │  Achieved:  $485,000,000  (97%)                        │   │
│  │                                                         │   │
│  │  ┌──────────────────────┬─────────────┬─────────────┐  │   │
│  │  │                      │   Before    │    After    │  │   │
│  │  ├──────────────────────┼─────────────┼─────────────┤  │   │
│  │  │ Leverage             │    4.00x    │    3.03x    │  │   │
│  │  │ Interest Coverage    │    5.00x    │    4.20x    │  │   │
│  │  │ Fixed Charge Cov.    │    5.00x    │    3.10x    │  │   │
│  │  └──────────────────────┴─────────────┴─────────────┘  │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                │
│  ┌─ Constraints Panel ────────────────────────────────────┐   │
│  │  [✓] Max Leverage: 4.0x (actual: 3.03x)               │   │
│  │  [✓] Min Fixed Charge Coverage: 3.0x (actual: 3.10x)  │   │
│  │  [✓] Max Critical Fraction: 30% (actual: 15%)         │   │
│  │                                                         │   │
│  │  Violations: None                                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                │
│  ┌─ Selected Assets ──────────────────────────────────────┐   │
│  │ ID    │ Type  │ Market     │ Crit │ Lease │ Proceeds  │   │
│  │───────┼───────┼────────────┼──────┼───────┼───────────│   │
│  │ A001  │ Store │ Dallas, TX │ 0.30 │ 0.80  │ $9.5M     │   │
│  │ A002  │ Store │ Austin, TX │ 0.25 │ 0.85  │ $8.2M     │   │
│  │ ...   │       │            │      │       │           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                │
│  ┌─ Explanation ──────────────────────────────────────────┐   │
│  │                                                         │   │
│  │  Summary:                                               │   │
│  │  Successfully structured a $485M SLB program by        │   │
│  │  selecting 45 store assets across 12 markets...        │   │
│  │                                                         │   │
│  │  Binding Constraints: [min_fixed_charge_coverage]      │   │
│  │                                                         │   │
│  │  Key Drivers:                                          │   │
│  │  • Selected low-criticality stores (avg 0.28)          │   │
│  │  • Prioritized high-leaseability assets (avg 0.82)     │   │
│  │                                                         │   │
│  │  Risks:                                                │   │
│  │  • Texas market concentration (35% of proceeds)        │   │
│  │  • 3 assets with coverage ratio sensitivity            │   │
│  │                                                         │   │
│  │  Alternatives Considered: None                         │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 11.2 Data Contract

| UI Component | Data Source |
|--------------|-------------|
| Status Badge | `outcome.status` |
| Target | `selector_spec.target_amount` |
| Achieved | `outcome.proceeds` |
| Leverage Before/After | `outcome.leverage_before`, `outcome.leverage_after` |
| Interest Coverage Before/After | `outcome.interest_coverage_before`, `outcome.interest_coverage_after` |
| Fixed Charge Coverage Before/After | `outcome.fixed_charge_coverage_before`, `outcome.fixed_charge_coverage_after` |
| Constraint Checks | Compare `outcome.*` vs `selector_spec.hard_constraints.*` |
| Violations List | `outcome.violations` |
| Asset Table | `outcome.selected_assets[].asset`, `.proceeds`, `.slb_rent` |
| Explanation Panels | `explanation.summary`, `.binding_constraints`, `.key_drivers`, `.risks`, `.alternatives_considered` |

### 11.3 Demo Portfolio

For the demo, the frontend can hardcode a sample portfolio:

```typescript
const DEMO_ASSETS: Asset[] = [
  // ~100 assets with varied types, markets, criticality, leaseability
  // Total NOI ~$150M, estimated market value ~$2.3B
];

const DEMO_CORPORATE_STATE: CorporateState = {
  net_debt: 2_000_000_000,
  ebitda: 500_000_000,
  interest_expense: 100_000_000,
};
```

---

## 12. Testing Strategy

### 12.1 Unit Tests for Engine (`tests/test_engine.py`)

#### Scenario A: Feasible Selection

**Setup:**
- 10 synthetic assets (mix of types, tiers)
- Total market value: ~$500M
- Target: $100M
- Hard constraints: leverage ≤ 4.0, fixed_charge_coverage ≥ 3.0
- Corporate state: leverage = 4.0, interest_coverage = 5.0

**Assertions:**
- `outcome.status == SelectionStatus.OK`
- `outcome.violations == []`
- `outcome.proceeds >= target * 0.98`
- `outcome.leverage_after <= 4.0`
- `outcome.fixed_charge_coverage_after >= 3.0`

#### Scenario B: Infeasible Target

**Setup:**
- Same 10 assets (~$500M market value)
- Target: $600M (impossible)
- Same constraints

**Assertions:**
- `outcome.status == SelectionStatus.INFEASIBLE`
- `"TARGET_NOT_MET" in [v.code for v in outcome.violations]`
- `outcome.proceeds < target * 0.98`

#### Scenario C: Constraint Violation

**Setup:**
- Assets where SLB would push fixed charge coverage below minimum
- Target achievable but would violate min_fixed_charge_coverage

**Assertions:**
- `outcome.status == SelectionStatus.INFEASIBLE`
- `"MIN_FIXED_CHARGE_COVERAGE" in [v.code for v in outcome.violations]`

#### Scenario D: Filter Eliminates All Assets

**Setup:**
- `asset_filters.max_criticality = 0.1`
- All assets have `criticality > 0.1`

**Assertions:**
- `outcome.status == SelectionStatus.INFEASIBLE`
- `"NO_ELIGIBLE_ASSETS" in [v.code for v in outcome.violations]`

### 12.2 Unit Tests for Validation (`tests/test_engine.py`)

**Test validate_spec:**
- Invalid: `target_amount = 0` → error
- Invalid: `max_net_leverage = 15` → error
- Invalid: `min_fixed_charge_coverage = -1` → error
- Valid: reasonable values → empty error list

**Test validate_assets:**
- Invalid: empty list → error
- Invalid: duplicate asset_ids → error
- Invalid: `noi < 0` → error
- Invalid: `criticality > 1` → error
- Valid: well-formed assets → empty error list

**Test validate_corporate_state:**
- Invalid: `net_debt < 0` → error
- Invalid: `interest_expense < 0` → error
- Warning: `ebitda < 0` → allowed but metrics will be None
- Valid: reasonable values → empty error list

### 12.3 Integration Tests for Orchestrator (`tests/test_orchestrator.py`)

Use mocked LLM calls to test orchestration logic.

#### Test: Revision Loop Terminates

**Setup:**
- Mock `generate_selector_spec` → returns spec with high target
- Mock `revise_selector_spec` → returns progressively lower targets
- `max_iterations = 3`

**Assertions:**
- Loop runs at most 3 times
- Final `target_amount <= initial_target_amount`

#### Test: Immutable Hard Constraints

**Setup:**
- Mock revision that attempts to increase `max_net_leverage`

**Assertions:**
- `enforce_revision_policy` rejects or adjusts
- Final spec has `max_net_leverage <= original`

#### Test: Target Never Increases

**Setup:**
- Mock revision that attempts to increase target

**Assertions:**
- `enforce_revision_policy` rejects
- `new_target <= prev_target` after enforcement

#### Test: Bounded Target Reduction

**Setup:**
- Mock revision that drops target by 50% in one step

**Assertions:**
- `enforce_revision_policy` limits to 20% reduction
- `new_target >= prev_target * 0.80`

### 12.4 Test Utilities

```python
# tests/conftest.py

@pytest.fixture
def sample_assets() -> List[Asset]:
    """Generate synthetic portfolio for testing."""
    ...

@pytest.fixture
def sample_corporate_state() -> CorporateState:
    """Standard corporate state for testing."""
    return CorporateState(
        net_debt=2_000_000_000,
        ebitda=500_000_000,
        interest_expense=100_000_000,
    )

@pytest.fixture
def mock_llm(monkeypatch):
    """Mock LLM calls for deterministic testing."""
    ...
```

### 12.5 Invariant Tests (`tests/test_invariants.py`)

Beyond example-based tests, add property-based tests that verify system invariants hold across random inputs.

#### 12.5.1 Metric Invariants

```python
from hypothesis import given, strategies as st

@given(
    net_debt=st.floats(min_value=0, max_value=1e12),
    ebitda=st.floats(min_value=-1e9, max_value=1e12),
    interest_expense=st.floats(min_value=0, max_value=1e11),
)
def test_baseline_metrics_no_nan_or_inf(net_debt, ebitda, interest_expense):
    """Baseline metrics should never produce unexpected NaN/Inf."""
    state = CorporateState(
        net_debt=net_debt,
        ebitda=ebitda,
        interest_expense=interest_expense,
    )
    metrics = compute_baseline_metrics(state, DEFAULT_ENGINE_CONFIG)

    # Metrics can be None (not meaningful) but should not be NaN/Inf
    if metrics.leverage is not None:
        assert not math.isnan(metrics.leverage)
        assert not math.isinf(metrics.leverage)
    if metrics.interest_coverage is not None:
        assert not math.isnan(metrics.interest_coverage)
        assert not math.isinf(metrics.interest_coverage)
```

#### 12.5.2 Transaction Invariants

```python
def test_net_debt_never_increases_from_slb(sample_assets, sample_corporate_state):
    """SLB always reduces or maintains net debt (never increases)."""
    spec = make_valid_spec(target_amount=100_000_000)
    outcome = select_assets(sample_assets, sample_corporate_state, spec, DEFAULT_ENGINE_CONFIG)

    if outcome.leverage_before is not None and outcome.leverage_after is not None:
        # Net debt after should be <= net debt before
        # (leverage = net_debt / ebitda, ebitda constant)
        assert outcome.leverage_after <= outcome.leverage_before


def test_idempotence_zero_proceeds(sample_assets, sample_corporate_state):
    """If no assets selected, after metrics should equal before metrics."""
    spec = make_valid_spec(target_amount=100_000_000)
    spec.asset_filters.max_criticality = 0.0  # Filter out everything

    outcome = select_assets(sample_assets, sample_corporate_state, spec, DEFAULT_ENGINE_CONFIG)

    assert outcome.selected_assets == []
    assert outcome.leverage_after == outcome.leverage_before
    assert outcome.interest_coverage_after == outcome.interest_coverage_before
    assert outcome.fixed_charge_coverage_after == outcome.fixed_charge_coverage_before
```

#### 12.5.3 Guardrail Invariants

```python
def test_interest_after_never_negative():
    """Interest expense should never go negative, even with large proceeds."""
    state = CorporateState(
        net_debt=100_000_000,
        ebitda=50_000_000,
        interest_expense=5_000_000,  # Small interest
    )
    # Proceeds >> interest / avg_cost_of_debt
    large_asset = Asset(
        asset_id="large",
        asset_type=AssetType.STORE,
        market="NYC",
        noi=50_000_000,  # ~$700M proceeds at 7% cap
        book_value=500_000_000,
        criticality=0.1,
        leaseability_score=0.9,
    )

    spec = make_valid_spec(target_amount=500_000_000)
    outcome = select_assets([large_asset], state, spec, DEFAULT_ENGINE_CONFIG)

    # Interest coverage after should be None (interest clamped to 0) or positive
    assert outcome.interest_coverage_after is None or outcome.interest_coverage_after > 0
    assert "clamped to zero" in " ".join(outcome.warnings).lower() or outcome.interest_coverage_after is None
```

### 12.6 Golden Baseline Portfolio (`tests/golden/`)

A canonical test portfolio with hand-verified results serves as a regression anchor.

#### 12.6.1 Portfolio Definition

```python
# tests/golden/canonical_portfolio.py

GOLDEN_ASSETS = [
    Asset(asset_id="G001", asset_type=AssetType.STORE, market="Dallas, TX",
          market_tier=MarketTier.TIER_2, noi=500_000, book_value=4_000_000,
          criticality=0.3, leaseability_score=0.8),
    Asset(asset_id="G002", asset_type=AssetType.STORE, market="Austin, TX",
          market_tier=MarketTier.TIER_2, noi=450_000, book_value=3_500_000,
          criticality=0.25, leaseability_score=0.85),
    Asset(asset_id="G003", asset_type=AssetType.DISTRIBUTION_CENTER, market="Chicago, IL",
          market_tier=MarketTier.TIER_1, noi=2_000_000, book_value=35_000_000,
          criticality=0.7, leaseability_score=0.6),
    Asset(asset_id="G004", asset_type=AssetType.STORE, market="Phoenix, AZ",
          market_tier=MarketTier.TIER_3, noi=300_000, book_value=2_500_000,
          criticality=0.2, leaseability_score=0.9),
    Asset(asset_id="G005", asset_type=AssetType.OFFICE, market="NYC",
          market_tier=MarketTier.TIER_1, noi=1_500_000, book_value=20_000_000,
          criticality=0.8, leaseability_score=0.5),
    # ... 5 more assets for a total of 10
]

GOLDEN_CORPORATE_STATE = CorporateState(
    net_debt=200_000_000,
    ebitda=50_000_000,
    interest_expense=10_000_000,
)

# Hand-verified expected results for specific scenarios
GOLDEN_SCENARIOS = {
    "baseline": {
        "target": 20_000_000,
        "expected_status": SelectionStatus.OK,
        "expected_proceeds_min": 19_600_000,  # 98% of target
        "expected_leverage_after_max": 4.0,
        "expected_selected_count_min": 2,
    },
    "high_target": {
        "target": 100_000_000,
        "expected_status": SelectionStatus.INFEASIBLE,
        "expected_violation_codes": ["TARGET_NOT_MET"],
    },
    "strict_coverage": {
        "target": 20_000_000,
        "min_fixed_charge_coverage": 5.0,  # Stricter than achievable
        "expected_status": SelectionStatus.INFEASIBLE,
        "expected_violation_codes": ["MIN_FIXED_CHARGE_COVERAGE"],
    },
}
```

#### 12.6.2 Golden Tests

```python
# tests/test_golden.py

def test_golden_baseline():
    """Regression test: baseline scenario produces expected results."""
    scenario = GOLDEN_SCENARIOS["baseline"]
    spec = make_spec_from_scenario(scenario)

    outcome = select_assets(GOLDEN_ASSETS, GOLDEN_CORPORATE_STATE, spec, DEFAULT_ENGINE_CONFIG)

    assert outcome.status == scenario["expected_status"]
    assert outcome.proceeds >= scenario["expected_proceeds_min"]
    assert outcome.leverage_after <= scenario["expected_leverage_after_max"]
    assert len(outcome.selected_assets) >= scenario["expected_selected_count_min"]


def test_golden_all_scenarios():
    """Run all golden scenarios and verify expected outcomes."""
    for name, scenario in GOLDEN_SCENARIOS.items():
        spec = make_spec_from_scenario(scenario)
        outcome = select_assets(GOLDEN_ASSETS, GOLDEN_CORPORATE_STATE, spec, DEFAULT_ENGINE_CONFIG)

        assert outcome.status == scenario["expected_status"], f"Failed: {name}"
        if "expected_violation_codes" in scenario:
            actual_codes = [v.code for v in outcome.violations]
            for expected_code in scenario["expected_violation_codes"]:
                assert expected_code in actual_codes, f"Failed: {name}, missing {expected_code}"
```

#### 12.6.3 Golden Baseline Maintenance

When changing the economics engine:
1. Run golden tests first — if they fail, the change affects outputs
2. Manually verify whether the new outputs are correct
3. If correct, update the expected values in `GOLDEN_SCENARIOS`
4. Document why the values changed in the commit message

**This forces conscious decision-making:** you cannot accidentally change economics without noticing.

---

## 13. Future Extensions

### 13.1 Additional Program Types

The architecture supports future program types with minimal changes:

| Program Type | Reusable Components | New Components |
|--------------|---------------------|----------------|
| **Revolver** | Asset, CorporateState, scoring, filters | Revolver-specific economics, borrowing base calc |
| **CMBS** | Asset, engine structure, LLM patterns | CMBS tranching, rating agency constraints |
| **Closures** | Asset, criticality scoring | Closure economics, severance modeling |

### 13.2 Enhanced Optimization

| Enhancement | Complexity | Benefit |
|-------------|------------|---------|
| OR-Tools MILP | Medium | Global optimum guarantee |
| Multi-objective Pareto | High | Trade-off visualization |
| Scenario analysis | Medium | Sensitivity to cap rates, interest rates |

### 13.3 Data Ingestion

| Data Source | Integration Pattern |
|-------------|---------------------|
| Excel portfolio | File upload → pandas → Asset[] |
| Argus exports | Parser → Asset economics |
| Data room documents | RAG pipeline → asset enrichment |

### 13.4 Production Readiness

| Area | V1 (Demo) | Production |
|------|-----------|------------|
| Auth | None | OAuth2 / API keys |
| Persistence | In-memory | PostgreSQL |
| Observability | Logs | OpenTelemetry, Prometheus |
| Deployment | Local | k8s, CI/CD |
| Rate limiting | None | Redis-based |

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **SLB** | Sale-Leaseback: sell property, lease it back, continue operations |
| **NOI** | Net Operating Income: property revenue minus operating expenses |
| **Cap Rate** | Capitalization rate: NOI / Market Value |
| **WAULT** | Weighted Average Unexpired Lease Term |
| **Leverage** | Net Debt / EBITDA ratio |
| **Coverage** | EBITDA / (Interest + Lease Expense) ratio |
| **Critical Fraction** | NOI from high-criticality assets / Total selected NOI |

---

## Appendix B: Sample Program Descriptions

**Conservative:**
> "Raise $300M via sale-leaseback to reduce leverage below 3.5x. Focus on non-critical retail stores with high re-leasing potential. Avoid any distribution centers."

**Aggressive:**
> "Maximize SLB proceeds to fully deleverage. Include all asset types. Maintain minimum 2.5x coverage."

**Constrained:**
> "Raise $500M via SLB, but keep Tier-1 DC exposure below 10% of proceeds. Maintain current leverage ratio of 4.0x and coverage above 3.0x."

---

*End of Design Document*
