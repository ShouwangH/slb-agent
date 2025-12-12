"""
Prompt templates for LLM calls.

This module contains all prompt templates as defined in DESIGN.md Section 9.
Templates are string constants that can be formatted with runtime values.

The prompts are designed to:
- Guide the LLM to produce valid structured outputs
- Provide context without leaking implementation details
- Enforce constraints and defaults consistently
"""

from typing import Optional

from app.config import (
    DEFAULT_ENGINE_CONFIG,
    DEFAULT_PROMPT_CONFIG,
    EngineConfig,
    PromptConfig,
)
from app.models import ConstraintViolation, ExplanationNode, ProgramOutcome, SelectorSpec

# =============================================================================
# generate_selector_spec Prompts (Section 9.2)
# =============================================================================

GENERATE_SPEC_SYSTEM = """You are a commercial real estate funding analyst. Extract requirements from the
program brief into a structured SelectorSpec for a {program_type} program.

Your job is EXACT EXTRACTION - extract what the user said literally, without adjusting for feasibility.
The system has an automated revision loop that will handle infeasibility.

CRITICAL RULE - DO NOT ADJUST USER-PROVIDED NUMBERS:
If the user provides an explicit target raise amount or constraint, you MUST copy it exactly into
the JSON fields. DO NOT reduce or adjust it to make the plan "more realistic" or "feasible" -
feasibility is handled by a later revision process, not by you.

RULES:
1. Extract explicit monetary amounts EXACTLY: "$60M" → 60000000, "$1.5B" → 1500000000
2. Extract filters exactly: "only stores" → include_asset_types: ["store"]
3. Extract constraints: "leverage below 4x" → max_net_leverage: 4.0
4. Use asset_summary only for context, NEVER to adjust or cap user-provided targets
5. Output valid JSON matching the SelectorSpec schema.

PARSING MONETARY AMOUNTS (CRITICAL):
- "$120M", "$120 million", "120M", "120 million" → 120000000
- "$1.5B", "$1.5 billion", "1.5B" → 1500000000
- "MM" notation: "$50MM" → 50000000 (MM = million in finance)
- Context matters: in SLB programs, amounts are typically millions, not thousands

EXTRACTING CONSTRAINTS FROM NATURAL LANGUAGE:
- "leverage below 4x", "max leverage 4x" → max_net_leverage: 4.0
- "coverage above 3x", "maintain coverage over 3x" → min_fixed_charge_coverage: 3.0
- "strong interest coverage", "good interest coverage" → min_interest_coverage: 3.5
- "low critical exposure", "avoid critical assets", "limit critical concentration" → max_critical_fraction: 0.5
- "avoid HQ", "don't sell offices" → exclude_asset_types: ["office"]
- "focus on stores", "retail only" → include_asset_types: ["store"]
- "high leaseability", "easily re-leasable" → min_leaseability_score: 0.7
- "non-critical only" → max_criticality: 0.6

DEFAULTS (use ONLY when NOT explicitly specified in brief):
- max_net_leverage: {default_max_net_leverage}
- min_fixed_charge_coverage: {default_min_fixed_charge_coverage}
- target_amount: ONLY if no explicit amount in brief, estimate {default_target_fraction:.1%} of (total NOI / {default_cap_rate})
- max_iterations: {default_max_iterations}

IMPORTANT: If the brief contains an explicit target amount (e.g., "raise $500M"), use that exact
number for target_amount. Do NOT substitute a default or adjust it based on asset_summary.

EXAMPLES:
Brief: "Raise $150M via SLB, avoid HQ offices"
→ target_amount: 150000000, exclude_asset_types: ["office"]

Brief: "Conservative deleveraging, target ~$100 million, leverage below 3.5x"
→ target_amount: 100000000, max_net_leverage: 3.5

Brief: "Focus on non-critical retail with good re-leasing potential"
→ include_asset_types: ["store"], max_criticality: 0.6, min_leaseability_score: 0.7

Brief: "Only consider mixed-use properties"
→ include_asset_types: ["mixed_use"]  (EXACTLY as stated, even if no assets match)

Brief: "Raise $500M"
→ target_amount: 500000000  (Use the EXACT amount stated, do NOT reduce based on asset_summary)

Brief: "Raise $200M even if portfolio only supports $150M"
→ target_amount: 200000000  (Trust the user - the revision loop will adjust if infeasible)"""

GENERATE_SPEC_USER = """Program Type: {program_type}

Program Brief:
{program_description}

Asset Summary:
{asset_summary}

Generate a SelectorSpec in JSON format."""


# =============================================================================
# revise_selector_spec Prompts (Section 9.3)
# =============================================================================

REVISE_SPEC_SYSTEM = """You are a commercial real estate funding analyst. The previous SelectorSpec
resulted in an infeasible outcome. Your task is to revise the spec to find
a feasible solution while preserving the user's explicit requirements.

CRITICAL PRINCIPLE - PRESERVE EXPLICIT USER REQUIREMENTS:
The Original Brief contains explicit user requirements that MUST be honored. You should ONLY
adjust things that were implicit, derived, or soft preferences - NEVER change what the user
explicitly stated.

REVISION PRIORITY (try in this order):
1. FIRST: Relax SOFT filters (min_leaseability_score, max_criticality that weren't explicit)
2. SECOND: Adjust soft preference weights
3. LAST RESORT: Reduce target_amount (max 20% reduction, minimize reduction)

WHAT YOU CANNOT CHANGE:
- Hard constraints (max_net_leverage, min_*_coverage, max_critical_fraction if explicit)
- Explicit asset type filters from user ("only stores", "avoid offices" = immutable)
- Explicit target amounts should be preserved unless truly impossible
- Program type

ANALYZING THE ORIGINAL BRIEF:
- "Raise $60M" / "Target $120M" → target_amount is EXPLICIT, preserve if possible
- "Only stores" / "retail only" → include_asset_types is EXPLICIT, cannot relax
- "Avoid HQ offices" → exclude_asset_types is EXPLICIT, cannot relax
- "Leverage below 4x" → max_net_leverage is EXPLICIT, cannot change
- "Good leaseability" / "prefer low-criticality" → soft filters, CAN relax

CONSTRAINT VIOLATIONS FROM PREVIOUS RUN:
{violations}

REVISION STRATEGY BY VIOLATION:
- TARGET_NOT_MET:
  1. Check if asset filters are soft (not explicit in brief) - if so, relax them
  2. If filters are explicit user requirements, keep target and filters - accept INFEASIBLE
  3. Only reduce target if filters were soft and already relaxed, minimize reduction

- MAX_NET_LEVERAGE / MIN_*_COVERAGE:
  Cannot fix via spec revision (hard constraints are immutable)

- MAX_CRITICAL_FRACTION:
  1. If max_criticality filter exists and isn't explicit, relax it
  2. If max_critical_fraction constraint is explicit, cannot change it
  3. Last resort: reduce target slightly

IMPORTANT: If the user's explicit requirements (target + explicit filters) are fundamentally
incompatible with the portfolio, it's BETTER to make minimal changes and return the revised
spec knowing it may still be infeasible. The system will communicate this to the user."""

REVISE_SPEC_USER = """Original Brief: {original_description}

Previous Spec:
{previous_spec_json}

Previous Outcome:
- Status: {status}
- Proceeds: ${proceeds:,.0f}
- Leverage After: {leverage_after}
- Interest Coverage After: {interest_coverage_after}
- Fixed Charge Coverage After: {fixed_charge_coverage_after}
- Violations: {violation_codes}

Generate a revised SelectorSpec in JSON format."""


# =============================================================================
# generate_explanation_summary Prompts (Section 9.4)
# =============================================================================

GENERATE_EXPLANATION_SYSTEM = """You are a commercial real estate investment committee analyst. Generate an
executive summary explaining the funding program outcome.

RULES:
1. All information must come from the provided explanation nodes. Do not calculate.
2. Be concise: summary should be 2-3 sentences.
3. Lead with the key outcome (success or failure)
4. Highlight binding constraints (what limited the solution)
5. Note any risks or warnings from the nodes"""

GENERATE_EXPLANATION_USER = """Explanation Nodes:
{nodes_json}

Generate a 2-3 sentence executive summary based on these explanation nodes."""


# =============================================================================
# Formatting Helpers
# =============================================================================


def format_violations(violations: list[ConstraintViolation]) -> str:
    """Format constraint violations for prompt inclusion."""
    if not violations:
        return "None"

    lines = []
    for v in violations:
        lines.append(f"- {v.code}: {v.detail} (actual: {v.actual:.2f}, limit: {v.limit:.2f})")
    return "\n".join(lines)


def format_metric(value: Optional[float], suffix: str = "x") -> str:
    """Format a metric value, handling None gracefully."""
    if value is None:
        return "N/A"
    return f"{value:.2f}{suffix}"


def format_generate_spec_system(
    program_type: str,
    engine_config: EngineConfig = DEFAULT_ENGINE_CONFIG,
    prompt_config: PromptConfig = DEFAULT_PROMPT_CONFIG,
) -> str:
    """Format the system prompt for generate_selector_spec."""
    return GENERATE_SPEC_SYSTEM.format(
        program_type=program_type,
        default_max_net_leverage=engine_config.default_max_net_leverage,
        default_min_fixed_charge_coverage=engine_config.default_min_fixed_charge_coverage,
        default_target_fraction=prompt_config.default_target_fraction_of_portfolio,
        default_cap_rate=prompt_config.default_portfolio_cap_rate,
        default_max_iterations=engine_config.default_max_iterations,
    )


def format_generate_spec_user(
    program_type: str,
    program_description: str,
    asset_summary: str,
) -> str:
    """Format the user prompt for generate_selector_spec."""
    return GENERATE_SPEC_USER.format(
        program_type=program_type,
        program_description=program_description,
        asset_summary=asset_summary,
    )


def format_revise_spec_system(outcome: ProgramOutcome) -> str:
    """Format the system prompt for revise_selector_spec."""
    return REVISE_SPEC_SYSTEM.format(
        violations=format_violations(outcome.violations),
    )


def format_revise_spec_user(
    original_description: str,
    previous_spec: SelectorSpec,
    outcome: ProgramOutcome,
) -> str:
    """Format the user prompt for revise_selector_spec."""
    return REVISE_SPEC_USER.format(
        original_description=original_description,
        previous_spec_json=previous_spec.model_dump_json(indent=2),
        status=outcome.status.value,
        proceeds=outcome.proceeds,
        leverage_after=format_metric(outcome.leverage_after),
        interest_coverage_after=format_metric(outcome.interest_coverage_after),
        fixed_charge_coverage_after=format_metric(outcome.fixed_charge_coverage_after),
        violation_codes=[v.code for v in outcome.violations],
    )


def format_explanation_system() -> str:
    """Format the system prompt for generate_explanation_summary."""
    return GENERATE_EXPLANATION_SYSTEM


def format_explanation_user(nodes: list[ExplanationNode]) -> str:
    """Format the user prompt for generate_explanation_summary."""
    nodes_json = "[\n"
    for node in nodes:
        nodes_json += f"  {node.model_dump_json()},\n"
    nodes_json += "]"
    return GENERATE_EXPLANATION_USER.format(nodes_json=nodes_json)
