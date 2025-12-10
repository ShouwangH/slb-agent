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

from app.config import DEFAULT_ENGINE_CONFIG, EngineConfig
from app.models import ConstraintViolation, ExplanationNode, ProgramOutcome, SelectorSpec

# =============================================================================
# generate_selector_spec Prompts (Section 9.2)
# =============================================================================

GENERATE_SPEC_SYSTEM = """You are a commercial real estate funding analyst. Your task is to interpret a
program brief and produce a structured SelectorSpec for a {program_type} program.

IMPORTANT RULES:
1. Never perform calculations. You do not know exact portfolio metrics.
2. Use the asset_summary for context, but do not reference specific values.
3. Apply default values when the brief is vague.
4. Output valid JSON matching the SelectorSpec schema.

DEFAULTS (use when not specified):
- max_net_leverage: {default_max_net_leverage}
- min_fixed_charge_coverage: {default_min_fixed_charge_coverage}
- target_amount: 20% of estimated portfolio market value (from summary)
- max_iterations: {default_max_iterations}"""

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
a feasible solution.

REVISION RULES:
1. You may REDUCE target_amount (never increase, max 20% reduction)
2. You may RELAX asset filters (increase max_criticality, decrease min_leaseability)
3. You may ADJUST soft preference weights
4. You may NOT change hard_constraints (leverage, coverage limits are immutable)
5. You may NOT change program_type

CONSTRAINT VIOLATIONS FROM PREVIOUS RUN:
{violations}

STRATEGY:
- If TARGET_NOT_MET: consider reducing target or relaxing filters
- If MAX_NET_LEVERAGE: cannot fix via spec (hard constraint)
- If MAX_CRITICAL_FRACTION: reduce max_criticality filter or reduce target"""

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
    config: EngineConfig = DEFAULT_ENGINE_CONFIG,
) -> str:
    """Format the system prompt for generate_selector_spec."""
    return GENERATE_SPEC_SYSTEM.format(
        program_type=program_type,
        default_max_net_leverage=config.default_max_net_leverage,
        default_min_fixed_charge_coverage=config.default_min_fixed_charge_coverage,
        default_max_iterations=config.default_max_iterations,
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
