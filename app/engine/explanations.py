"""
Explanation generation for SLB Agent.

This module generates structured ExplanationNode objects from selection results
as defined in DESIGN.md Section 4.6.

All functions are pure and deterministic. The LLM layer uses these nodes
to generate human-readable summaries.

Node Categories:
- constraint: Binding constraints that limited the solution
- driver: Key factors in asset selection
- risk: Risk factors in the selected pool
- alternative: Alternatives considered or rejected
"""

from typing import Optional

from app.config import EngineConfig
from app.models import (
    CorporateState,
    ExplanationNode,
    ProgramOutcome,
    SelectionStatus,
    SelectorSpec,
)


# =============================================================================
# Main Entry Point
# =============================================================================


def generate_explanation_nodes(
    spec: SelectorSpec,
    outcome: ProgramOutcome,
    state: CorporateState,
    config: EngineConfig,
) -> list[ExplanationNode]:
    """
    Generate structured explanation nodes from selection results.

    Produces nodes for:
    - Binding/violated constraints
    - Selection drivers (preferences that influenced selection)
    - Risk factors (concentration, critical assets)

    Args:
        spec: The selector specification used
        outcome: The program outcome from selection
        state: Pre-transaction corporate state
        config: Engine configuration

    Returns:
        List of ExplanationNode objects for LLM/UI rendering
    """
    nodes: list[ExplanationNode] = []

    # Generate constraint nodes
    nodes.extend(_generate_constraint_nodes(spec, outcome, config))

    # Generate driver nodes
    nodes.extend(_generate_driver_nodes(spec, outcome))

    # Generate risk nodes
    nodes.extend(_generate_risk_nodes(spec, outcome, config))

    return nodes


# =============================================================================
# Constraint Nodes (Binding/Violated Constraints)
# =============================================================================


def _generate_constraint_nodes(
    spec: SelectorSpec,
    outcome: ProgramOutcome,
    config: EngineConfig,
) -> list[ExplanationNode]:
    """Generate nodes for violated and near-binding constraints."""
    nodes: list[ExplanationNode] = []

    # Add nodes for each violation
    for violation in outcome.violations:
        node = _violation_to_node(violation)
        nodes.append(node)

    # Add nodes for near-binding constraints (within 10% of threshold)
    nodes.extend(_check_near_binding_constraints(spec, outcome, config))

    return nodes


def _violation_to_node(violation) -> ExplanationNode:
    """Convert a ConstraintViolation to an ExplanationNode."""
    # Map violation codes to metric names
    metric_map = {
        "MAX_NET_LEVERAGE": "leverage",
        "MIN_INTEREST_COVERAGE": "interest_coverage",
        "MIN_FIXED_CHARGE_COVERAGE": "fixed_charge_coverage",
        "MAX_CRITICAL_FRACTION": "critical_fraction",
        "TARGET_NOT_MET": "proceeds",
        "NO_ELIGIBLE_ASSETS": None,
    }

    return ExplanationNode(
        id=f"constraint_{violation.code.lower()}",
        label=_format_violation_label(violation.code),
        severity="error",
        category="constraint",
        metric=metric_map.get(violation.code),
        post_value=violation.actual if not _is_nan(violation.actual) else None,
        threshold=violation.limit,
        detail=violation.detail,
    )


def _format_violation_label(code: str) -> str:
    """Format violation code as human-readable label."""
    labels = {
        "MAX_NET_LEVERAGE": "Leverage Constraint Violated",
        "MIN_INTEREST_COVERAGE": "Interest Coverage Below Minimum",
        "MIN_FIXED_CHARGE_COVERAGE": "Fixed Charge Coverage Below Minimum",
        "MAX_CRITICAL_FRACTION": "Critical Asset Concentration Exceeded",
        "TARGET_NOT_MET": "Target Proceeds Not Achieved",
        "NO_ELIGIBLE_ASSETS": "No Eligible Assets Available",
    }
    return labels.get(code, f"Constraint Violated: {code}")


def _check_near_binding_constraints(
    spec: SelectorSpec,
    outcome: ProgramOutcome,
    config: EngineConfig,
) -> list[ExplanationNode]:
    """Check for constraints that are close to binding (within 10%)."""
    nodes: list[ExplanationNode] = []
    hc = spec.hard_constraints

    # Skip if already violated (those are handled separately)
    violated_codes = {v.code for v in outcome.violations}

    # Check leverage (near binding if within 10% of limit)
    if (
        hc.max_net_leverage is not None
        and outcome.leverage_after is not None
        and "MAX_NET_LEVERAGE" not in violated_codes
    ):
        ratio = outcome.leverage_after / hc.max_net_leverage
        if ratio >= 0.9:  # Within 10% of limit
            nodes.append(ExplanationNode(
                id="constraint_leverage_near_binding",
                label="Leverage Near Limit",
                severity="warning",
                category="constraint",
                metric="leverage",
                baseline_value=outcome.leverage_before,
                post_value=outcome.leverage_after,
                threshold=hc.max_net_leverage,
                detail=f"Post-SLB leverage {outcome.leverage_after:.2f}x is {ratio:.0%} of the {hc.max_net_leverage:.2f}x limit",
            ))

    # Check fixed charge coverage (near binding if within 10% above minimum)
    if (
        hc.min_fixed_charge_coverage is not None
        and outcome.fixed_charge_coverage_after is not None
        and "MIN_FIXED_CHARGE_COVERAGE" not in violated_codes
    ):
        ratio = outcome.fixed_charge_coverage_after / hc.min_fixed_charge_coverage
        if ratio <= 1.1:  # Within 10% above minimum
            nodes.append(ExplanationNode(
                id="constraint_fcc_near_binding",
                label="Fixed Charge Coverage Near Minimum",
                severity="warning",
                category="constraint",
                metric="fixed_charge_coverage",
                baseline_value=outcome.fixed_charge_coverage_before,
                post_value=outcome.fixed_charge_coverage_after,
                threshold=hc.min_fixed_charge_coverage,
                detail=f"Post-SLB coverage {outcome.fixed_charge_coverage_after:.2f}x is close to minimum {hc.min_fixed_charge_coverage:.2f}x",
            ))

    # Check interest coverage
    if (
        hc.min_interest_coverage is not None
        and outcome.interest_coverage_after is not None
        and "MIN_INTEREST_COVERAGE" not in violated_codes
    ):
        ratio = outcome.interest_coverage_after / hc.min_interest_coverage
        if ratio <= 1.1:
            nodes.append(ExplanationNode(
                id="constraint_ic_near_binding",
                label="Interest Coverage Near Minimum",
                severity="warning",
                category="constraint",
                metric="interest_coverage",
                baseline_value=outcome.interest_coverage_before,
                post_value=outcome.interest_coverage_after,
                threshold=hc.min_interest_coverage,
                detail=f"Post-SLB interest coverage {outcome.interest_coverage_after:.2f}x is close to minimum {hc.min_interest_coverage:.2f}x",
            ))

    # Check critical fraction
    if (
        hc.max_critical_fraction is not None
        and "MAX_CRITICAL_FRACTION" not in violated_codes
    ):
        ratio = outcome.critical_fraction / hc.max_critical_fraction
        if ratio >= 0.9:
            nodes.append(ExplanationNode(
                id="constraint_critical_near_binding",
                label="Critical Concentration Near Limit",
                severity="warning",
                category="constraint",
                metric="critical_fraction",
                post_value=outcome.critical_fraction,
                threshold=hc.max_critical_fraction,
                detail=f"Critical asset concentration {outcome.critical_fraction:.1%} is {ratio:.0%} of the {hc.max_critical_fraction:.1%} limit",
            ))

    return nodes


# =============================================================================
# Driver Nodes (Selection Factors)
# =============================================================================


def _generate_driver_nodes(
    spec: SelectorSpec,
    outcome: ProgramOutcome,
) -> list[ExplanationNode]:
    """Generate nodes explaining key selection drivers."""
    nodes: list[ExplanationNode] = []

    # Only generate driver nodes for successful selections
    if outcome.status != SelectionStatus.OK or not outcome.selected_assets:
        return nodes

    prefs = spec.soft_preferences

    # Preference-based drivers
    if prefs.prefer_low_criticality:
        avg_criticality = sum(a.asset.criticality for a in outcome.selected_assets) / len(outcome.selected_assets)
        nodes.append(ExplanationNode(
            id="driver_low_criticality",
            label="Low-Criticality Assets Preferred",
            severity="info",
            category="driver",
            metric="criticality",
            post_value=avg_criticality,
            detail=f"Selected assets have average criticality of {avg_criticality:.2f} (weight: {prefs.weight_criticality})",
        ))

    if prefs.prefer_high_leaseability:
        avg_leaseability = sum(a.asset.leaseability_score for a in outcome.selected_assets) / len(outcome.selected_assets)
        nodes.append(ExplanationNode(
            id="driver_high_leaseability",
            label="High-Leaseability Assets Preferred",
            severity="info",
            category="driver",
            metric="leaseability_score",
            post_value=avg_leaseability,
            detail=f"Selected assets have average leaseability of {avg_leaseability:.2f} (weight: {prefs.weight_leaseability})",
        ))

    # Target achievement driver
    if outcome.proceeds >= spec.target_amount:
        nodes.append(ExplanationNode(
            id="driver_target_achieved",
            label="Target Proceeds Achieved",
            severity="info",
            category="driver",
            metric="proceeds",
            post_value=outcome.proceeds,
            threshold=spec.target_amount,
            detail=f"Achieved ${outcome.proceeds:,.0f} vs target ${spec.target_amount:,.0f} ({outcome.proceeds/spec.target_amount:.1%})",
        ))

    # Leverage improvement driver
    if (
        outcome.leverage_before is not None
        and outcome.leverage_after is not None
        and outcome.leverage_after < outcome.leverage_before
    ):
        improvement = outcome.leverage_before - outcome.leverage_after
        nodes.append(ExplanationNode(
            id="driver_leverage_reduction",
            label="Leverage Reduced",
            severity="info",
            category="driver",
            metric="leverage",
            baseline_value=outcome.leverage_before,
            post_value=outcome.leverage_after,
            detail=f"Net leverage reduced by {improvement:.2f}x (from {outcome.leverage_before:.2f}x to {outcome.leverage_after:.2f}x)",
        ))

    return nodes


# =============================================================================
# Risk Nodes (Concentration & Other Risks)
# =============================================================================


def _generate_risk_nodes(
    spec: SelectorSpec,
    outcome: ProgramOutcome,
    config: EngineConfig,
) -> list[ExplanationNode]:
    """Generate nodes identifying risk factors in the selection."""
    nodes: list[ExplanationNode] = []

    if not outcome.selected_assets:
        return nodes

    # Critical asset concentration risk
    if outcome.critical_fraction > 0:
        severity = _get_concentration_severity(outcome.critical_fraction)
        nodes.append(ExplanationNode(
            id="risk_critical_concentration",
            label="Critical Asset Exposure",
            severity=severity,
            category="risk",
            metric="critical_fraction",
            post_value=outcome.critical_fraction,
            threshold=spec.hard_constraints.max_critical_fraction,
            detail=f"{outcome.critical_fraction:.1%} of selected NOI comes from critical assets (criticality > {config.criticality_threshold})",
            asset_ids=_get_critical_asset_ids(outcome, config),
        ))

    # Market concentration risk
    market_concentration = _calculate_market_concentration(outcome)
    if market_concentration["max_share"] > 0.5:  # More than 50% in one market
        nodes.append(ExplanationNode(
            id="risk_market_concentration",
            label="Market Concentration",
            severity="warning" if market_concentration["max_share"] > 0.7 else "info",
            category="risk",
            post_value=market_concentration["max_share"],
            detail=f"{market_concentration['max_share']:.1%} of proceeds from {market_concentration['top_market']}",
        ))

    # Asset type concentration risk
    type_concentration = _calculate_type_concentration(outcome)
    if type_concentration["max_share"] > 0.7:  # More than 70% in one type
        nodes.append(ExplanationNode(
            id="risk_type_concentration",
            label="Asset Type Concentration",
            severity="warning" if type_concentration["max_share"] > 0.9 else "info",
            category="risk",
            post_value=type_concentration["max_share"],
            detail=f"{type_concentration['max_share']:.1%} of proceeds from {type_concentration['top_type'].value} assets",
        ))

    # Single-asset risk
    if len(outcome.selected_assets) == 1:
        nodes.append(ExplanationNode(
            id="risk_single_asset",
            label="Single Asset Selection",
            severity="warning",
            category="risk",
            detail="All proceeds come from a single asset, creating concentration risk",
            asset_ids=[outcome.selected_assets[0].asset.asset_id],
        ))

    # Fixed charge coverage degradation risk
    if (
        outcome.fixed_charge_coverage_before is not None
        and outcome.fixed_charge_coverage_after is not None
        and outcome.fixed_charge_coverage_after < outcome.fixed_charge_coverage_before
    ):
        degradation = (outcome.fixed_charge_coverage_before - outcome.fixed_charge_coverage_after) / outcome.fixed_charge_coverage_before
        if degradation > 0.1:  # More than 10% degradation
            nodes.append(ExplanationNode(
                id="risk_fcc_degradation",
                label="Fixed Charge Coverage Reduced",
                severity="warning" if degradation > 0.2 else "info",
                category="risk",
                metric="fixed_charge_coverage",
                baseline_value=outcome.fixed_charge_coverage_before,
                post_value=outcome.fixed_charge_coverage_after,
                detail=f"Fixed charge coverage decreased by {degradation:.1%} due to added lease expense",
            ))

    return nodes


def _get_concentration_severity(fraction: float) -> str:
    """Determine severity based on concentration level."""
    if fraction > 0.5:
        return "warning"
    elif fraction > 0.3:
        return "info"
    return "info"


def _get_critical_asset_ids(outcome: ProgramOutcome, config: EngineConfig) -> list[str]:
    """Get IDs of critical assets in the selection."""
    return [
        a.asset.asset_id
        for a in outcome.selected_assets
        if a.asset.criticality > config.criticality_threshold
    ]


def _calculate_market_concentration(outcome: ProgramOutcome) -> dict:
    """Calculate market concentration metrics."""
    market_proceeds: dict[str, float] = {}
    total = 0.0

    for selection in outcome.selected_assets:
        market = selection.asset.market
        market_proceeds[market] = market_proceeds.get(market, 0) + selection.proceeds
        total += selection.proceeds

    if total == 0:
        return {"max_share": 0.0, "top_market": "N/A"}

    top_market = max(market_proceeds, key=market_proceeds.get)
    max_share = market_proceeds[top_market] / total

    return {"max_share": max_share, "top_market": top_market}


def _calculate_type_concentration(outcome: ProgramOutcome) -> dict:
    """Calculate asset type concentration metrics."""
    type_proceeds: dict = {}
    total = 0.0

    for selection in outcome.selected_assets:
        asset_type = selection.asset.asset_type
        type_proceeds[asset_type] = type_proceeds.get(asset_type, 0) + selection.proceeds
        total += selection.proceeds

    if total == 0:
        return {"max_share": 0.0, "top_type": None}

    top_type = max(type_proceeds, key=type_proceeds.get)
    max_share = type_proceeds[top_type] / total

    return {"max_share": max_share, "top_type": top_type}


def _is_nan(value: float) -> bool:
    """Check if value is NaN."""
    import math
    return math.isnan(value) if value is not None else False
