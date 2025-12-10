"""
Selection algorithm for SLB Agent.

This module contains the asset selection logic as defined in DESIGN.md Section 6.
All functions are pure and deterministic.

Key components:
- compute_score: Scores assets based on soft preferences
- apply_filters: Filters assets based on AssetFilters criteria
- select_assets: Greedy selection algorithm (single entry point for selection)
"""

import math
from typing import Tuple

from app.config import EngineConfig
from app.engine.metrics import (
    check_constraints,
    compute_asset_slb_metrics,
    compute_baseline_metrics,
)
from app.models import (
    Asset,
    AssetFilters,
    AssetSelection,
    ConstraintViolation,
    CorporateState,
    ProgramOutcome,
    SelectorSpec,
    SelectionStatus,
    SoftPreferences,
)


# =============================================================================
# Scoring Function (Section 6.1)
# =============================================================================


def compute_score(
    asset: Asset,
    preferences: SoftPreferences,
) -> float:
    """
    Compute selection score for an asset based on soft preferences.

    Higher score = more preferred for selection.

    Args:
        asset: The asset to score
        preferences: Soft preferences with weights and preference flags

    Returns:
        Float score (higher = better)
    """
    score = 0.0

    # Criticality: prefer low (score decreases with criticality)
    if preferences.prefer_low_criticality:
        score -= preferences.weight_criticality * asset.criticality

    # Leaseability: prefer high (score increases with leaseability)
    if preferences.prefer_high_leaseability:
        score += preferences.weight_leaseability * asset.leaseability_score

    return score


# =============================================================================
# Filter Application (Section 6.2)
# =============================================================================


def apply_filters(
    assets: list[Asset],
    filters: AssetFilters,
) -> list[Asset]:
    """
    Apply pre-selection filters to asset list.

    Args:
        assets: List of assets to filter
        filters: Filter criteria

    Returns:
        Filtered list of eligible assets
    """
    eligible = assets

    # Whitelist: only include specified types (None = all)
    if filters.include_asset_types:
        eligible = [a for a in eligible if a.asset_type in filters.include_asset_types]

    # Blacklist: exclude specified types
    if filters.exclude_asset_types:
        eligible = [a for a in eligible if a.asset_type not in filters.exclude_asset_types]

    # Exclude specific markets
    if filters.exclude_markets:
        eligible = [a for a in eligible if a.market not in filters.exclude_markets]

    # Minimum leaseability threshold
    if filters.min_leaseability_score is not None:
        eligible = [a for a in eligible if a.leaseability_score >= filters.min_leaseability_score]

    # Maximum criticality threshold
    if filters.max_criticality is not None:
        eligible = [a for a in eligible if a.criticality <= filters.max_criticality]

    return eligible


# =============================================================================
# Selection Algorithm (Section 6.4)
# =============================================================================


def select_assets(
    assets: list[Asset],
    corporate_state: CorporateState,
    spec: SelectorSpec,
    config: EngineConfig,
) -> ProgramOutcome:
    """
    Greedy asset selection algorithm.

    Adds assets in score order until target met, skipping any that would
    violate hard constraints. This is the SINGLE entry point for selection
    - API/UI/LLM layers must not recompute metrics.

    Args:
        assets: Available assets to select from
        corporate_state: Pre-transaction corporate financial state
        spec: Selection specification with constraints and preferences
        config: Engine configuration

    Returns:
        ProgramOutcome with selected assets, metrics, and status
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
    candidates: list[Tuple[Asset, float, float, float]] = []  # (asset, proceeds, slb_rent, score)
    for asset in eligible:
        slb_metrics = compute_asset_slb_metrics(asset, config)
        score = compute_score(asset, spec.soft_preferences)
        candidates.append((asset, slb_metrics.proceeds, slb_metrics.slb_rent, score))

    # Step 4: Sort by score descending
    candidates.sort(key=lambda x: x[3], reverse=True)

    # Step 5: Greedy selection
    selected: list[AssetSelection] = []
    all_warnings: list[str] = []

    for asset, proceeds, slb_rent, score in candidates:
        # Tentatively add this asset
        tentative = selected + [AssetSelection(
            asset=asset,
            proceeds=proceeds,
            slb_rent=slb_rent,
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
        if metrics.total_proceeds >= spec.target_amount * (1 - config.target_tolerance):
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
    if math.isnan(final_metrics.total_proceeds) or math.isinf(final_metrics.total_proceeds):
        status = SelectionStatus.NUMERIC_ERROR
    elif final_violations:
        status = SelectionStatus.INFEASIBLE
    else:
        status = SelectionStatus.OK

    return ProgramOutcome(
        status=status,
        selected_assets=selected,
        proceeds=final_metrics.total_proceeds,
        leverage_before=final_metrics.baseline.leverage,
        leverage_after=final_metrics.post.leverage,
        interest_coverage_before=final_metrics.baseline.interest_coverage,
        interest_coverage_after=final_metrics.post.interest_coverage,
        fixed_charge_coverage_before=final_metrics.baseline.fixed_charge_coverage,
        fixed_charge_coverage_after=final_metrics.post.fixed_charge_coverage,
        critical_fraction=final_metrics.critical_fraction,
        violations=final_violations,
        warnings=all_warnings,
    )
