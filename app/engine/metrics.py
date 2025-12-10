"""
Metrics computation engine for SLB Agent.

This module contains pure functions for all metric computation as defined
in DESIGN.md Sections 5.2-5.3 and 6.3.

All functions are:
- Pure (no side effects)
- Deterministic (same inputs → same outputs)
- Return warnings as lists, not logged
"""

from typing import Optional, Tuple

from app.config import EngineConfig
from app.models import (
    Asset,
    AssetSelection,
    AssetSLBMetrics,
    BaselineMetrics,
    ConstraintViolation,
    CorporateState,
    HardConstraints,
    PortfolioMetrics,
    PostMetrics,
)


# =============================================================================
# Per-Asset Metrics (Section 5.2)
# =============================================================================


def compute_asset_slb_metrics(
    asset: Asset,
    config: EngineConfig,
) -> AssetSLBMetrics:
    """
    Compute SLB economics for a single asset.

    Args:
        asset: The asset to compute metrics for
        config: Engine configuration with cap rates and parameters

    Returns:
        AssetSLBMetrics with market_value, proceeds, slb_rent, cap_rate
    """
    # Use asset's market tier or default from config
    tier = asset.market_tier or config.default_market_tier

    # Look up cap rate from curve
    cap_rate = config.cap_rate_curve[asset.asset_type][tier]

    # Market value = NOI / cap rate
    market_value = asset.noi / cap_rate

    # Proceeds = market value minus transaction costs
    proceeds = market_value * (1 - config.transaction_haircut)

    # SLB rent uses multiplier to allow aggressive/conservative modeling
    slb_rent = asset.noi * config.slb_rent_multiplier

    return AssetSLBMetrics(
        market_value=market_value,
        proceeds=proceeds,
        slb_rent=slb_rent,
        cap_rate=cap_rate,
    )


# =============================================================================
# Baseline Metrics (Section 5.3)
# =============================================================================


def compute_baseline_metrics(
    state: CorporateState,
    config: EngineConfig,
) -> BaselineMetrics:
    """
    Compute pre-transaction metrics from corporate state.

    Applies numeric guardrails: returns None for metrics with
    denominators near zero.

    Args:
        state: Corporate financial state
        config: Engine configuration with epsilon threshold

    Returns:
        BaselineMetrics with leverage and coverage ratios (None if undefined)
    """
    # Leverage = net_debt / ebitda
    if abs(state.ebitda) < config.epsilon:
        leverage = None  # Not meaningful
    else:
        leverage = state.net_debt / state.ebitda

    # Interest coverage = ebitda / interest_expense
    if abs(state.interest_expense) < config.epsilon:
        interest_coverage = None  # Not meaningful (no interest)
    else:
        interest_coverage = state.ebitda / state.interest_expense

    # Fixed charge coverage = ebitda / (interest + lease_expense)
    total_fixed_charges = state.interest_expense + (state.lease_expense or 0)
    if abs(total_fixed_charges) < config.epsilon:
        fixed_charge_coverage = None  # Not meaningful
    else:
        fixed_charge_coverage = state.ebitda / total_fixed_charges

    return BaselineMetrics(
        leverage=leverage,
        interest_coverage=interest_coverage,
        fixed_charge_coverage=fixed_charge_coverage,
    )


# =============================================================================
# Post-Transaction Metrics (Section 5.3)
# =============================================================================


def compute_post_transaction_metrics(
    state: CorporateState,
    selected: list[AssetSelection],
    config: EngineConfig,
) -> Tuple[PostMetrics, list[str]]:
    """
    Compute post-transaction metrics after SLB execution.

    Applies numeric guardrails:
    - Caps debt repayment at net_debt (emits warning)
    - Clamps interest to zero if reduction exceeds expense (emits warning)
    - Returns None for metrics with denominators near zero

    Args:
        state: Pre-transaction corporate state
        selected: List of selected assets with their SLB economics
        config: Engine configuration

    Returns:
        Tuple of (PostMetrics, warnings list)
    """
    warnings: list[str] = []

    # Aggregate proceeds and rent from selected assets
    total_proceeds = sum(a.proceeds for a in selected)
    total_slb_rent = sum(a.slb_rent for a in selected)

    # Debt repayment with guardrails
    debt_repaid = total_proceeds  # Assume 100% to debt paydown

    # Guard: cannot repay more debt than exists
    if debt_repaid > state.net_debt:
        surplus = debt_repaid - state.net_debt
        warnings.append(
            f"Proceeds (${debt_repaid:,.0f}) exceed net_debt (${state.net_debt:,.0f}); "
            f"surplus ${surplus:,.0f} ignored for debt paydown"
        )
        debt_repaid = state.net_debt

    net_debt_after = state.net_debt - debt_repaid

    # Interest reduction with clamping
    interest_reduction = debt_repaid * config.avg_cost_of_debt
    interest_after = state.interest_expense - interest_reduction

    if interest_after < 0:
        warnings.append(
            f"Interest reduction (${interest_reduction:,.0f}) exceeds interest_expense "
            f"(${state.interest_expense:,.0f}); clamped to zero"
        )
        interest_after = 0

    # EBITDA unchanged (demo simplification per DESIGN.md Section 5.4)
    ebitda = state.ebitda

    # Lease expense increases by SLB rent
    lease_expense_after = (state.lease_expense or 0) + total_slb_rent

    # Compute coverage metrics with guardrails
    if abs(ebitda) < config.epsilon:
        leverage_after: Optional[float] = None
        interest_coverage_after: Optional[float] = None
        fixed_charge_coverage_after: Optional[float] = None
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
        net_debt=net_debt_after,
        interest_expense=interest_after,
        total_lease_expense=lease_expense_after,
        leverage=leverage_after,
        interest_coverage=interest_coverage_after,
        fixed_charge_coverage=fixed_charge_coverage_after,
    ), warnings


# =============================================================================
# Critical Fraction (Section 5.3)
# =============================================================================


def compute_critical_fraction(
    selected: list[AssetSelection],
    config: EngineConfig,
) -> float:
    """
    Compute fraction of selected NOI from critical assets.

    Critical assets are those with criticality > config.criticality_threshold.

    Args:
        selected: List of selected assets
        config: Engine configuration with criticality_threshold

    Returns:
        Fraction of NOI from critical assets (0.0 if no assets selected)
    """
    if not selected:
        return 0.0

    critical_noi = sum(
        a.asset.noi for a in selected
        if a.asset.criticality > config.criticality_threshold
    )
    total_noi = sum(a.asset.noi for a in selected)

    if abs(total_noi) < config.epsilon:
        return 0.0  # No NOI (shouldn't happen with valid assets)

    return critical_noi / total_noi


# =============================================================================
# Constraint Checking (Section 6.3)
# =============================================================================


def check_constraints(
    selected: list[AssetSelection],
    corporate_state: CorporateState,
    hard_constraints: HardConstraints,
    target_amount: float,
    config: EngineConfig,
) -> Tuple[PortfolioMetrics, list[ConstraintViolation], list[str]]:
    """
    Compute all metrics and check hard constraints.

    Args:
        selected: List of selected assets with SLB economics
        corporate_state: Pre-transaction corporate state
        hard_constraints: Constraints to check
        target_amount: Target proceeds amount
        config: Engine configuration

    Returns:
        Tuple of (PortfolioMetrics, violations list, warnings list)
    """
    # Compute all metrics
    baseline = compute_baseline_metrics(corporate_state, config)
    post_metrics, warnings = compute_post_transaction_metrics(
        corporate_state, selected, config
    )
    critical_fraction = compute_critical_fraction(selected, config)

    # Compute totals
    total_proceeds = sum(a.proceeds for a in selected)
    total_slb_rent = sum(a.slb_rent for a in selected)

    violations: list[ConstraintViolation] = []

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
                detail=f"Post-SLB leverage {post_metrics.leverage:.2f}x exceeds limit {hard_constraints.max_net_leverage:.2f}x",
                actual=post_metrics.leverage,
                limit=hard_constraints.max_net_leverage,
            ))

    # Check interest coverage
    if hard_constraints.min_interest_coverage is not None:
        if post_metrics.interest_coverage is None:
            # Interest ≈ 0 means effectively infinite coverage; not a violation
            pass
        elif post_metrics.interest_coverage < hard_constraints.min_interest_coverage:
            violations.append(ConstraintViolation(
                code="MIN_INTEREST_COVERAGE",
                detail=f"Post-SLB interest coverage {post_metrics.interest_coverage:.2f}x below minimum {hard_constraints.min_interest_coverage:.2f}x",
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
                detail=f"Post-SLB fixed charge coverage {post_metrics.fixed_charge_coverage:.2f}x below minimum {hard_constraints.min_fixed_charge_coverage:.2f}x",
                actual=post_metrics.fixed_charge_coverage,
                limit=hard_constraints.min_fixed_charge_coverage,
            ))

    # Check critical fraction
    if hard_constraints.max_critical_fraction is not None:
        if critical_fraction > hard_constraints.max_critical_fraction:
            violations.append(ConstraintViolation(
                code="MAX_CRITICAL_FRACTION",
                detail=f"Critical asset concentration {critical_fraction:.1%} exceeds limit {hard_constraints.max_critical_fraction:.1%}",
                actual=critical_fraction,
                limit=hard_constraints.max_critical_fraction,
            ))

    # Check target met
    target_threshold = target_amount * (1 - config.target_tolerance)
    if total_proceeds < target_threshold:
        violations.append(ConstraintViolation(
            code="TARGET_NOT_MET",
            detail=f"Proceeds ${total_proceeds:,.0f} below target ${target_amount:,.0f} (threshold ${target_threshold:,.0f})",
            actual=total_proceeds,
            limit=target_threshold,
        ))

    # Assemble full metrics object
    metrics = PortfolioMetrics(
        baseline=baseline,
        post=post_metrics,
        total_proceeds=total_proceeds,
        total_slb_rent=total_slb_rent,
        critical_fraction=critical_fraction,
    )

    return metrics, violations, warnings
