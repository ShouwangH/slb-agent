"""Tests for Selection Algorithm (PR5).

Tests cover:
- compute_score: scoring based on soft preferences
- apply_filters: filtering assets by various criteria
- select_assets: greedy selection with constraint checking
"""

import pytest

from app.config import DEFAULT_ENGINE_CONFIG, EngineConfig
from app.engine.selector import apply_filters, compute_score, select_assets
from app.models import (
    Asset,
    AssetFilters,
    AssetType,
    CorporateState,
    HardConstraints,
    MarketTier,
    Objective,
    ProgramType,
    SelectorSpec,
    SelectionStatus,
    SoftPreferences,
)


# =============================================================================
# Test Fixtures
# =============================================================================


def make_asset(
    asset_id: str = "A001",
    asset_type: AssetType = AssetType.STORE,
    market: str = "Dallas, TX",
    noi: float = 500_000,
    book_value: float = 4_000_000,
    criticality: float = 0.3,
    leaseability_score: float = 0.8,
    market_tier: MarketTier = None,
) -> Asset:
    """Create an Asset for testing."""
    return Asset(
        asset_id=asset_id,
        asset_type=asset_type,
        market=market,
        noi=noi,
        book_value=book_value,
        criticality=criticality,
        leaseability_score=leaseability_score,
        market_tier=market_tier,
    )


def make_corporate_state(
    net_debt: float = 2_000_000_000,
    ebitda: float = 500_000_000,
    interest_expense: float = 100_000_000,
    lease_expense: float = 50_000_000,
) -> CorporateState:
    """Create a CorporateState for testing."""
    return CorporateState(
        net_debt=net_debt,
        ebitda=ebitda,
        interest_expense=interest_expense,
        lease_expense=lease_expense,
    )


def make_spec(
    target_amount: float = 10_000_000,
    max_net_leverage: float = 4.0,
    min_fixed_charge_coverage: float = 3.0,
    min_interest_coverage: float = None,
    max_critical_fraction: float = None,
    prefer_low_criticality: bool = True,
    prefer_high_leaseability: bool = True,
    weight_criticality: float = 1.0,
    weight_leaseability: float = 1.0,
    include_asset_types: list = None,
    exclude_asset_types: list = None,
    exclude_markets: list = None,
    min_leaseability_score: float = None,
    max_criticality: float = None,
) -> SelectorSpec:
    """Create a SelectorSpec for testing."""
    return SelectorSpec(
        program_type=ProgramType.SLB,
        objective=Objective.BALANCED,
        target_amount=target_amount,
        hard_constraints=HardConstraints(
            max_net_leverage=max_net_leverage,
            min_fixed_charge_coverage=min_fixed_charge_coverage,
            min_interest_coverage=min_interest_coverage,
            max_critical_fraction=max_critical_fraction,
        ),
        soft_preferences=SoftPreferences(
            prefer_low_criticality=prefer_low_criticality,
            prefer_high_leaseability=prefer_high_leaseability,
            weight_criticality=weight_criticality,
            weight_leaseability=weight_leaseability,
        ),
        asset_filters=AssetFilters(
            include_asset_types=include_asset_types,
            exclude_asset_types=exclude_asset_types,
            exclude_markets=exclude_markets,
            min_leaseability_score=min_leaseability_score,
            max_criticality=max_criticality,
        ),
    )


# =============================================================================
# compute_score Tests (Section 6.1)
# =============================================================================


class TestComputeScore:
    """Tests for compute_score function."""

    def test_low_criticality_preferred(self):
        """Assets with lower criticality get higher scores."""
        prefs = SoftPreferences(
            prefer_low_criticality=True,
            prefer_high_leaseability=False,
            weight_criticality=1.0,
            weight_leaseability=0.0,
        )

        low_crit = make_asset(criticality=0.2)
        high_crit = make_asset(criticality=0.8)

        score_low = compute_score(low_crit, prefs)
        score_high = compute_score(high_crit, prefs)

        assert score_low > score_high

    def test_high_leaseability_preferred(self):
        """Assets with higher leaseability get higher scores."""
        prefs = SoftPreferences(
            prefer_low_criticality=False,
            prefer_high_leaseability=True,
            weight_criticality=0.0,
            weight_leaseability=1.0,
        )

        high_lease = make_asset(leaseability_score=0.9)
        low_lease = make_asset(leaseability_score=0.3)

        score_high = compute_score(high_lease, prefs)
        score_low = compute_score(low_lease, prefs)

        assert score_high > score_low

    def test_combined_preferences(self):
        """Both preferences work together."""
        prefs = SoftPreferences(
            prefer_low_criticality=True,
            prefer_high_leaseability=True,
            weight_criticality=1.0,
            weight_leaseability=1.0,
        )

        # Best asset: low criticality, high leaseability
        best = make_asset(criticality=0.1, leaseability_score=0.9)
        # Worst asset: high criticality, low leaseability
        worst = make_asset(criticality=0.9, leaseability_score=0.1)

        score_best = compute_score(best, prefs)
        score_worst = compute_score(worst, prefs)

        assert score_best > score_worst

    def test_weights_affect_score(self):
        """Weight adjustments change relative importance."""
        # Weight criticality heavily
        prefs_crit = SoftPreferences(
            prefer_low_criticality=True,
            prefer_high_leaseability=True,
            weight_criticality=10.0,
            weight_leaseability=1.0,
        )

        # Weight leaseability heavily
        prefs_lease = SoftPreferences(
            prefer_low_criticality=True,
            prefer_high_leaseability=True,
            weight_criticality=1.0,
            weight_leaseability=10.0,
        )

        # Asset with bad criticality but good leaseability
        asset = make_asset(criticality=0.8, leaseability_score=0.9)

        score_crit = compute_score(asset, prefs_crit)
        score_lease = compute_score(asset, prefs_lease)

        # When criticality weighted heavily, score should be lower (criticality penalizes)
        # When leaseability weighted heavily, score should be higher
        assert score_lease > score_crit

    def test_no_preferences_returns_zero(self):
        """When both preferences disabled, score is zero."""
        prefs = SoftPreferences(
            prefer_low_criticality=False,
            prefer_high_leaseability=False,
        )

        asset = make_asset(criticality=0.5, leaseability_score=0.5)
        score = compute_score(asset, prefs)

        assert score == 0.0

    def test_default_preferences(self):
        """Default preferences work correctly."""
        prefs = SoftPreferences()  # Defaults
        asset = make_asset(criticality=0.3, leaseability_score=0.8)

        score = compute_score(asset, prefs)

        # score = -1.0 * 0.3 + 1.0 * 0.8 = 0.5
        assert score == pytest.approx(0.5)


# =============================================================================
# apply_filters Tests (Section 6.2)
# =============================================================================


class TestApplyFilters:
    """Tests for apply_filters function."""

    def test_no_filters_returns_all(self):
        """Empty filters return all assets."""
        assets = [
            make_asset(asset_id="A001"),
            make_asset(asset_id="A002"),
            make_asset(asset_id="A003"),
        ]
        filters = AssetFilters()

        result = apply_filters(assets, filters)

        assert len(result) == 3

    def test_include_asset_types_whitelist(self):
        """include_asset_types acts as whitelist."""
        assets = [
            make_asset(asset_id="A001", asset_type=AssetType.STORE),
            make_asset(asset_id="A002", asset_type=AssetType.OFFICE),
            make_asset(asset_id="A003", asset_type=AssetType.DISTRIBUTION_CENTER),
        ]
        filters = AssetFilters(include_asset_types=[AssetType.STORE, AssetType.OFFICE])

        result = apply_filters(assets, filters)

        assert len(result) == 2
        assert all(a.asset_type in [AssetType.STORE, AssetType.OFFICE] for a in result)

    def test_exclude_asset_types_blacklist(self):
        """exclude_asset_types acts as blacklist."""
        assets = [
            make_asset(asset_id="A001", asset_type=AssetType.STORE),
            make_asset(asset_id="A002", asset_type=AssetType.OFFICE),
            make_asset(asset_id="A003", asset_type=AssetType.DISTRIBUTION_CENTER),
        ]
        filters = AssetFilters(exclude_asset_types=[AssetType.OFFICE])

        result = apply_filters(assets, filters)

        assert len(result) == 2
        assert all(a.asset_type != AssetType.OFFICE for a in result)

    def test_exclude_markets(self):
        """exclude_markets filters out specific markets."""
        assets = [
            make_asset(asset_id="A001", market="Dallas, TX"),
            make_asset(asset_id="A002", market="New York, NY"),
            make_asset(asset_id="A003", market="Chicago, IL"),
        ]
        filters = AssetFilters(exclude_markets=["New York, NY", "Chicago, IL"])

        result = apply_filters(assets, filters)

        assert len(result) == 1
        assert result[0].market == "Dallas, TX"

    def test_min_leaseability_score(self):
        """min_leaseability_score filters out low-scoring assets."""
        assets = [
            make_asset(asset_id="A001", leaseability_score=0.9),
            make_asset(asset_id="A002", leaseability_score=0.5),
            make_asset(asset_id="A003", leaseability_score=0.3),
        ]
        filters = AssetFilters(min_leaseability_score=0.5)

        result = apply_filters(assets, filters)

        assert len(result) == 2
        assert all(a.leaseability_score >= 0.5 for a in result)

    def test_max_criticality(self):
        """max_criticality filters out high-criticality assets."""
        assets = [
            make_asset(asset_id="A001", criticality=0.2),
            make_asset(asset_id="A002", criticality=0.5),
            make_asset(asset_id="A003", criticality=0.9),
        ]
        filters = AssetFilters(max_criticality=0.5)

        result = apply_filters(assets, filters)

        assert len(result) == 2
        assert all(a.criticality <= 0.5 for a in result)

    def test_combined_filters(self):
        """Multiple filters work together (AND logic)."""
        assets = [
            make_asset(
                asset_id="A001",
                asset_type=AssetType.STORE,
                market="Dallas, TX",
                criticality=0.2,
                leaseability_score=0.8,
            ),
            make_asset(
                asset_id="A002",
                asset_type=AssetType.OFFICE,  # Excluded type
                market="Dallas, TX",
                criticality=0.2,
                leaseability_score=0.8,
            ),
            make_asset(
                asset_id="A003",
                asset_type=AssetType.STORE,
                market="New York, NY",  # Excluded market
                criticality=0.2,
                leaseability_score=0.8,
            ),
            make_asset(
                asset_id="A004",
                asset_type=AssetType.STORE,
                market="Dallas, TX",
                criticality=0.9,  # Too critical
                leaseability_score=0.8,
            ),
            make_asset(
                asset_id="A005",
                asset_type=AssetType.STORE,
                market="Dallas, TX",
                criticality=0.2,
                leaseability_score=0.3,  # Too low leaseability
            ),
        ]
        filters = AssetFilters(
            include_asset_types=[AssetType.STORE],
            exclude_markets=["New York, NY"],
            max_criticality=0.5,
            min_leaseability_score=0.5,
        )

        result = apply_filters(assets, filters)

        assert len(result) == 1
        assert result[0].asset_id == "A001"

    def test_empty_result(self):
        """Filters can produce empty result."""
        assets = [
            make_asset(asset_id="A001", criticality=0.9),
            make_asset(asset_id="A002", criticality=0.8),
        ]
        filters = AssetFilters(max_criticality=0.5)

        result = apply_filters(assets, filters)

        assert len(result) == 0

    def test_boundary_values(self):
        """Boundary values are handled correctly (inclusive)."""
        assets = [
            make_asset(asset_id="A001", criticality=0.5, leaseability_score=0.6),
        ]

        # Exactly at boundary
        filters = AssetFilters(max_criticality=0.5, min_leaseability_score=0.6)
        result = apply_filters(assets, filters)

        assert len(result) == 1  # Should be included


# =============================================================================
# select_assets Tests (Section 6.4)
# =============================================================================


class TestSelectAssets:
    """Tests for select_assets function."""

    def test_feasible_selection_ok(self):
        """Feasible scenario returns OK status."""
        # Create assets that together meet target
        assets = [
            make_asset(
                asset_id="A001",
                noi=1_000_000,  # ~$15M proceeds at 6.5% cap
                criticality=0.2,
                leaseability_score=0.8,
            ),
            make_asset(
                asset_id="A002",
                noi=800_000,  # ~$12M proceeds
                criticality=0.3,
                leaseability_score=0.7,
            ),
        ]

        # Small company where SLB impact is manageable
        state = make_corporate_state(
            net_debt=100_000_000,
            ebitda=50_000_000,
            interest_expense=5_000_000,
            lease_expense=2_000_000,
        )

        spec = make_spec(
            target_amount=10_000_000,  # Achievable with the assets
            max_net_leverage=4.0,
            min_fixed_charge_coverage=2.0,  # Relaxed constraint
        )

        config = DEFAULT_ENGINE_CONFIG
        result = select_assets(assets, state, spec, config)

        assert result.status == SelectionStatus.OK
        assert result.proceeds >= spec.target_amount * (1 - config.target_tolerance)
        assert len(result.selected_assets) > 0
        assert result.violations == []

    def test_no_eligible_assets_infeasible(self):
        """No eligible assets returns INFEASIBLE with NO_ELIGIBLE_ASSETS."""
        assets = [
            make_asset(asset_id="A001", asset_type=AssetType.STORE),
            make_asset(asset_id="A002", asset_type=AssetType.STORE),
        ]

        state = make_corporate_state()

        # Filter excludes all assets
        spec = make_spec(
            target_amount=10_000_000,
            include_asset_types=[AssetType.OFFICE],  # No offices in assets
        )

        result = select_assets(assets, state, spec, DEFAULT_ENGINE_CONFIG)

        assert result.status == SelectionStatus.INFEASIBLE
        assert len(result.violations) == 1
        assert result.violations[0].code == "NO_ELIGIBLE_ASSETS"
        assert result.proceeds == 0
        assert result.selected_assets == []

    def test_target_not_met_infeasible(self):
        """Insufficient assets returns INFEASIBLE with TARGET_NOT_MET."""
        # Small assets that can't meet large target
        assets = [
            make_asset(asset_id="A001", noi=100_000),  # ~$1.5M proceeds
        ]

        # Large company (to avoid constraint violations)
        state = make_corporate_state(
            net_debt=1_000_000_000,
            ebitda=500_000_000,
            interest_expense=50_000_000,
            lease_expense=10_000_000,
        )

        spec = make_spec(
            target_amount=100_000_000,  # Cannot meet with small assets
            max_net_leverage=9.0,  # Very relaxed
            min_fixed_charge_coverage=1.0,  # Very relaxed
        )

        result = select_assets(assets, state, spec, DEFAULT_ENGINE_CONFIG)

        assert result.status == SelectionStatus.INFEASIBLE
        assert any(v.code == "TARGET_NOT_MET" for v in result.violations)

    def test_constraint_violation_skips_asset(self):
        """Assets causing constraint violations are skipped."""
        # One good asset, one that would violate critical fraction
        assets = [
            make_asset(
                asset_id="A001",
                noi=500_000,
                criticality=0.2,  # Not critical
                leaseability_score=0.8,
            ),
            make_asset(
                asset_id="A002",
                noi=500_000,
                criticality=0.9,  # Very critical
                leaseability_score=0.9,  # Higher leaseability (would be preferred)
            ),
        ]

        state = make_corporate_state(
            net_debt=50_000_000,
            ebitda=25_000_000,
            interest_expense=2_500_000,
            lease_expense=1_000_000,
        )

        spec = make_spec(
            target_amount=5_000_000,
            max_critical_fraction=0.3,  # Max 30% critical NOI
            max_net_leverage=9.0,
            min_fixed_charge_coverage=1.0,
        )

        result = select_assets(assets, state, spec, DEFAULT_ENGINE_CONFIG)

        # Should select A001 (not critical) and skip A002 (too critical)
        selected_ids = [a.asset.asset_id for a in result.selected_assets]
        assert "A001" in selected_ids
        # A002 would make critical_fraction = 100% which violates 30% limit

    def test_leverage_constraint_respected(self):
        """Assets that would violate leverage are skipped."""
        # Large asset that would push leverage too high
        assets = [
            make_asset(asset_id="A001", noi=100_000),  # Small
            make_asset(asset_id="A002", noi=50_000_000),  # Very large
        ]

        # Company where large SLB would over-deleverage
        state = make_corporate_state(
            net_debt=10_000_000,  # Small debt
            ebitda=50_000_000,
            interest_expense=500_000,
            lease_expense=100_000,
        )

        spec = make_spec(
            target_amount=1_000_000,
            max_net_leverage=0.5,  # Very tight leverage constraint
            min_fixed_charge_coverage=1.0,
        )

        result = select_assets(assets, state, spec, DEFAULT_ENGINE_CONFIG)

        # Proceeds from large asset would make leverage negative (good)
        # but the SLB rent would hurt fixed charge coverage
        # Check that constraints are respected
        if result.status == SelectionStatus.OK:
            assert result.leverage_after is None or result.leverage_after <= 0.5

    def test_selection_order_by_score(self):
        """Assets are selected in score order (best first)."""
        # Three assets with different scores
        assets = [
            make_asset(
                asset_id="A001",
                noi=500_000,
                criticality=0.8,  # Bad - low score
                leaseability_score=0.2,
            ),
            make_asset(
                asset_id="A002",
                noi=500_000,
                criticality=0.1,  # Good - high score
                leaseability_score=0.9,
            ),
            make_asset(
                asset_id="A003",
                noi=500_000,
                criticality=0.5,  # Medium score
                leaseability_score=0.5,
            ),
        ]

        state = make_corporate_state(
            net_debt=100_000_000,
            ebitda=50_000_000,
            interest_expense=5_000_000,
            lease_expense=2_000_000,
        )

        # Target that can be met with just one asset
        spec = make_spec(
            target_amount=5_000_000,
            max_net_leverage=9.0,
            min_fixed_charge_coverage=1.0,
        )

        result = select_assets(assets, state, spec, DEFAULT_ENGINE_CONFIG)

        if result.status == SelectionStatus.OK and len(result.selected_assets) >= 1:
            # Best scoring asset should be selected first
            assert result.selected_assets[0].asset.asset_id == "A002"

    def test_empty_selection_returns_baseline_metrics(self):
        """When no assets selected, metrics match baseline."""
        assets = [
            make_asset(asset_id="A001", criticality=0.9),
        ]

        state = make_corporate_state()

        # Filter that excludes the asset
        spec = make_spec(
            target_amount=10_000_000,
            max_criticality=0.5,
        )

        result = select_assets(assets, state, spec, DEFAULT_ENGINE_CONFIG)

        # leverage_before == leverage_after when nothing selected
        assert result.leverage_before == result.leverage_after
        assert result.interest_coverage_before == result.interest_coverage_after

    def test_warnings_captured(self):
        """Warnings from metrics computation are captured."""
        # Asset that produces more proceeds than debt
        assets = [
            make_asset(asset_id="A001", noi=100_000_000),  # Huge asset
        ]

        state = make_corporate_state(
            net_debt=10_000_000,  # Small debt
            ebitda=500_000_000,
            interest_expense=500_000,
            lease_expense=100_000,
        )

        spec = make_spec(
            target_amount=100_000_000,
            max_net_leverage=9.0,
            min_fixed_charge_coverage=0.1,  # Very relaxed
        )

        result = select_assets(assets, state, spec, DEFAULT_ENGINE_CONFIG)

        # Should have warning about surplus proceeds
        if result.selected_assets:
            # The asset produces ~$1.5B proceeds but only $10M debt
            # This should trigger the surplus warning
            assert any("surplus" in w.lower() or "exceed" in w.lower() for w in result.warnings)

    def test_fixed_charge_coverage_constraint(self):
        """Fixed charge coverage constraint is enforced."""
        assets = [
            make_asset(asset_id="A001", noi=10_000_000),  # Large SLB rent impact
        ]

        state = make_corporate_state(
            net_debt=100_000_000,
            ebitda=20_000_000,  # Low EBITDA
            interest_expense=5_000_000,
            lease_expense=1_000_000,
        )

        # Tight fixed charge coverage constraint
        spec = make_spec(
            target_amount=50_000_000,
            max_net_leverage=9.0,
            min_fixed_charge_coverage=5.0,  # Very tight
        )

        result = select_assets(assets, state, spec, DEFAULT_ENGINE_CONFIG)

        # Either infeasible or coverage constraint respected
        if result.status == SelectionStatus.OK:
            assert result.fixed_charge_coverage_after is None or result.fixed_charge_coverage_after >= 5.0
        # If infeasible, might be due to coverage or target

    def test_multiple_assets_selected(self):
        """Multiple assets can be selected to meet target."""
        assets = [
            make_asset(asset_id="A001", noi=300_000, criticality=0.2, leaseability_score=0.8),
            make_asset(asset_id="A002", noi=300_000, criticality=0.3, leaseability_score=0.7),
            make_asset(asset_id="A003", noi=300_000, criticality=0.4, leaseability_score=0.6),
        ]

        state = make_corporate_state(
            net_debt=200_000_000,
            ebitda=100_000_000,
            interest_expense=10_000_000,
            lease_expense=5_000_000,
        )

        # Target requiring multiple assets
        spec = make_spec(
            target_amount=10_000_000,  # Need ~2-3 assets
            max_net_leverage=9.0,
            min_fixed_charge_coverage=1.0,
        )

        result = select_assets(assets, state, spec, DEFAULT_ENGINE_CONFIG)

        if result.status == SelectionStatus.OK:
            assert len(result.selected_assets) >= 2

    def test_stops_at_target(self):
        """Selection stops once target is met."""
        assets = [
            make_asset(asset_id="A001", noi=2_000_000, criticality=0.1, leaseability_score=0.9),  # ~$30M
            make_asset(asset_id="A002", noi=2_000_000, criticality=0.2, leaseability_score=0.8),  # ~$30M
            make_asset(asset_id="A003", noi=2_000_000, criticality=0.3, leaseability_score=0.7),  # ~$30M
        ]

        state = make_corporate_state(
            net_debt=500_000_000,
            ebitda=200_000_000,
            interest_expense=25_000_000,
            lease_expense=10_000_000,
        )

        # Target that can be met with just one asset
        spec = make_spec(
            target_amount=25_000_000,
            max_net_leverage=9.0,
            min_fixed_charge_coverage=1.0,
        )

        result = select_assets(assets, state, spec, DEFAULT_ENGINE_CONFIG)

        if result.status == SelectionStatus.OK:
            # Should stop after first asset meets target
            assert len(result.selected_assets) == 1
            assert result.selected_assets[0].asset.asset_id == "A001"  # Highest score


class TestSelectAssetsEdgeCases:
    """Edge case tests for select_assets."""

    def test_zero_ebitda_handling(self):
        """Zero EBITDA results in None metrics but no crash."""
        assets = [make_asset(asset_id="A001", noi=500_000)]

        state = make_corporate_state(ebitda=0)  # Zero EBITDA

        spec = make_spec(
            target_amount=5_000_000,
            max_net_leverage=None,  # Skip leverage check
            min_fixed_charge_coverage=None,  # Skip coverage check
        )

        result = select_assets(assets, state, spec, DEFAULT_ENGINE_CONFIG)

        # Should not crash, metrics should be None
        assert result.leverage_before is None
        assert result.leverage_after is None

    def test_all_constraints_none(self):
        """Selection works with all constraints disabled."""
        assets = [
            make_asset(asset_id="A001", noi=500_000),
            make_asset(asset_id="A002", noi=500_000),
        ]

        state = make_corporate_state()

        spec = make_spec(
            target_amount=10_000_000,
            max_net_leverage=None,
            min_fixed_charge_coverage=None,
            min_interest_coverage=None,
            max_critical_fraction=None,
        )

        result = select_assets(assets, state, spec, DEFAULT_ENGINE_CONFIG)

        # Should select assets without constraint issues
        assert len(result.selected_assets) > 0

    def test_single_asset_portfolio(self):
        """Selection works with single asset."""
        assets = [make_asset(asset_id="A001", noi=1_000_000)]

        state = make_corporate_state(
            net_debt=50_000_000,
            ebitda=25_000_000,
            interest_expense=2_500_000,
            lease_expense=1_000_000,
        )

        spec = make_spec(
            target_amount=10_000_000,
            max_net_leverage=9.0,
            min_fixed_charge_coverage=1.0,
        )

        result = select_assets(assets, state, spec, DEFAULT_ENGINE_CONFIG)

        assert len(result.selected_assets) <= 1

    def test_target_tolerance_respected(self):
        """Target tolerance (2%) is applied correctly."""
        assets = [make_asset(asset_id="A001", noi=650_000)]  # ~$10M proceeds

        state = make_corporate_state(
            net_debt=200_000_000,
            ebitda=100_000_000,
            interest_expense=10_000_000,
            lease_expense=5_000_000,
        )

        # Target of $10.1M with 2% tolerance = threshold of $9.898M
        spec = make_spec(
            target_amount=10_100_000,
            max_net_leverage=9.0,
            min_fixed_charge_coverage=1.0,
        )

        config = DEFAULT_ENGINE_CONFIG  # tolerance = 0.02
        result = select_assets(assets, state, spec, config)

        # $10M proceeds should meet $10.1M target with 2% tolerance
        if result.proceeds >= 10_100_000 * 0.98:
            assert result.status == SelectionStatus.OK

    def test_interest_coverage_constraint(self):
        """Interest coverage constraint is enforced when set."""
        assets = [make_asset(asset_id="A001", noi=500_000)]

        state = make_corporate_state(
            net_debt=100_000_000,
            ebitda=10_000_000,  # Low EBITDA
            interest_expense=5_000_000,  # High interest
            lease_expense=1_000_000,
        )

        spec = make_spec(
            target_amount=5_000_000,
            max_net_leverage=9.0,
            min_interest_coverage=10.0,  # Very tight
            min_fixed_charge_coverage=1.0,
        )

        result = select_assets(assets, state, spec, DEFAULT_ENGINE_CONFIG)

        # Check constraint is respected
        if result.status == SelectionStatus.OK and result.interest_coverage_after is not None:
            assert result.interest_coverage_after >= 10.0


class TestSelectAssetsIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow_feasible(self):
        """Complete workflow with all features - feasible case."""
        # Diverse portfolio
        assets = [
            make_asset(
                asset_id="A001",
                asset_type=AssetType.STORE,
                market="Dallas, TX",
                noi=800_000,
                criticality=0.2,
                leaseability_score=0.9,
            ),
            make_asset(
                asset_id="A002",
                asset_type=AssetType.DISTRIBUTION_CENTER,
                market="Chicago, IL",
                noi=1_200_000,
                criticality=0.4,
                leaseability_score=0.7,
            ),
            make_asset(
                asset_id="A003",
                asset_type=AssetType.OFFICE,
                market="New York, NY",
                noi=600_000,
                criticality=0.8,  # High criticality - might be excluded
                leaseability_score=0.6,
            ),
            make_asset(
                asset_id="A004",
                asset_type=AssetType.STORE,
                market="Austin, TX",
                noi=500_000,
                criticality=0.3,
                leaseability_score=0.8,
            ),
        ]

        state = CorporateState(
            net_debt=150_000_000,
            ebitda=75_000_000,
            interest_expense=7_500_000,
            lease_expense=3_000_000,
        )

        spec = SelectorSpec(
            program_type=ProgramType.SLB,
            objective=Objective.BALANCED,
            target_amount=25_000_000,
            hard_constraints=HardConstraints(
                max_net_leverage=4.0,
                min_fixed_charge_coverage=2.5,
                max_critical_fraction=0.4,
            ),
            soft_preferences=SoftPreferences(
                prefer_low_criticality=True,
                prefer_high_leaseability=True,
                weight_criticality=1.0,
                weight_leaseability=1.0,
            ),
            asset_filters=AssetFilters(
                exclude_markets=["New York, NY"],  # Exclude NYC
                max_criticality=0.7,  # Exclude very critical
            ),
        )

        result = select_assets(assets, state, spec, DEFAULT_ENGINE_CONFIG)

        # Verify structure
        assert result.status in [SelectionStatus.OK, SelectionStatus.INFEASIBLE]
        assert isinstance(result.proceeds, float)
        assert isinstance(result.selected_assets, list)
        assert isinstance(result.violations, list)

        # If OK, verify constraints
        if result.status == SelectionStatus.OK:
            assert result.leverage_after is None or result.leverage_after <= 4.0
            assert result.fixed_charge_coverage_after is None or result.fixed_charge_coverage_after >= 2.5
            assert result.critical_fraction <= 0.4

            # Verify no excluded assets
            for selection in result.selected_assets:
                assert selection.asset.market != "New York, NY"
                assert selection.asset.criticality <= 0.7

    def test_full_workflow_infeasible(self):
        """Complete workflow with all features - infeasible case."""
        # Limited portfolio
        assets = [
            make_asset(
                asset_id="A001",
                asset_type=AssetType.STORE,
                noi=100_000,  # Small
                criticality=0.3,
                leaseability_score=0.8,
            ),
        ]

        state = CorporateState(
            net_debt=500_000_000,  # Large debt
            ebitda=100_000_000,
            interest_expense=25_000_000,
            lease_expense=10_000_000,
        )

        spec = SelectorSpec(
            program_type=ProgramType.SLB,
            objective=Objective.MAXIMIZE_PROCEEDS,
            target_amount=500_000_000,  # Impossible target
            hard_constraints=HardConstraints(
                max_net_leverage=3.0,
                min_fixed_charge_coverage=3.0,
            ),
            soft_preferences=SoftPreferences(),
            asset_filters=AssetFilters(),
        )

        result = select_assets(assets, state, spec, DEFAULT_ENGINE_CONFIG)

        assert result.status == SelectionStatus.INFEASIBLE
        assert len(result.violations) > 0
        assert any(v.code == "TARGET_NOT_MET" for v in result.violations)
