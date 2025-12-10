"""
Tests for app/engine/metrics.py

Covers:
- Asset metrics: known inputs → expected outputs, different types/tiers
- Baseline metrics: normal inputs, ebitda ≈ 0 → None
- Post-transaction: various proceeds, over-repayment warning, interest clamping
- Idempotence: empty selection → before == after
- Constraints: each type passing/failing, None handling, multiple violations
"""

import math

import pytest

from app.config import DEFAULT_ENGINE_CONFIG, EngineConfig
from app.engine.metrics import (
    check_constraints,
    compute_asset_slb_metrics,
    compute_baseline_metrics,
    compute_critical_fraction,
    compute_post_transaction_metrics,
)
from app.models import (
    Asset,
    AssetSelection,
    AssetType,
    CorporateState,
    HardConstraints,
    MarketTier,
)


# =============================================================================
# Test Fixtures
# =============================================================================


def make_asset(
    asset_id: str = "A001",
    asset_type: AssetType = AssetType.STORE,
    market_tier: MarketTier = None,
    noi: float = 500_000,
    criticality: float = 0.3,
    leaseability_score: float = 0.8,
) -> Asset:
    """Create a test asset."""
    return Asset(
        asset_id=asset_id,
        asset_type=asset_type,
        market="Dallas, TX",
        market_tier=market_tier,
        noi=noi,
        book_value=noi * 10,  # Arbitrary
        criticality=criticality,
        leaseability_score=leaseability_score,
    )


def make_corporate_state(
    net_debt: float = 2_000_000_000,
    ebitda: float = 500_000_000,
    interest_expense: float = 100_000_000,
    lease_expense: float = None,
) -> CorporateState:
    """Create a test corporate state."""
    return CorporateState(
        net_debt=net_debt,
        ebitda=ebitda,
        interest_expense=interest_expense,
        lease_expense=lease_expense,
    )


def make_selection(asset: Asset, config: EngineConfig = None) -> AssetSelection:
    """Create an AssetSelection from an Asset using computed metrics."""
    config = config or DEFAULT_ENGINE_CONFIG
    metrics = compute_asset_slb_metrics(asset, config)
    return AssetSelection(
        asset=asset,
        proceeds=metrics.proceeds,
        slb_rent=metrics.slb_rent,
    )


# =============================================================================
# compute_asset_slb_metrics Tests
# =============================================================================


class TestComputeAssetSLBMetrics:
    """Test compute_asset_slb_metrics function."""

    def test_basic_calculation(self):
        """Test basic SLB metrics calculation."""
        asset = make_asset(
            asset_type=AssetType.STORE,
            market_tier=MarketTier.TIER_2,
            noi=650_000,
        )
        config = DEFAULT_ENGINE_CONFIG

        metrics = compute_asset_slb_metrics(asset, config)

        # Cap rate for STORE/TIER_2 = 0.065
        expected_cap_rate = 0.065
        expected_market_value = 650_000 / 0.065  # = 10,000,000
        expected_proceeds = expected_market_value * (1 - 0.025)  # = 9,750,000
        expected_rent = 650_000 * 1.0  # multiplier = 1.0

        assert metrics.cap_rate == expected_cap_rate
        assert metrics.market_value == pytest.approx(expected_market_value)
        assert metrics.proceeds == pytest.approx(expected_proceeds)
        assert metrics.slb_rent == pytest.approx(expected_rent)

    def test_different_asset_types(self):
        """Different asset types have different cap rates."""
        config = DEFAULT_ENGINE_CONFIG

        store = make_asset(asset_type=AssetType.STORE, market_tier=MarketTier.TIER_1, noi=1_000_000)
        dc = make_asset(asset_type=AssetType.DISTRIBUTION_CENTER, market_tier=MarketTier.TIER_1, noi=1_000_000)
        office = make_asset(asset_type=AssetType.OFFICE, market_tier=MarketTier.TIER_1, noi=1_000_000)

        store_metrics = compute_asset_slb_metrics(store, config)
        dc_metrics = compute_asset_slb_metrics(dc, config)
        office_metrics = compute_asset_slb_metrics(office, config)

        # Distribution centers have lowest cap rates (highest values)
        assert dc_metrics.cap_rate < store_metrics.cap_rate
        assert dc_metrics.market_value > store_metrics.market_value

        # Office has highest cap rate of these three
        assert office_metrics.cap_rate > store_metrics.cap_rate

    def test_different_market_tiers(self):
        """Higher tiers have higher cap rates (lower values)."""
        config = DEFAULT_ENGINE_CONFIG
        noi = 1_000_000

        tier1 = make_asset(market_tier=MarketTier.TIER_1, noi=noi)
        tier2 = make_asset(market_tier=MarketTier.TIER_2, noi=noi)
        tier3 = make_asset(market_tier=MarketTier.TIER_3, noi=noi)

        m1 = compute_asset_slb_metrics(tier1, config)
        m2 = compute_asset_slb_metrics(tier2, config)
        m3 = compute_asset_slb_metrics(tier3, config)

        assert m1.cap_rate < m2.cap_rate < m3.cap_rate
        assert m1.market_value > m2.market_value > m3.market_value

    def test_default_market_tier(self):
        """Assets without market_tier use config default."""
        config = DEFAULT_ENGINE_CONFIG
        asset = make_asset(market_tier=None)  # No tier specified

        metrics = compute_asset_slb_metrics(asset, config)

        # Default is TIER_2
        expected_cap_rate = config.cap_rate_curve[AssetType.STORE][MarketTier.TIER_2]
        assert metrics.cap_rate == expected_cap_rate

    def test_custom_rent_multiplier(self):
        """Custom SLB rent multiplier affects rent calculation."""
        asset = make_asset(noi=1_000_000)

        # Aggressive terms (rent below NOI)
        aggressive_config = EngineConfig(slb_rent_multiplier=0.9)
        aggressive_metrics = compute_asset_slb_metrics(asset, aggressive_config)
        assert aggressive_metrics.slb_rent == pytest.approx(900_000)

        # Conservative terms (rent above NOI)
        conservative_config = EngineConfig(slb_rent_multiplier=1.1)
        conservative_metrics = compute_asset_slb_metrics(asset, conservative_config)
        assert conservative_metrics.slb_rent == pytest.approx(1_100_000)

    def test_custom_transaction_haircut(self):
        """Custom transaction haircut affects proceeds."""
        asset = make_asset(noi=650_000, market_tier=MarketTier.TIER_2)

        # Higher haircut
        config = EngineConfig(transaction_haircut=0.05)  # 5%
        metrics = compute_asset_slb_metrics(asset, config)

        expected_market_value = 650_000 / 0.065
        expected_proceeds = expected_market_value * 0.95
        assert metrics.proceeds == pytest.approx(expected_proceeds)


# =============================================================================
# compute_baseline_metrics Tests
# =============================================================================


class TestComputeBaselineMetrics:
    """Test compute_baseline_metrics function."""

    def test_normal_calculation(self):
        """Test baseline metrics with normal inputs."""
        state = make_corporate_state(
            net_debt=2_000_000_000,
            ebitda=500_000_000,
            interest_expense=100_000_000,
            lease_expense=50_000_000,
        )
        config = DEFAULT_ENGINE_CONFIG

        metrics = compute_baseline_metrics(state, config)

        # leverage = net_debt / ebitda = 2B / 500M = 4.0
        assert metrics.leverage == pytest.approx(4.0)

        # interest_coverage = ebitda / interest = 500M / 100M = 5.0
        assert metrics.interest_coverage == pytest.approx(5.0)

        # fixed_charge_coverage = ebitda / (interest + lease) = 500M / 150M = 3.33
        assert metrics.fixed_charge_coverage == pytest.approx(3.333, rel=0.01)

    def test_zero_ebitda_returns_none_leverage(self):
        """Zero EBITDA results in None leverage but valid coverage."""
        state = make_corporate_state(ebitda=0)
        config = DEFAULT_ENGINE_CONFIG

        metrics = compute_baseline_metrics(state, config)

        # Leverage is undefined when EBITDA ≈ 0 (denominator)
        assert metrics.leverage is None
        # Interest coverage = 0 / interest = 0.0 (valid, means no coverage)
        assert metrics.interest_coverage == pytest.approx(0.0)
        # Fixed charge coverage = 0 / (interest + lease) = 0.0
        assert metrics.fixed_charge_coverage == pytest.approx(0.0)

    def test_near_zero_ebitda_returns_none(self):
        """Near-zero EBITDA (< epsilon) results in None."""
        state = make_corporate_state(ebitda=1e-10)
        config = DEFAULT_ENGINE_CONFIG

        metrics = compute_baseline_metrics(state, config)

        assert metrics.leverage is None

    def test_zero_interest_returns_none_coverage(self):
        """Zero interest expense results in None interest coverage."""
        state = make_corporate_state(
            interest_expense=0,
            lease_expense=50_000_000,
        )
        config = DEFAULT_ENGINE_CONFIG

        metrics = compute_baseline_metrics(state, config)

        assert metrics.leverage is not None
        assert metrics.interest_coverage is None  # No interest
        assert metrics.fixed_charge_coverage is not None  # Still has lease expense

    def test_no_fixed_charges_returns_none(self):
        """Zero interest and zero lease expense results in None fixed charge coverage."""
        state = make_corporate_state(
            interest_expense=0,
            lease_expense=0,
        )
        config = DEFAULT_ENGINE_CONFIG

        metrics = compute_baseline_metrics(state, config)

        assert metrics.fixed_charge_coverage is None

    def test_negative_ebitda(self):
        """Negative EBITDA (distressed) still computes metrics."""
        state = make_corporate_state(
            net_debt=1_000_000_000,
            ebitda=-100_000_000,  # Loss
            interest_expense=50_000_000,
        )
        config = DEFAULT_ENGINE_CONFIG

        metrics = compute_baseline_metrics(state, config)

        # Leverage = 1B / -100M = -10 (negative leverage indicates distress)
        assert metrics.leverage == pytest.approx(-10.0)

        # Interest coverage = -100M / 50M = -2 (negative coverage)
        assert metrics.interest_coverage == pytest.approx(-2.0)

    def test_none_lease_expense_treated_as_zero(self):
        """None lease_expense is treated as zero."""
        state = make_corporate_state(
            interest_expense=100_000_000,
            lease_expense=None,
        )
        config = DEFAULT_ENGINE_CONFIG

        metrics = compute_baseline_metrics(state, config)

        # fixed_charge_coverage = ebitda / (interest + 0)
        expected = 500_000_000 / 100_000_000
        assert metrics.fixed_charge_coverage == pytest.approx(expected)


# =============================================================================
# compute_post_transaction_metrics Tests
# =============================================================================


class TestComputePostTransactionMetrics:
    """Test compute_post_transaction_metrics function."""

    def test_normal_transaction(self):
        """Test post-transaction metrics with normal inputs."""
        state = make_corporate_state(
            net_debt=2_000_000_000,
            ebitda=500_000_000,
            interest_expense=100_000_000,
            lease_expense=10_000_000,
        )
        config = DEFAULT_ENGINE_CONFIG

        # Create selections with known proceeds/rent
        asset = make_asset(noi=650_000, market_tier=MarketTier.TIER_2)
        selection = make_selection(asset, config)

        # Single asset: ~9.75M proceeds, 650K rent
        post, warnings = compute_post_transaction_metrics(state, [selection], config)

        assert len(warnings) == 0

        # Net debt reduced by proceeds
        expected_net_debt = 2_000_000_000 - selection.proceeds
        assert post.net_debt == pytest.approx(expected_net_debt)

        # Interest reduced by debt_repaid * avg_cost_of_debt
        expected_interest_reduction = selection.proceeds * 0.06
        expected_interest = 100_000_000 - expected_interest_reduction
        assert post.interest_expense == pytest.approx(expected_interest)

        # Lease expense increased by SLB rent
        expected_lease = 10_000_000 + selection.slb_rent
        assert post.total_lease_expense == pytest.approx(expected_lease)

    def test_empty_selection_no_change(self):
        """Empty selection means no change from baseline."""
        state = make_corporate_state()
        config = DEFAULT_ENGINE_CONFIG

        post, warnings = compute_post_transaction_metrics(state, [], config)

        assert len(warnings) == 0
        assert post.net_debt == state.net_debt
        assert post.interest_expense == state.interest_expense
        assert post.total_lease_expense == (state.lease_expense or 0)

    def test_over_repayment_warning(self):
        """Proceeds exceeding net_debt emit warning and cap repayment."""
        state = make_corporate_state(
            net_debt=5_000_000,  # Small debt
            interest_expense=300_000,
        )
        config = DEFAULT_ENGINE_CONFIG

        # Create large asset that generates more proceeds than debt
        asset = make_asset(noi=1_000_000, market_tier=MarketTier.TIER_1)
        selection = make_selection(asset, config)

        # Proceeds should be ~17.7M, much more than 5M debt
        assert selection.proceeds > state.net_debt

        post, warnings = compute_post_transaction_metrics(state, [selection], config)

        # Should have warning about surplus
        assert len(warnings) == 1
        assert "exceed" in warnings[0].lower() or "surplus" in warnings[0].lower()

        # Net debt should be zero (fully repaid)
        assert post.net_debt == 0

    def test_interest_clamping_warning(self):
        """Interest reduction exceeding expense emits warning and clamps to zero."""
        state = make_corporate_state(
            net_debt=100_000_000,
            interest_expense=1_000_000,  # Very low interest
        )
        config = DEFAULT_ENGINE_CONFIG

        # Create asset with large proceeds
        asset = make_asset(noi=5_000_000, market_tier=MarketTier.TIER_1)
        selection = make_selection(asset, config)

        # Interest reduction = proceeds * 0.06 should exceed 1M
        expected_reduction = selection.proceeds * 0.06
        assert expected_reduction > state.interest_expense

        post, warnings = compute_post_transaction_metrics(state, [selection], config)

        # Should have warning about interest clamping
        assert any("clamp" in w.lower() or "interest" in w.lower() for w in warnings)

        # Interest should be clamped to zero
        assert post.interest_expense == 0

    def test_zero_ebitda_returns_none_metrics(self):
        """Zero EBITDA results in None for coverage metrics."""
        state = make_corporate_state(ebitda=0)
        config = DEFAULT_ENGINE_CONFIG

        asset = make_asset()
        selection = make_selection(asset, config)

        post, _ = compute_post_transaction_metrics(state, [selection], config)

        assert post.leverage is None
        assert post.interest_coverage is None
        assert post.fixed_charge_coverage is None

    def test_multiple_assets(self):
        """Multiple assets aggregate correctly."""
        state = make_corporate_state()
        config = DEFAULT_ENGINE_CONFIG

        assets = [
            make_asset(asset_id="A001", noi=500_000),
            make_asset(asset_id="A002", noi=700_000),
            make_asset(asset_id="A003", noi=300_000),
        ]
        selections = [make_selection(a, config) for a in assets]

        total_proceeds = sum(s.proceeds for s in selections)
        total_rent = sum(s.slb_rent for s in selections)

        post, _ = compute_post_transaction_metrics(state, selections, config)

        expected_net_debt = state.net_debt - total_proceeds
        expected_lease = (state.lease_expense or 0) + total_rent

        assert post.net_debt == pytest.approx(expected_net_debt)
        assert post.total_lease_expense == pytest.approx(expected_lease)


# =============================================================================
# compute_critical_fraction Tests
# =============================================================================


class TestComputeCriticalFraction:
    """Test compute_critical_fraction function."""

    def test_no_critical_assets(self):
        """All assets below threshold results in zero fraction."""
        config = DEFAULT_ENGINE_CONFIG  # threshold = 0.7

        assets = [
            make_asset(asset_id="A001", criticality=0.3, noi=500_000),
            make_asset(asset_id="A002", criticality=0.5, noi=700_000),
            make_asset(asset_id="A003", criticality=0.6, noi=300_000),
        ]
        selections = [make_selection(a, config) for a in assets]

        fraction = compute_critical_fraction(selections, config)

        assert fraction == 0.0

    def test_all_critical_assets(self):
        """All assets above threshold results in 1.0 fraction."""
        config = DEFAULT_ENGINE_CONFIG  # threshold = 0.7

        assets = [
            make_asset(asset_id="A001", criticality=0.8, noi=500_000),
            make_asset(asset_id="A002", criticality=0.9, noi=700_000),
        ]
        selections = [make_selection(a, config) for a in assets]

        fraction = compute_critical_fraction(selections, config)

        assert fraction == pytest.approx(1.0)

    def test_mixed_criticality(self):
        """Mixed criticality computes correct fraction."""
        config = DEFAULT_ENGINE_CONFIG  # threshold = 0.7

        assets = [
            make_asset(asset_id="A001", criticality=0.3, noi=400_000),  # Not critical
            make_asset(asset_id="A002", criticality=0.8, noi=600_000),  # Critical
        ]
        selections = [make_selection(a, config) for a in assets]

        fraction = compute_critical_fraction(selections, config)

        # Critical NOI = 600K, Total NOI = 1M
        assert fraction == pytest.approx(0.6)

    def test_empty_selection(self):
        """Empty selection returns 0.0."""
        config = DEFAULT_ENGINE_CONFIG

        fraction = compute_critical_fraction([], config)

        assert fraction == 0.0

    def test_custom_threshold(self):
        """Custom criticality threshold works correctly."""
        config = EngineConfig(criticality_threshold=0.5)

        assets = [
            make_asset(asset_id="A001", criticality=0.4, noi=500_000),  # Not critical
            make_asset(asset_id="A002", criticality=0.6, noi=500_000),  # Critical
        ]
        selections = [make_selection(a, config) for a in assets]

        fraction = compute_critical_fraction(selections, config)

        assert fraction == pytest.approx(0.5)

    def test_boundary_criticality(self):
        """Asset exactly at threshold is NOT critical (> not >=)."""
        config = EngineConfig(criticality_threshold=0.7)

        assets = [
            make_asset(asset_id="A001", criticality=0.7, noi=500_000),  # Exactly at threshold
            make_asset(asset_id="A002", criticality=0.71, noi=500_000),  # Just above
        ]
        selections = [make_selection(a, config) for a in assets]

        fraction = compute_critical_fraction(selections, config)

        # Only A002 is critical
        assert fraction == pytest.approx(0.5)


# =============================================================================
# check_constraints Tests
# =============================================================================


class TestCheckConstraints:
    """Test check_constraints function."""

    def test_all_constraints_pass(self):
        """All constraints passing returns no violations."""
        state = make_corporate_state(
            net_debt=2_000_000_000,
            ebitda=500_000_000,
            interest_expense=100_000_000,
        )
        config = DEFAULT_ENGINE_CONFIG

        # Create selections that don't violate constraints
        assets = [make_asset(noi=10_000_000, criticality=0.3)]
        selections = [make_selection(a, config) for a in assets]
        total_proceeds = sum(s.proceeds for s in selections)

        constraints = HardConstraints(
            max_net_leverage=5.0,  # Will be well under
            min_fixed_charge_coverage=2.0,  # Will be well over
        )

        metrics, violations, warnings = check_constraints(
            selections, state, constraints, total_proceeds * 0.9, config  # Target below proceeds
        )

        assert len(violations) == 0
        assert metrics.total_proceeds == pytest.approx(total_proceeds)

    def test_leverage_violation(self):
        """Leverage exceeding limit creates violation."""
        state = make_corporate_state(
            net_debt=2_000_000_000,
            ebitda=400_000_000,  # leverage = 5.0
            interest_expense=50_000_000,
        )
        config = DEFAULT_ENGINE_CONFIG

        # Small proceeds won't help much
        assets = [make_asset(noi=100_000)]
        selections = [make_selection(a, config) for a in assets]

        constraints = HardConstraints(max_net_leverage=4.0)

        metrics, violations, _ = check_constraints(
            selections, state, constraints, 0, config
        )

        # Should have leverage violation
        leverage_violations = [v for v in violations if v.code == "MAX_NET_LEVERAGE"]
        assert len(leverage_violations) == 1
        assert leverage_violations[0].actual > 4.0

    def test_leverage_undefined_when_ebitda_zero(self):
        """Zero EBITDA creates leverage violation with NaN."""
        state = make_corporate_state(ebitda=0)
        config = DEFAULT_ENGINE_CONFIG

        selections = []
        constraints = HardConstraints(max_net_leverage=4.0)

        _, violations, _ = check_constraints(
            selections, state, constraints, 0, config
        )

        leverage_violations = [v for v in violations if v.code == "MAX_NET_LEVERAGE"]
        assert len(leverage_violations) == 1
        assert math.isnan(leverage_violations[0].actual)
        assert "undefined" in leverage_violations[0].detail.lower()

    def test_interest_coverage_violation(self):
        """Interest coverage below minimum creates violation."""
        state = make_corporate_state(
            ebitda=100_000_000,
            interest_expense=50_000_000,  # coverage = 2.0
        )
        config = DEFAULT_ENGINE_CONFIG

        selections = []
        constraints = HardConstraints(min_interest_coverage=3.0)

        _, violations, _ = check_constraints(
            selections, state, constraints, 0, config
        )

        coverage_violations = [v for v in violations if v.code == "MIN_INTEREST_COVERAGE"]
        assert len(coverage_violations) == 1
        assert coverage_violations[0].actual < 3.0

    def test_interest_coverage_none_not_violation(self):
        """Zero interest (None coverage) is NOT a violation (infinite coverage)."""
        state = make_corporate_state(
            ebitda=100_000_000,
            interest_expense=0,  # No interest
        )
        config = DEFAULT_ENGINE_CONFIG

        selections = []
        constraints = HardConstraints(min_interest_coverage=3.0)

        _, violations, _ = check_constraints(
            selections, state, constraints, 0, config
        )

        coverage_violations = [v for v in violations if v.code == "MIN_INTEREST_COVERAGE"]
        assert len(coverage_violations) == 0

    def test_fixed_charge_coverage_violation(self):
        """Fixed charge coverage below minimum creates violation."""
        state = make_corporate_state(
            ebitda=100_000_000,
            interest_expense=25_000_000,
            lease_expense=25_000_000,  # total = 50M, coverage = 2.0
        )
        config = DEFAULT_ENGINE_CONFIG

        selections = []
        constraints = HardConstraints(min_fixed_charge_coverage=3.0)

        _, violations, _ = check_constraints(
            selections, state, constraints, 0, config
        )

        coverage_violations = [v for v in violations if v.code == "MIN_FIXED_CHARGE_COVERAGE"]
        assert len(coverage_violations) == 1

    def test_critical_fraction_violation(self):
        """Critical fraction exceeding limit creates violation."""
        state = make_corporate_state()
        config = DEFAULT_ENGINE_CONFIG

        # All critical assets
        assets = [
            make_asset(asset_id="A001", criticality=0.9, noi=500_000),
            make_asset(asset_id="A002", criticality=0.8, noi=500_000),
        ]
        selections = [make_selection(a, config) for a in assets]

        constraints = HardConstraints(max_critical_fraction=0.5)

        _, violations, _ = check_constraints(
            selections, state, constraints, 0, config
        )

        crit_violations = [v for v in violations if v.code == "MAX_CRITICAL_FRACTION"]
        assert len(crit_violations) == 1
        assert crit_violations[0].actual == pytest.approx(1.0)

    def test_target_not_met_violation(self):
        """Proceeds below target creates violation."""
        state = make_corporate_state()
        config = DEFAULT_ENGINE_CONFIG

        assets = [make_asset(noi=100_000)]  # Small proceeds
        selections = [make_selection(a, config) for a in assets]
        total_proceeds = sum(s.proceeds for s in selections)

        constraints = HardConstraints()
        target = total_proceeds * 10  # Way above actual proceeds

        _, violations, _ = check_constraints(
            selections, state, constraints, target, config
        )

        target_violations = [v for v in violations if v.code == "TARGET_NOT_MET"]
        assert len(target_violations) == 1

    def test_target_within_tolerance_passes(self):
        """Proceeds within tolerance of target passes."""
        state = make_corporate_state()
        config = DEFAULT_ENGINE_CONFIG

        assets = [make_asset(noi=1_000_000)]
        selections = [make_selection(a, config) for a in assets]
        total_proceeds = sum(s.proceeds for s in selections)

        constraints = HardConstraints()
        # Target slightly above proceeds but within 2% tolerance
        target = total_proceeds * 1.01

        _, violations, _ = check_constraints(
            selections, state, constraints, target, config
        )

        target_violations = [v for v in violations if v.code == "TARGET_NOT_MET"]
        assert len(target_violations) == 0

    def test_multiple_violations(self):
        """Multiple constraint violations all reported."""
        state = make_corporate_state(
            net_debt=5_000_000_000,  # High leverage
            ebitda=500_000_000,
            interest_expense=200_000_000,  # Low coverage
        )
        config = DEFAULT_ENGINE_CONFIG

        # Critical assets
        assets = [make_asset(criticality=0.9, noi=100_000)]
        selections = [make_selection(a, config) for a in assets]

        constraints = HardConstraints(
            max_net_leverage=4.0,  # Will violate (leverage ~10)
            min_fixed_charge_coverage=5.0,  # Will violate
            max_critical_fraction=0.5,  # Will violate
        )

        _, violations, _ = check_constraints(
            selections, state, constraints, 1_000_000_000, config  # Huge target
        )

        # Should have multiple violations
        codes = {v.code for v in violations}
        assert "MAX_NET_LEVERAGE" in codes
        assert "MIN_FIXED_CHARGE_COVERAGE" in codes
        assert "MAX_CRITICAL_FRACTION" in codes
        assert "TARGET_NOT_MET" in codes

    def test_none_constraints_not_checked(self):
        """None constraints are not checked."""
        state = make_corporate_state(
            net_debt=10_000_000_000,  # Very high leverage
        )
        config = DEFAULT_ENGINE_CONFIG

        selections = []
        constraints = HardConstraints()  # All None

        _, violations, _ = check_constraints(
            selections, state, constraints, 0, config
        )

        # No violations since no constraints specified
        constraint_violations = [v for v in violations if v.code != "TARGET_NOT_MET"]
        assert len(constraint_violations) == 0

    def test_metrics_returned_correctly(self):
        """PortfolioMetrics is assembled correctly."""
        state = make_corporate_state(
            net_debt=2_000_000_000,
            ebitda=500_000_000,
            interest_expense=100_000_000,
            lease_expense=20_000_000,
        )
        config = DEFAULT_ENGINE_CONFIG

        assets = [make_asset(noi=500_000, criticality=0.8)]
        selections = [make_selection(a, config) for a in assets]

        constraints = HardConstraints()

        metrics, _, _ = check_constraints(
            selections, state, constraints, 0, config
        )

        # Check baseline is present
        assert metrics.baseline.leverage == pytest.approx(4.0)
        assert metrics.baseline.interest_coverage == pytest.approx(5.0)

        # Check post is present
        assert metrics.post.net_debt < state.net_debt  # Reduced by proceeds

        # Check totals
        assert metrics.total_proceeds == pytest.approx(sum(s.proceeds for s in selections))
        assert metrics.total_slb_rent == pytest.approx(sum(s.slb_rent for s in selections))
        assert metrics.critical_fraction == pytest.approx(1.0)  # All critical


# =============================================================================
# Idempotence Tests
# =============================================================================


class TestIdempotence:
    """Test that empty selection results in no change."""

    def test_empty_selection_preserves_baseline(self):
        """Empty selection should give post == baseline."""
        state = make_corporate_state(
            net_debt=2_000_000_000,
            ebitda=500_000_000,
            interest_expense=100_000_000,
            lease_expense=20_000_000,
        )
        config = DEFAULT_ENGINE_CONFIG

        baseline = compute_baseline_metrics(state, config)
        post, warnings = compute_post_transaction_metrics(state, [], config)

        assert len(warnings) == 0

        # Post should match baseline for absolute values
        assert post.net_debt == state.net_debt
        assert post.interest_expense == state.interest_expense
        assert post.total_lease_expense == (state.lease_expense or 0)

        # Coverage metrics should match
        assert post.leverage == pytest.approx(baseline.leverage)
        assert post.interest_coverage == pytest.approx(baseline.interest_coverage)
        assert post.fixed_charge_coverage == pytest.approx(baseline.fixed_charge_coverage)
