"""
Toy scenario tests for SLB Agent.

These tests validate the engine with hand-calculated expected values
before creating the full golden portfolio in PR12.

Scenarios:
1. OK - Two stores selected, target met, all constraints pass
2. INFEASIBLE - No eligible assets (filter excludes everything)
3. Revision - Target too high initially, 10% reduction makes it feasible
"""

import pytest

from app.config import EngineConfig
from app.engine.metrics import (
    compute_asset_slb_metrics,
    compute_baseline_metrics,
)
from app.engine.selector import apply_filters, compute_score, select_assets
from app.models import (
    Asset,
    AssetFilters,
    AssetType,
    ConstraintViolation,
    CorporateState,
    HardConstraints,
    MarketTier,
    Objective,
    ProgramType,
    SelectionStatus,
    SelectorSpec,
    SoftPreferences,
)
from app.revision_policy import enforce_revision_policy


# =============================================================================
# Test Configuration (matching user's spec)
# =============================================================================

TEST_CONFIG = EngineConfig(
    cap_rate_curve={
        AssetType.STORE: {
            MarketTier.TIER_1: 0.06,
            MarketTier.TIER_2: 0.065,
        },
        AssetType.OFFICE: {
            MarketTier.TIER_1: 0.055,
        },
        # Minimal entries for other types to avoid KeyError
        AssetType.DISTRIBUTION_CENTER: {MarketTier.TIER_1: 0.05, MarketTier.TIER_2: 0.06},
        AssetType.MIXED_USE: {MarketTier.TIER_1: 0.06, MarketTier.TIER_2: 0.07},
        AssetType.OTHER: {MarketTier.TIER_1: 0.07, MarketTier.TIER_2: 0.08},
    },
    transaction_haircut=0.025,
    slb_rent_multiplier=1.0,
    avg_cost_of_debt=0.06,
    epsilon=1e-6,
    criticality_threshold=0.7,
    target_tolerance=0.05,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def corporate_state() -> CorporateState:
    """Corporate state for all scenarios."""
    return CorporateState(
        net_debt=100_000_000,
        ebitda=40_000_000,
        interest_expense=6_000_000,
        lease_expense=0,
    )


@pytest.fixture
def asset_hq() -> Asset:
    """HQ Office - highly critical, low leaseability."""
    return Asset(
        asset_id="HQ",
        asset_type=AssetType.OFFICE,
        market="New York, NY",
        market_tier=MarketTier.TIER_1,
        noi=4_000_000,
        book_value=50_000_000,
        criticality=0.95,
        leaseability_score=0.30,
    )


@pytest.fixture
def asset_store_nyc() -> Asset:
    """Store NYC - Tier 1, good box."""
    return Asset(
        asset_id="STORE_NYC",
        asset_type=AssetType.STORE,
        market="New York, NY",
        market_tier=MarketTier.TIER_1,
        noi=2_000_000,
        book_value=25_000_000,
        criticality=0.40,
        leaseability_score=0.90,
    )


@pytest.fixture
def asset_store_austin() -> Asset:
    """Store Austin - Tier 2, still decent."""
    return Asset(
        asset_id="STORE_AUSTIN",
        asset_type=AssetType.STORE,
        market="Austin, TX",
        market_tier=MarketTier.TIER_2,
        noi=1_500_000,
        book_value=18_000_000,
        criticality=0.30,
        leaseability_score=0.80,
    )


@pytest.fixture
def all_assets(asset_hq, asset_store_nyc, asset_store_austin) -> list[Asset]:
    """All three assets."""
    return [asset_hq, asset_store_nyc, asset_store_austin]


@pytest.fixture
def scenario1_spec() -> SelectorSpec:
    """Scenario 1: Target 50M, filters allow only stores."""
    return SelectorSpec(
        program_type=ProgramType.SLB,
        objective=Objective.BALANCED,
        target_amount=50_000_000,
        hard_constraints=HardConstraints(
            max_net_leverage=4.0,
            min_interest_coverage=3.0,
            min_fixed_charge_coverage=2.5,
            max_critical_fraction=0.7,
        ),
        soft_preferences=SoftPreferences(
            prefer_low_criticality=True,
            prefer_high_leaseability=True,
            weight_criticality=1.0,
            weight_leaseability=1.0,
        ),
        asset_filters=AssetFilters(
            include_asset_types=[AssetType.STORE],
            max_criticality=0.8,
            min_leaseability_score=0.7,
        ),
        max_iterations=3,
    )


# =============================================================================
# Baseline Metrics Tests
# =============================================================================


class TestBaselineMetrics:
    """Verify baseline metrics calculation."""

    def test_baseline_leverage(self, corporate_state):
        """leverage = 100M / 40M = 2.50x"""
        baseline = compute_baseline_metrics(corporate_state, TEST_CONFIG)
        assert baseline.leverage == pytest.approx(2.50, rel=1e-3)

    def test_baseline_interest_coverage(self, corporate_state):
        """interest_coverage = 40M / 6M ≈ 6.67x"""
        baseline = compute_baseline_metrics(corporate_state, TEST_CONFIG)
        assert baseline.interest_coverage == pytest.approx(6.667, rel=1e-2)

    def test_baseline_fixed_charge_coverage(self, corporate_state):
        """fixed_charge_coverage = 40M / (6M + 0) ≈ 6.67x"""
        baseline = compute_baseline_metrics(corporate_state, TEST_CONFIG)
        assert baseline.fixed_charge_coverage == pytest.approx(6.667, rel=1e-2)


# =============================================================================
# Asset SLB Metrics Tests
# =============================================================================


class TestAssetSLBMetrics:
    """Verify per-asset SLB calculations."""

    def test_hq_metrics(self, asset_hq):
        """HQ Office: cap=5.5%, mv=72.73M, proceeds=70.91M, rent=4M"""
        metrics = compute_asset_slb_metrics(asset_hq, TEST_CONFIG)

        assert metrics.cap_rate == pytest.approx(0.055, rel=1e-3)
        assert metrics.market_value == pytest.approx(72_727_272.73, rel=1e-3)
        assert metrics.proceeds == pytest.approx(70_909_090.91, rel=1e-3)
        assert metrics.slb_rent == pytest.approx(4_000_000, rel=1e-3)

    def test_store_nyc_metrics(self, asset_store_nyc):
        """Store NYC: cap=6.0%, mv=33.33M, proceeds=32.50M, rent=2M"""
        metrics = compute_asset_slb_metrics(asset_store_nyc, TEST_CONFIG)

        assert metrics.cap_rate == pytest.approx(0.06, rel=1e-3)
        assert metrics.market_value == pytest.approx(33_333_333.33, rel=1e-3)
        assert metrics.proceeds == pytest.approx(32_500_000, rel=1e-3)
        assert metrics.slb_rent == pytest.approx(2_000_000, rel=1e-3)

    def test_store_austin_metrics(self, asset_store_austin):
        """Store Austin: cap=6.5%, mv=23.08M, proceeds=22.50M, rent=1.5M"""
        metrics = compute_asset_slb_metrics(asset_store_austin, TEST_CONFIG)

        assert metrics.cap_rate == pytest.approx(0.065, rel=1e-3)
        assert metrics.market_value == pytest.approx(23_076_923.08, rel=1e-3)
        assert metrics.proceeds == pytest.approx(22_500_000, rel=1e-3)
        assert metrics.slb_rent == pytest.approx(1_500_000, rel=1e-3)

    def test_combined_stores_proceeds(self, asset_store_nyc, asset_store_austin):
        """Combined stores: 32.50M + 22.50M = 55.00M"""
        nyc = compute_asset_slb_metrics(asset_store_nyc, TEST_CONFIG)
        austin = compute_asset_slb_metrics(asset_store_austin, TEST_CONFIG)

        total_proceeds = nyc.proceeds + austin.proceeds
        total_rent = nyc.slb_rent + austin.slb_rent

        assert total_proceeds == pytest.approx(55_000_000, rel=1e-3)
        assert total_rent == pytest.approx(3_500_000, rel=1e-3)


# =============================================================================
# Scoring and Filtering Tests
# =============================================================================


class TestScoringAndFiltering:
    """Verify scoring and filter behavior."""

    def test_scores(self, asset_hq, asset_store_nyc, asset_store_austin):
        """Scores: HQ=-0.65, NYC=0.50, Austin=0.50"""
        prefs = SoftPreferences(
            prefer_low_criticality=True,
            prefer_high_leaseability=True,
            weight_criticality=1.0,
            weight_leaseability=1.0,
        )

        # score = -criticality + leaseability
        assert compute_score(asset_hq, prefs) == pytest.approx(-0.65, rel=1e-3)
        assert compute_score(asset_store_nyc, prefs) == pytest.approx(0.50, rel=1e-3)
        assert compute_score(asset_store_austin, prefs) == pytest.approx(0.50, rel=1e-3)

    def test_store_filter(self, all_assets, scenario1_spec):
        """Filter with include_asset_types=[STORE] excludes HQ."""
        eligible = apply_filters(all_assets, scenario1_spec.asset_filters)

        assert len(eligible) == 2
        asset_ids = [a.asset_id for a in eligible]
        assert "HQ" not in asset_ids
        assert "STORE_NYC" in asset_ids
        assert "STORE_AUSTIN" in asset_ids

    def test_mixed_use_filter_excludes_all(self, all_assets):
        """Filter with include_asset_types=[MIXED_USE] returns empty."""
        filters = AssetFilters(include_asset_types=[AssetType.MIXED_USE])
        eligible = apply_filters(all_assets, filters)
        assert len(eligible) == 0


# =============================================================================
# Scenario 1: OK - Two stores selected
# =============================================================================


class TestScenario1OK:
    """Scenario 1: Happy path - both stores selected, target met."""

    def test_selection_status_ok(self, all_assets, corporate_state, scenario1_spec):
        """Selection should return OK status."""
        outcome = select_assets(all_assets, corporate_state, scenario1_spec, TEST_CONFIG)
        assert outcome.status == SelectionStatus.OK

    def test_both_stores_selected(self, all_assets, corporate_state, scenario1_spec):
        """Both stores should be selected."""
        outcome = select_assets(all_assets, corporate_state, scenario1_spec, TEST_CONFIG)

        assert len(outcome.selected_assets) == 2
        selected_ids = [s.asset.asset_id for s in outcome.selected_assets]
        assert "STORE_NYC" in selected_ids
        assert "STORE_AUSTIN" in selected_ids
        assert "HQ" not in selected_ids

    def test_proceeds(self, all_assets, corporate_state, scenario1_spec):
        """Total proceeds should be 55M."""
        outcome = select_assets(all_assets, corporate_state, scenario1_spec, TEST_CONFIG)
        assert outcome.proceeds == pytest.approx(55_000_000, rel=1e-3)

    def test_leverage_before_after(self, all_assets, corporate_state, scenario1_spec):
        """Leverage: 2.50x -> 1.125x"""
        outcome = select_assets(all_assets, corporate_state, scenario1_spec, TEST_CONFIG)

        assert outcome.leverage_before == pytest.approx(2.50, rel=1e-3)
        assert outcome.leverage_after == pytest.approx(1.125, rel=1e-3)

    def test_interest_coverage_before_after(self, all_assets, corporate_state, scenario1_spec):
        """Interest coverage: 6.67x -> 14.81x"""
        outcome = select_assets(all_assets, corporate_state, scenario1_spec, TEST_CONFIG)

        assert outcome.interest_coverage_before == pytest.approx(6.667, rel=1e-2)
        assert outcome.interest_coverage_after == pytest.approx(14.815, rel=1e-2)

    def test_fixed_charge_coverage_before_after(self, all_assets, corporate_state, scenario1_spec):
        """Fixed charge coverage: 6.67x -> 6.45x"""
        outcome = select_assets(all_assets, corporate_state, scenario1_spec, TEST_CONFIG)

        assert outcome.fixed_charge_coverage_before == pytest.approx(6.667, rel=1e-2)
        assert outcome.fixed_charge_coverage_after == pytest.approx(6.452, rel=1e-2)

    def test_critical_fraction_zero(self, all_assets, corporate_state, scenario1_spec):
        """Critical fraction should be 0.0 (both stores have criticality < 0.7)."""
        outcome = select_assets(all_assets, corporate_state, scenario1_spec, TEST_CONFIG)
        assert outcome.critical_fraction == pytest.approx(0.0, abs=1e-6)

    def test_no_violations(self, all_assets, corporate_state, scenario1_spec):
        """No constraint violations."""
        outcome = select_assets(all_assets, corporate_state, scenario1_spec, TEST_CONFIG)
        assert outcome.violations == []

    def test_no_warnings(self, all_assets, corporate_state, scenario1_spec):
        """No warnings."""
        outcome = select_assets(all_assets, corporate_state, scenario1_spec, TEST_CONFIG)
        assert outcome.warnings == []


# =============================================================================
# Scenario 2: INFEASIBLE - No eligible assets
# =============================================================================


class TestScenario2NoEligibleAssets:
    """Scenario 2: No assets match filter -> INFEASIBLE."""

    @pytest.fixture
    def scenario2_spec(self) -> SelectorSpec:
        """Spec with MIXED_USE filter (no assets match)."""
        return SelectorSpec(
            program_type=ProgramType.SLB,
            objective=Objective.BALANCED,
            target_amount=20_000_000,
            hard_constraints=HardConstraints(
                max_net_leverage=4.0,
                min_interest_coverage=3.0,
                min_fixed_charge_coverage=2.5,
                max_critical_fraction=0.7,
            ),
            soft_preferences=SoftPreferences(),
            asset_filters=AssetFilters(
                include_asset_types=[AssetType.MIXED_USE],
            ),
            max_iterations=3,
        )

    def test_status_infeasible(self, all_assets, corporate_state, scenario2_spec):
        """Status should be INFEASIBLE."""
        outcome = select_assets(all_assets, corporate_state, scenario2_spec, TEST_CONFIG)
        assert outcome.status == SelectionStatus.INFEASIBLE

    def test_no_assets_selected(self, all_assets, corporate_state, scenario2_spec):
        """No assets selected."""
        outcome = select_assets(all_assets, corporate_state, scenario2_spec, TEST_CONFIG)
        assert outcome.selected_assets == []

    def test_proceeds_zero(self, all_assets, corporate_state, scenario2_spec):
        """Proceeds should be 0."""
        outcome = select_assets(all_assets, corporate_state, scenario2_spec, TEST_CONFIG)
        assert outcome.proceeds == 0

    def test_metrics_passthrough(self, all_assets, corporate_state, scenario2_spec):
        """Before/after metrics should be identical (baseline passthrough)."""
        outcome = select_assets(all_assets, corporate_state, scenario2_spec, TEST_CONFIG)

        # Leverage
        assert outcome.leverage_before == pytest.approx(2.50, rel=1e-3)
        assert outcome.leverage_after == pytest.approx(2.50, rel=1e-3)

        # Interest coverage
        assert outcome.interest_coverage_before == pytest.approx(6.667, rel=1e-2)
        assert outcome.interest_coverage_after == pytest.approx(6.667, rel=1e-2)

        # Fixed charge coverage
        assert outcome.fixed_charge_coverage_before == pytest.approx(6.667, rel=1e-2)
        assert outcome.fixed_charge_coverage_after == pytest.approx(6.667, rel=1e-2)

    def test_critical_fraction_zero(self, all_assets, corporate_state, scenario2_spec):
        """Critical fraction should be 0."""
        outcome = select_assets(all_assets, corporate_state, scenario2_spec, TEST_CONFIG)
        assert outcome.critical_fraction == 0.0

    def test_no_eligible_assets_violation(self, all_assets, corporate_state, scenario2_spec):
        """Should have NO_ELIGIBLE_ASSETS violation."""
        outcome = select_assets(all_assets, corporate_state, scenario2_spec, TEST_CONFIG)

        assert len(outcome.violations) == 1
        assert outcome.violations[0].code == "NO_ELIGIBLE_ASSETS"


# =============================================================================
# Scenario 3: Revision - Target too high, then 10% lower works
# =============================================================================


class TestScenario3Revision:
    """Scenario 3: Target 60M fails, revision to 54M succeeds."""

    @pytest.fixture
    def scenario3a_spec(self) -> SelectorSpec:
        """Initial spec with target=60M (too high)."""
        return SelectorSpec(
            program_type=ProgramType.SLB,
            objective=Objective.BALANCED,
            target_amount=60_000_000,
            hard_constraints=HardConstraints(
                max_net_leverage=4.0,
                min_interest_coverage=3.0,
                min_fixed_charge_coverage=2.5,
                max_critical_fraction=0.7,
            ),
            soft_preferences=SoftPreferences(
                prefer_low_criticality=True,
                prefer_high_leaseability=True,
                weight_criticality=1.0,
                weight_leaseability=1.0,
            ),
            asset_filters=AssetFilters(
                include_asset_types=[AssetType.STORE],
                max_criticality=0.8,
                min_leaseability_score=0.7,
            ),
            max_iterations=3,
        )

    @pytest.fixture
    def scenario3b_spec(self, scenario3a_spec) -> SelectorSpec:
        """Revised spec with target=54M (10% reduction)."""
        return scenario3a_spec.model_copy(update={"target_amount": 54_000_000})

    # --- 3a: Initial spec fails ---

    def test_3a_status_infeasible(self, all_assets, corporate_state, scenario3a_spec):
        """Initial spec should be INFEASIBLE."""
        outcome = select_assets(all_assets, corporate_state, scenario3a_spec, TEST_CONFIG)
        assert outcome.status == SelectionStatus.INFEASIBLE

    def test_3a_both_stores_selected(self, all_assets, corporate_state, scenario3a_spec):
        """Both stores still selected (greedy takes them)."""
        outcome = select_assets(all_assets, corporate_state, scenario3a_spec, TEST_CONFIG)

        assert len(outcome.selected_assets) == 2
        selected_ids = [s.asset.asset_id for s in outcome.selected_assets]
        assert "STORE_NYC" in selected_ids
        assert "STORE_AUSTIN" in selected_ids

    def test_3a_proceeds(self, all_assets, corporate_state, scenario3a_spec):
        """Proceeds should be 55M."""
        outcome = select_assets(all_assets, corporate_state, scenario3a_spec, TEST_CONFIG)
        assert outcome.proceeds == pytest.approx(55_000_000, rel=1e-3)

    def test_3a_target_not_met_violation(self, all_assets, corporate_state, scenario3a_spec):
        """Should have TARGET_NOT_MET violation."""
        outcome = select_assets(all_assets, corporate_state, scenario3a_spec, TEST_CONFIG)

        assert len(outcome.violations) == 1
        assert outcome.violations[0].code == "TARGET_NOT_MET"

        # Target threshold = 60M * 0.95 = 57M
        assert outcome.violations[0].limit == pytest.approx(57_000_000, rel=1e-3)
        assert outcome.violations[0].actual == pytest.approx(55_000_000, rel=1e-3)

    def test_3a_metrics_same_as_scenario1(self, all_assets, corporate_state, scenario3a_spec):
        """Metrics should match scenario 1 (same assets selected)."""
        outcome = select_assets(all_assets, corporate_state, scenario3a_spec, TEST_CONFIG)

        assert outcome.leverage_before == pytest.approx(2.50, rel=1e-3)
        assert outcome.leverage_after == pytest.approx(1.125, rel=1e-3)
        assert outcome.interest_coverage_after == pytest.approx(14.815, rel=1e-2)
        assert outcome.fixed_charge_coverage_after == pytest.approx(6.452, rel=1e-2)
        assert outcome.critical_fraction == pytest.approx(0.0, abs=1e-6)

    # --- Revision policy check ---

    def test_revision_policy_accepts_10_percent_drop(self, scenario3a_spec, scenario3b_spec):
        """Policy should accept 10% target reduction."""
        immutable_hard = scenario3a_spec.hard_constraints.model_copy(deep=True)
        original_target = scenario3a_spec.target_amount

        result = enforce_revision_policy(
            immutable_hard=immutable_hard,
            original_target=original_target,
            prev_spec=scenario3a_spec,
            new_spec=scenario3b_spec,
        )

        assert result.valid is True
        assert result.spec is not None
        assert result.spec.target_amount == 54_000_000

    def test_revision_policy_bounds(self, scenario3a_spec):
        """Verify policy bounds: 20% per-iter max, 50% global floor."""
        immutable_hard = scenario3a_spec.hard_constraints.model_copy(deep=True)
        original_target = 60_000_000

        # 20% drop per iteration: 60M * 0.80 = 48M minimum
        # 50% global floor: 60M * 0.50 = 30M minimum

        # Attempt 25% drop (should be clamped to 20%)
        over_drop_spec = scenario3a_spec.model_copy(update={"target_amount": 45_000_000})
        result = enforce_revision_policy(
            immutable_hard=immutable_hard,
            original_target=original_target,
            prev_spec=scenario3a_spec,
            new_spec=over_drop_spec,
        )

        assert result.valid is True
        assert result.spec.target_amount == pytest.approx(48_000_000, rel=1e-3)  # Clamped to 20%

    # --- 3b: Revised spec succeeds ---

    def test_3b_status_ok(self, all_assets, corporate_state, scenario3b_spec):
        """Revised spec should return OK."""
        outcome = select_assets(all_assets, corporate_state, scenario3b_spec, TEST_CONFIG)
        assert outcome.status == SelectionStatus.OK

    def test_3b_target_met(self, all_assets, corporate_state, scenario3b_spec):
        """Target should now be met (55M >= 54M * 0.95 = 51.3M)."""
        outcome = select_assets(all_assets, corporate_state, scenario3b_spec, TEST_CONFIG)

        # No violations
        assert outcome.violations == []

        # Proceeds >= target threshold
        target_threshold = 54_000_000 * 0.95  # 51.3M
        assert outcome.proceeds >= target_threshold

    def test_3b_same_selection_as_scenario1(self, all_assets, corporate_state, scenario3b_spec):
        """Same assets selected, same metrics."""
        outcome = select_assets(all_assets, corporate_state, scenario3b_spec, TEST_CONFIG)

        assert len(outcome.selected_assets) == 2
        assert outcome.proceeds == pytest.approx(55_000_000, rel=1e-3)
        assert outcome.leverage_after == pytest.approx(1.125, rel=1e-3)
        assert outcome.fixed_charge_coverage_after == pytest.approx(6.452, rel=1e-2)


# =============================================================================
# Full Revision Loop Integration Test
# =============================================================================


class TestRevisionLoopIntegration:
    """Test the full revision loop pattern (without orchestrator)."""

    def test_revision_loop_succeeds_after_one_iteration(
        self, all_assets, corporate_state
    ):
        """Simulate: initial spec fails -> revise -> succeeds."""
        # Initial spec with target=60M
        initial_spec = SelectorSpec(
            program_type=ProgramType.SLB,
            objective=Objective.BALANCED,
            target_amount=60_000_000,
            hard_constraints=HardConstraints(
                max_net_leverage=4.0,
                min_interest_coverage=3.0,
                min_fixed_charge_coverage=2.5,
                max_critical_fraction=0.7,
            ),
            soft_preferences=SoftPreferences(),
            asset_filters=AssetFilters(
                include_asset_types=[AssetType.STORE],
                max_criticality=0.8,
                min_leaseability_score=0.7,
            ),
            max_iterations=3,
        )

        # Capture immutable constraints
        immutable_hard = initial_spec.hard_constraints.model_copy(deep=True)
        original_target = initial_spec.target_amount

        # First run: should be INFEASIBLE
        outcome1 = select_assets(all_assets, corporate_state, initial_spec, TEST_CONFIG)
        assert outcome1.status == SelectionStatus.INFEASIBLE
        assert any(v.code == "TARGET_NOT_MET" for v in outcome1.violations)

        # Simulate LLM revision: drop target by 10%
        revised_spec = initial_spec.model_copy(update={"target_amount": 54_000_000})

        # Policy check
        policy_result = enforce_revision_policy(
            immutable_hard=immutable_hard,
            original_target=original_target,
            prev_spec=initial_spec,
            new_spec=revised_spec,
        )
        assert policy_result.valid is True

        # Second run: should be OK
        outcome2 = select_assets(
            all_assets, corporate_state, policy_result.spec, TEST_CONFIG
        )
        assert outcome2.status == SelectionStatus.OK
        assert outcome2.violations == []
        assert outcome2.proceeds == pytest.approx(55_000_000, rel=1e-3)
