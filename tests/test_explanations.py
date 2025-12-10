"""Tests for Explanation Generation (PR6).

Tests cover:
- Constraint nodes: binding and violated constraints
- Driver nodes: selection factors and preferences
- Risk nodes: concentration and degradation risks
- Node uniqueness and structure
"""

import pytest

from app.config import DEFAULT_ENGINE_CONFIG
from app.engine.explanations import generate_explanation_nodes
from app.models import (
    Asset,
    AssetFilters,
    AssetSelection,
    AssetType,
    ConstraintViolation,
    CorporateState,
    HardConstraints,
    MarketTier,
    Objective,
    ProgramOutcome,
    ProgramType,
    SelectionStatus,
    SelectorSpec,
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
    target_amount: float = 100_000_000,
    max_net_leverage: float = 4.0,
    min_fixed_charge_coverage: float = 3.0,
    min_interest_coverage: float = None,
    max_critical_fraction: float = None,
    prefer_low_criticality: bool = True,
    prefer_high_leaseability: bool = True,
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
        ),
        asset_filters=AssetFilters(),
    )


def make_selection(
    asset_id: str = "A001",
    proceeds: float = 10_000_000,
    slb_rent: float = 700_000,
    criticality: float = 0.3,
    leaseability_score: float = 0.8,
    asset_type: AssetType = AssetType.STORE,
    market: str = "Dallas, TX",
) -> AssetSelection:
    """Create an AssetSelection for testing."""
    asset = make_asset(
        asset_id=asset_id,
        criticality=criticality,
        leaseability_score=leaseability_score,
        asset_type=asset_type,
        market=market,
    )
    return AssetSelection(asset=asset, proceeds=proceeds, slb_rent=slb_rent)


def make_ok_outcome(
    selected_assets: list = None,
    proceeds: float = 100_000_000,
    leverage_before: float = 4.0,
    leverage_after: float = 3.5,
    interest_coverage_before: float = 5.0,
    interest_coverage_after: float = 5.5,
    fixed_charge_coverage_before: float = 3.5,
    fixed_charge_coverage_after: float = 3.0,
    critical_fraction: float = 0.1,
) -> ProgramOutcome:
    """Create an OK ProgramOutcome for testing."""
    if selected_assets is None:
        selected_assets = [make_selection()]
    return ProgramOutcome(
        status=SelectionStatus.OK,
        selected_assets=selected_assets,
        proceeds=proceeds,
        leverage_before=leverage_before,
        leverage_after=leverage_after,
        interest_coverage_before=interest_coverage_before,
        interest_coverage_after=interest_coverage_after,
        fixed_charge_coverage_before=fixed_charge_coverage_before,
        fixed_charge_coverage_after=fixed_charge_coverage_after,
        critical_fraction=critical_fraction,
    )


def make_infeasible_outcome(
    violations: list = None,
) -> ProgramOutcome:
    """Create an INFEASIBLE ProgramOutcome for testing."""
    if violations is None:
        violations = [ConstraintViolation(
            code="TARGET_NOT_MET",
            detail="Proceeds below target",
            actual=50_000_000,
            limit=100_000_000,
        )]
    return ProgramOutcome(
        status=SelectionStatus.INFEASIBLE,
        selected_assets=[],
        proceeds=0,
        leverage_before=4.0,
        leverage_after=4.0,
        violations=violations,
    )


# =============================================================================
# Constraint Node Tests
# =============================================================================


class TestConstraintNodes:
    """Tests for constraint node generation."""

    def test_violation_generates_error_node(self):
        """Violated constraints generate error severity nodes."""
        violation = ConstraintViolation(
            code="MAX_NET_LEVERAGE",
            detail="Post-SLB leverage 4.5x exceeds limit 4.0x",
            actual=4.5,
            limit=4.0,
        )
        outcome = make_infeasible_outcome(violations=[violation])
        spec = make_spec()
        state = make_corporate_state()

        nodes = generate_explanation_nodes(spec, outcome, state, DEFAULT_ENGINE_CONFIG)

        constraint_nodes = [n for n in nodes if n.category == "constraint"]
        assert len(constraint_nodes) >= 1

        leverage_node = next(n for n in constraint_nodes if "leverage" in n.id.lower())
        assert leverage_node.severity == "error"
        assert leverage_node.post_value == 4.5
        assert leverage_node.threshold == 4.0

    def test_target_not_met_generates_node(self):
        """TARGET_NOT_MET violation generates constraint node."""
        outcome = make_infeasible_outcome()
        spec = make_spec(target_amount=100_000_000)
        state = make_corporate_state()

        nodes = generate_explanation_nodes(spec, outcome, state, DEFAULT_ENGINE_CONFIG)

        target_nodes = [n for n in nodes if "target" in n.id.lower()]
        assert len(target_nodes) >= 1
        assert target_nodes[0].severity == "error"

    def test_no_eligible_assets_generates_node(self):
        """NO_ELIGIBLE_ASSETS violation generates constraint node."""
        violation = ConstraintViolation(
            code="NO_ELIGIBLE_ASSETS",
            detail="No assets pass the filter criteria",
            actual=0,
            limit=1,
        )
        outcome = make_infeasible_outcome(violations=[violation])
        spec = make_spec()
        state = make_corporate_state()

        nodes = generate_explanation_nodes(spec, outcome, state, DEFAULT_ENGINE_CONFIG)

        eligible_nodes = [n for n in nodes if "eligible" in n.id.lower()]
        assert len(eligible_nodes) >= 1

    def test_near_binding_leverage_generates_warning(self):
        """Leverage near limit generates warning node."""
        outcome = make_ok_outcome(
            leverage_before=4.0,
            leverage_after=3.8,  # 95% of 4.0 limit
        )
        spec = make_spec(max_net_leverage=4.0)
        state = make_corporate_state()

        nodes = generate_explanation_nodes(spec, outcome, state, DEFAULT_ENGINE_CONFIG)

        near_binding = [n for n in nodes if "near_binding" in n.id or "near" in n.label.lower()]
        leverage_near = [n for n in near_binding if n.metric == "leverage"]
        assert len(leverage_near) >= 1
        assert leverage_near[0].severity == "warning"

    def test_near_binding_fcc_generates_warning(self):
        """Fixed charge coverage near minimum generates warning node."""
        outcome = make_ok_outcome(
            fixed_charge_coverage_before=3.5,
            fixed_charge_coverage_after=3.2,  # Within 10% above 3.0 minimum
        )
        spec = make_spec(min_fixed_charge_coverage=3.0)
        state = make_corporate_state()

        nodes = generate_explanation_nodes(spec, outcome, state, DEFAULT_ENGINE_CONFIG)

        fcc_nodes = [n for n in nodes if n.metric == "fixed_charge_coverage" and n.category == "constraint"]
        near_binding = [n for n in fcc_nodes if "warning" == n.severity]
        assert len(near_binding) >= 1

    def test_no_near_binding_when_comfortable(self):
        """No near-binding nodes when metrics are comfortable."""
        outcome = make_ok_outcome(
            leverage_before=4.0,
            leverage_after=2.0,  # 50% of 4.0 limit - very comfortable
            fixed_charge_coverage_before=3.5,
            fixed_charge_coverage_after=5.0,  # Well above 3.0 minimum
        )
        spec = make_spec(max_net_leverage=4.0, min_fixed_charge_coverage=3.0)
        state = make_corporate_state()

        nodes = generate_explanation_nodes(spec, outcome, state, DEFAULT_ENGINE_CONFIG)

        near_binding = [n for n in nodes if "near_binding" in n.id]
        assert len(near_binding) == 0


# =============================================================================
# Driver Node Tests
# =============================================================================


class TestDriverNodes:
    """Tests for driver node generation."""

    def test_low_criticality_driver_generated(self):
        """Low criticality preference generates driver node."""
        selections = [
            make_selection(asset_id="A001", criticality=0.2),
            make_selection(asset_id="A002", criticality=0.3),
        ]
        outcome = make_ok_outcome(selected_assets=selections)
        spec = make_spec(prefer_low_criticality=True)
        state = make_corporate_state()

        nodes = generate_explanation_nodes(spec, outcome, state, DEFAULT_ENGINE_CONFIG)

        driver_nodes = [n for n in nodes if n.category == "driver"]
        crit_drivers = [n for n in driver_nodes if "criticality" in n.id.lower()]
        assert len(crit_drivers) >= 1
        assert crit_drivers[0].severity == "info"

    def test_high_leaseability_driver_generated(self):
        """High leaseability preference generates driver node."""
        selections = [
            make_selection(asset_id="A001", leaseability_score=0.9),
            make_selection(asset_id="A002", leaseability_score=0.8),
        ]
        outcome = make_ok_outcome(selected_assets=selections)
        spec = make_spec(prefer_high_leaseability=True)
        state = make_corporate_state()

        nodes = generate_explanation_nodes(spec, outcome, state, DEFAULT_ENGINE_CONFIG)

        driver_nodes = [n for n in nodes if n.category == "driver"]
        lease_drivers = [n for n in driver_nodes if "leaseability" in n.id.lower()]
        assert len(lease_drivers) >= 1

    def test_target_achieved_driver_generated(self):
        """Target achievement generates driver node."""
        outcome = make_ok_outcome(proceeds=110_000_000)
        spec = make_spec(target_amount=100_000_000)
        state = make_corporate_state()

        nodes = generate_explanation_nodes(spec, outcome, state, DEFAULT_ENGINE_CONFIG)

        target_drivers = [n for n in nodes if "target" in n.id.lower() and n.category == "driver"]
        assert len(target_drivers) >= 1
        assert target_drivers[0].post_value == 110_000_000

    def test_leverage_reduction_driver_generated(self):
        """Leverage reduction generates driver node."""
        outcome = make_ok_outcome(
            leverage_before=4.5,
            leverage_after=3.0,
        )
        spec = make_spec()
        state = make_corporate_state()

        nodes = generate_explanation_nodes(spec, outcome, state, DEFAULT_ENGINE_CONFIG)

        leverage_drivers = [n for n in nodes if "leverage" in n.id.lower() and n.category == "driver"]
        assert len(leverage_drivers) >= 1
        assert leverage_drivers[0].baseline_value == 4.5
        assert leverage_drivers[0].post_value == 3.0

    def test_no_drivers_for_infeasible(self):
        """No driver nodes generated for infeasible outcomes."""
        outcome = make_infeasible_outcome()
        spec = make_spec()
        state = make_corporate_state()

        nodes = generate_explanation_nodes(spec, outcome, state, DEFAULT_ENGINE_CONFIG)

        driver_nodes = [n for n in nodes if n.category == "driver"]
        assert len(driver_nodes) == 0


# =============================================================================
# Risk Node Tests
# =============================================================================


class TestRiskNodes:
    """Tests for risk node generation."""

    def test_critical_concentration_generates_risk(self):
        """Critical asset concentration generates risk node."""
        selections = [
            make_selection(asset_id="A001", criticality=0.9),  # Critical
        ]
        outcome = make_ok_outcome(
            selected_assets=selections,
            critical_fraction=0.4,
        )
        spec = make_spec(max_critical_fraction=0.5)
        state = make_corporate_state()

        nodes = generate_explanation_nodes(spec, outcome, state, DEFAULT_ENGINE_CONFIG)

        risk_nodes = [n for n in nodes if n.category == "risk"]
        critical_risks = [n for n in risk_nodes if "critical" in n.id.lower()]
        assert len(critical_risks) >= 1
        assert critical_risks[0].post_value == 0.4

    def test_market_concentration_generates_risk(self):
        """High market concentration generates risk node."""
        selections = [
            make_selection(asset_id="A001", market="Dallas, TX", proceeds=80_000_000),
            make_selection(asset_id="A002", market="Austin, TX", proceeds=10_000_000),
        ]
        outcome = make_ok_outcome(
            selected_assets=selections,
            proceeds=90_000_000,
        )
        spec = make_spec()
        state = make_corporate_state()

        nodes = generate_explanation_nodes(spec, outcome, state, DEFAULT_ENGINE_CONFIG)

        market_risks = [n for n in nodes if "market" in n.id.lower() and n.category == "risk"]
        # 80M / 90M ≈ 89% in Dallas, which is > 50%
        assert len(market_risks) >= 1

    def test_type_concentration_generates_risk(self):
        """High asset type concentration generates risk node."""
        selections = [
            make_selection(asset_id="A001", asset_type=AssetType.STORE, proceeds=90_000_000),
            make_selection(asset_id="A002", asset_type=AssetType.OFFICE, proceeds=10_000_000),
        ]
        outcome = make_ok_outcome(
            selected_assets=selections,
            proceeds=100_000_000,
        )
        spec = make_spec()
        state = make_corporate_state()

        nodes = generate_explanation_nodes(spec, outcome, state, DEFAULT_ENGINE_CONFIG)

        type_risks = [n for n in nodes if "type" in n.id.lower() and n.category == "risk"]
        # 90M / 100M = 90% in stores, which is > 70%
        assert len(type_risks) >= 1

    def test_single_asset_generates_risk(self):
        """Single asset selection generates risk node."""
        selections = [make_selection(asset_id="A001")]
        outcome = make_ok_outcome(selected_assets=selections)
        spec = make_spec()
        state = make_corporate_state()

        nodes = generate_explanation_nodes(spec, outcome, state, DEFAULT_ENGINE_CONFIG)

        single_risks = [n for n in nodes if "single" in n.id.lower()]
        assert len(single_risks) >= 1
        assert single_risks[0].severity == "warning"
        assert "A001" in single_risks[0].asset_ids

    def test_fcc_degradation_generates_risk(self):
        """Significant FCC degradation generates risk node."""
        outcome = make_ok_outcome(
            fixed_charge_coverage_before=4.0,
            fixed_charge_coverage_after=3.0,  # 25% degradation
        )
        spec = make_spec()
        state = make_corporate_state()

        nodes = generate_explanation_nodes(spec, outcome, state, DEFAULT_ENGINE_CONFIG)

        fcc_risks = [n for n in nodes if "fcc" in n.id.lower() or "fixed_charge" in str(n.metric)]
        degradation_risks = [n for n in fcc_risks if n.category == "risk"]
        assert len(degradation_risks) >= 1

    def test_no_risks_for_empty_selection(self):
        """No risk nodes for empty selection."""
        outcome = make_infeasible_outcome()
        spec = make_spec()
        state = make_corporate_state()

        nodes = generate_explanation_nodes(spec, outcome, state, DEFAULT_ENGINE_CONFIG)

        risk_nodes = [n for n in nodes if n.category == "risk"]
        assert len(risk_nodes) == 0


# =============================================================================
# Node Structure Tests
# =============================================================================


class TestNodeStructure:
    """Tests for node structure and uniqueness."""

    def test_all_nodes_have_required_fields(self):
        """All nodes have required id, label, severity, category."""
        selections = [
            make_selection(asset_id="A001", criticality=0.9),
            make_selection(asset_id="A002", criticality=0.2, market="Chicago, IL"),
        ]
        outcome = make_ok_outcome(
            selected_assets=selections,
            critical_fraction=0.3,
        )
        spec = make_spec(max_critical_fraction=0.5)
        state = make_corporate_state()

        nodes = generate_explanation_nodes(spec, outcome, state, DEFAULT_ENGINE_CONFIG)

        for node in nodes:
            assert node.id is not None and len(node.id) > 0
            assert node.label is not None and len(node.label) > 0
            assert node.severity in ["info", "warning", "error"]
            assert node.category in ["constraint", "driver", "risk", "alternative"]

    def test_node_ids_are_unique(self):
        """All node IDs are unique within a generation."""
        selections = [
            make_selection(asset_id="A001", criticality=0.9),
            make_selection(asset_id="A002", market="Chicago, IL"),
        ]
        outcome = make_ok_outcome(
            selected_assets=selections,
            critical_fraction=0.3,
            leverage_after=3.9,  # Near binding
        )
        spec = make_spec(max_net_leverage=4.0, max_critical_fraction=0.5)
        state = make_corporate_state()

        nodes = generate_explanation_nodes(spec, outcome, state, DEFAULT_ENGINE_CONFIG)

        ids = [n.id for n in nodes]
        assert len(ids) == len(set(ids)), f"Duplicate IDs found: {ids}"

    def test_nodes_have_appropriate_metrics(self):
        """Nodes reference appropriate metrics."""
        violation = ConstraintViolation(
            code="MIN_FIXED_CHARGE_COVERAGE",
            detail="Coverage below minimum",
            actual=2.5,
            limit=3.0,
        )
        outcome = make_infeasible_outcome(violations=[violation])
        spec = make_spec()
        state = make_corporate_state()

        nodes = generate_explanation_nodes(spec, outcome, state, DEFAULT_ENGINE_CONFIG)

        fcc_nodes = [n for n in nodes if n.metric == "fixed_charge_coverage"]
        assert len(fcc_nodes) >= 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestExplanationIntegration:
    """Integration tests for complete explanation generation."""

    def test_successful_selection_full_explanation(self):
        """Successful selection generates comprehensive explanation."""
        selections = [
            make_selection(
                asset_id="A001",
                criticality=0.2,
                leaseability_score=0.9,
                proceeds=60_000_000,
            ),
            make_selection(
                asset_id="A002",
                criticality=0.3,
                leaseability_score=0.8,
                proceeds=50_000_000,
            ),
        ]
        outcome = make_ok_outcome(
            selected_assets=selections,
            proceeds=110_000_000,
            leverage_before=4.0,
            leverage_after=3.0,
            fixed_charge_coverage_before=4.0,
            fixed_charge_coverage_after=3.3,  # Near 3.0 minimum
            critical_fraction=0.1,
        )
        spec = make_spec(
            target_amount=100_000_000,
            max_net_leverage=4.0,
            min_fixed_charge_coverage=3.0,
        )
        state = make_corporate_state()

        nodes = generate_explanation_nodes(spec, outcome, state, DEFAULT_ENGINE_CONFIG)

        # Should have driver nodes
        drivers = [n for n in nodes if n.category == "driver"]
        assert len(drivers) >= 1

        # Should have near-binding constraint node (FCC at 3.3 vs 3.0)
        constraints = [n for n in nodes if n.category == "constraint"]
        assert len(constraints) >= 1

        # All nodes should be valid
        for node in nodes:
            assert node.id
            assert node.label
            assert node.severity
            assert node.category

    def test_infeasible_selection_explanation(self):
        """Infeasible selection generates appropriate explanation."""
        violations = [
            ConstraintViolation(
                code="MAX_NET_LEVERAGE",
                detail="Leverage 5.0x exceeds limit 4.0x",
                actual=5.0,
                limit=4.0,
            ),
            ConstraintViolation(
                code="TARGET_NOT_MET",
                detail="Proceeds insufficient",
                actual=50_000_000,
                limit=100_000_000,
            ),
        ]
        outcome = make_infeasible_outcome(violations=violations)
        spec = make_spec(target_amount=100_000_000, max_net_leverage=4.0)
        state = make_corporate_state()

        nodes = generate_explanation_nodes(spec, outcome, state, DEFAULT_ENGINE_CONFIG)

        # Should have constraint nodes for each violation
        constraint_nodes = [n for n in nodes if n.category == "constraint"]
        assert len(constraint_nodes) >= 2

        # All should be errors
        errors = [n for n in constraint_nodes if n.severity == "error"]
        assert len(errors) >= 2

        # Should have no driver or risk nodes
        drivers = [n for n in nodes if n.category == "driver"]
        risks = [n for n in nodes if n.category == "risk"]
        assert len(drivers) == 0
        assert len(risks) == 0
