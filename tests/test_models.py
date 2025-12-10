"""
Tests for app/models.py

Covers:
- Pydantic validation (required fields, bounds)
- Enum serialization
- Invalid value rejection
"""

import pytest
from pydantic import ValidationError

from app.models import (
    Asset,
    AssetFilters,
    AssetSelection,
    AssetSLBMetrics,
    AssetType,
    ConstraintViolation,
    CorporateState,
    Explanation,
    ExplanationNode,
    HardConstraints,
    MarketTier,
    Objective,
    ProgramRequest,
    ProgramType,
    SelectionStatus,
    SelectorSpec,
    SoftPreferences,
)


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Test enum serialization and values."""

    def test_asset_type_values(self):
        assert AssetType.STORE.value == "store"
        assert AssetType.DISTRIBUTION_CENTER.value == "distribution_center"

    def test_market_tier_values(self):
        assert MarketTier.TIER_1.value == 1
        assert MarketTier.TIER_2.value == 2
        assert MarketTier.TIER_3.value == 3

    def test_program_type_values(self):
        assert ProgramType.SLB.value == "slb"

    def test_objective_values(self):
        assert Objective.MAXIMIZE_PROCEEDS.value == "maximize_proceeds"
        assert Objective.BALANCED.value == "balanced"

    def test_selection_status_values(self):
        assert SelectionStatus.OK.value == "ok"
        assert SelectionStatus.INFEASIBLE.value == "infeasible"

    def test_enum_json_serialization(self):
        """Enums should serialize to their string values."""
        asset = Asset(
            asset_id="A001",
            asset_type=AssetType.STORE,
            market="Dallas, TX",
            noi=500_000,
            book_value=4_000_000,
            criticality=0.3,
            leaseability_score=0.8,
        )
        data = asset.model_dump()
        assert data["asset_type"] == "store"


# =============================================================================
# Asset Tests
# =============================================================================


class TestAsset:
    """Test Asset model validation."""

    def test_valid_asset(self):
        """Valid asset should pass validation."""
        asset = Asset(
            asset_id="A001",
            asset_type=AssetType.STORE,
            market="Dallas, TX",
            noi=500_000,
            book_value=4_000_000,
            criticality=0.3,
            leaseability_score=0.8,
        )
        assert asset.asset_id == "A001"
        assert asset.noi == 500_000

    def test_asset_with_optional_fields(self):
        """Optional fields should work."""
        asset = Asset(
            asset_id="A001",
            asset_type=AssetType.DISTRIBUTION_CENTER,
            market="Chicago, IL",
            market_tier=MarketTier.TIER_1,
            noi=2_000_000,
            book_value=35_000_000,
            criticality=0.7,
            leaseability_score=0.6,
            name="Chicago DC",
            tenant_name="Acme Corp",
        )
        assert asset.market_tier == MarketTier.TIER_1
        assert asset.name == "Chicago DC"

    def test_asset_noi_must_be_positive(self):
        """NOI must be > 0."""
        with pytest.raises(ValidationError) as exc_info:
            Asset(
                asset_id="A001",
                asset_type=AssetType.STORE,
                market="Dallas, TX",
                noi=0,  # Invalid
                book_value=4_000_000,
                criticality=0.3,
                leaseability_score=0.8,
            )
        assert "noi" in str(exc_info.value)

    def test_asset_noi_negative_rejected(self):
        """Negative NOI should be rejected."""
        with pytest.raises(ValidationError):
            Asset(
                asset_id="A001",
                asset_type=AssetType.STORE,
                market="Dallas, TX",
                noi=-100_000,
                book_value=4_000_000,
                criticality=0.3,
                leaseability_score=0.8,
            )

    def test_asset_criticality_bounds(self):
        """Criticality must be in [0, 1]."""
        # Valid at boundaries
        Asset(
            asset_id="A001",
            asset_type=AssetType.STORE,
            market="Dallas, TX",
            noi=500_000,
            book_value=4_000_000,
            criticality=0.0,
            leaseability_score=0.8,
        )
        Asset(
            asset_id="A002",
            asset_type=AssetType.STORE,
            market="Dallas, TX",
            noi=500_000,
            book_value=4_000_000,
            criticality=1.0,
            leaseability_score=0.8,
        )

        # Invalid: > 1
        with pytest.raises(ValidationError):
            Asset(
                asset_id="A003",
                asset_type=AssetType.STORE,
                market="Dallas, TX",
                noi=500_000,
                book_value=4_000_000,
                criticality=1.1,
                leaseability_score=0.8,
            )

        # Invalid: < 0
        with pytest.raises(ValidationError):
            Asset(
                asset_id="A004",
                asset_type=AssetType.STORE,
                market="Dallas, TX",
                noi=500_000,
                book_value=4_000_000,
                criticality=-0.1,
                leaseability_score=0.8,
            )

    def test_asset_leaseability_bounds(self):
        """Leaseability must be in [0, 1]."""
        with pytest.raises(ValidationError):
            Asset(
                asset_id="A001",
                asset_type=AssetType.STORE,
                market="Dallas, TX",
                noi=500_000,
                book_value=4_000_000,
                criticality=0.3,
                leaseability_score=1.5,  # Invalid
            )

    def test_asset_missing_required_field(self):
        """Missing required fields should fail."""
        with pytest.raises(ValidationError):
            Asset(
                asset_id="A001",
                asset_type=AssetType.STORE,
                # market missing
                noi=500_000,
                book_value=4_000_000,
                criticality=0.3,
                leaseability_score=0.8,
            )


# =============================================================================
# CorporateState Tests
# =============================================================================


class TestCorporateState:
    """Test CorporateState model validation."""

    def test_valid_corporate_state(self):
        """Valid state should pass."""
        state = CorporateState(
            net_debt=2_000_000_000,
            ebitda=500_000_000,
            interest_expense=100_000_000,
        )
        assert state.net_debt == 2_000_000_000
        assert state.lease_expense is None  # Optional

    def test_corporate_state_with_lease_expense(self):
        """Lease expense is optional."""
        state = CorporateState(
            net_debt=2_000_000_000,
            ebitda=500_000_000,
            interest_expense=100_000_000,
            lease_expense=50_000_000,
        )
        assert state.lease_expense == 50_000_000

    def test_corporate_state_negative_ebitda_allowed(self):
        """Negative EBITDA is allowed (distressed company)."""
        state = CorporateState(
            net_debt=2_000_000_000,
            ebitda=-100_000_000,  # Distressed
            interest_expense=100_000_000,
        )
        assert state.ebitda == -100_000_000

    def test_corporate_state_negative_net_debt_rejected(self):
        """Net debt must be >= 0."""
        with pytest.raises(ValidationError):
            CorporateState(
                net_debt=-1_000_000,  # Invalid
                ebitda=500_000_000,
                interest_expense=100_000_000,
            )

    def test_corporate_state_negative_interest_rejected(self):
        """Interest expense must be >= 0."""
        with pytest.raises(ValidationError):
            CorporateState(
                net_debt=2_000_000_000,
                ebitda=500_000_000,
                interest_expense=-100_000,  # Invalid
            )


# =============================================================================
# Spec Model Tests
# =============================================================================


class TestHardConstraints:
    """Test HardConstraints validation."""

    def test_valid_hard_constraints(self):
        """Valid constraints should pass."""
        hc = HardConstraints(
            max_net_leverage=4.0,
            min_fixed_charge_coverage=3.0,
        )
        assert hc.max_net_leverage == 4.0
        assert hc.min_interest_coverage is None

    def test_hard_constraints_all_optional(self):
        """All fields are optional."""
        hc = HardConstraints()
        assert hc.max_net_leverage is None
        assert hc.min_interest_coverage is None
        assert hc.min_fixed_charge_coverage is None

    def test_hard_constraints_leverage_bounds(self):
        """Leverage must be in (0, 10)."""
        with pytest.raises(ValidationError):
            HardConstraints(max_net_leverage=0)  # Must be > 0

        with pytest.raises(ValidationError):
            HardConstraints(max_net_leverage=15)  # Must be < 10


class TestSoftPreferences:
    """Test SoftPreferences defaults and validation."""

    def test_soft_preferences_defaults(self):
        """Defaults should be applied."""
        sp = SoftPreferences()
        assert sp.prefer_low_criticality is True
        assert sp.prefer_high_leaseability is True
        assert sp.weight_criticality == 1.0
        assert sp.weight_leaseability == 1.0

    def test_soft_preferences_weights_non_negative(self):
        """Weights must be >= 0."""
        with pytest.raises(ValidationError):
            SoftPreferences(weight_criticality=-0.5)


class TestAssetFilters:
    """Test AssetFilters validation."""

    def test_asset_filters_all_none(self):
        """All filters can be None."""
        af = AssetFilters()
        assert af.include_asset_types is None
        assert af.exclude_markets is None

    def test_asset_filters_with_values(self):
        """Filters with values should work."""
        af = AssetFilters(
            include_asset_types=[AssetType.STORE, AssetType.OFFICE],
            exclude_markets=["NYC", "LA"],
            min_leaseability_score=0.5,
            max_criticality=0.7,
        )
        assert len(af.include_asset_types) == 2
        assert af.max_criticality == 0.7

    def test_asset_filters_score_bounds(self):
        """Score filters must be in [0, 1]."""
        with pytest.raises(ValidationError):
            AssetFilters(min_leaseability_score=1.5)

        with pytest.raises(ValidationError):
            AssetFilters(max_criticality=-0.1)


class TestSelectorSpec:
    """Test SelectorSpec validation."""

    def test_valid_selector_spec(self):
        """Valid spec should pass."""
        spec = SelectorSpec(
            program_type=ProgramType.SLB,
            objective=Objective.BALANCED,
            target_amount=500_000_000,
            hard_constraints=HardConstraints(max_net_leverage=4.0),
            soft_preferences=SoftPreferences(),
            asset_filters=AssetFilters(),
        )
        assert spec.target_amount == 500_000_000
        assert spec.max_iterations == 3  # Default

    def test_selector_spec_target_must_be_positive(self):
        """Target amount must be > 0."""
        with pytest.raises(ValidationError):
            SelectorSpec(
                program_type=ProgramType.SLB,
                objective=Objective.BALANCED,
                target_amount=0,  # Invalid
                hard_constraints=HardConstraints(),
                soft_preferences=SoftPreferences(),
                asset_filters=AssetFilters(),
            )

    def test_selector_spec_max_iterations_bounds(self):
        """Max iterations must be in [1, 10]."""
        with pytest.raises(ValidationError):
            SelectorSpec(
                program_type=ProgramType.SLB,
                objective=Objective.BALANCED,
                target_amount=100_000_000,
                hard_constraints=HardConstraints(),
                soft_preferences=SoftPreferences(),
                asset_filters=AssetFilters(),
                max_iterations=0,  # Invalid
            )

        with pytest.raises(ValidationError):
            SelectorSpec(
                program_type=ProgramType.SLB,
                objective=Objective.BALANCED,
                target_amount=100_000_000,
                hard_constraints=HardConstraints(),
                soft_preferences=SoftPreferences(),
                asset_filters=AssetFilters(),
                max_iterations=11,  # Invalid
            )


# =============================================================================
# Output Model Tests
# =============================================================================


class TestConstraintViolation:
    """Test ConstraintViolation model."""

    def test_valid_violation(self):
        """Valid violation should pass."""
        v = ConstraintViolation(
            code="MAX_NET_LEVERAGE",
            detail="Post-SLB leverage 4.5x exceeds limit",
            actual=4.5,
            limit=4.0,
        )
        assert v.code == "MAX_NET_LEVERAGE"
        assert v.actual == 4.5


class TestAssetSLBMetrics:
    """Test AssetSLBMetrics model."""

    def test_valid_metrics(self):
        """Valid metrics should pass."""
        m = AssetSLBMetrics(
            market_value=10_000_000,
            proceeds=9_750_000,
            slb_rent=650_000,
            cap_rate=0.065,
        )
        assert m.proceeds == 9_750_000

    def test_metrics_cap_rate_bounds(self):
        """Cap rate must be in (0, 1)."""
        with pytest.raises(ValidationError):
            AssetSLBMetrics(
                market_value=10_000_000,
                proceeds=9_750_000,
                slb_rent=650_000,
                cap_rate=0,  # Invalid
            )

        with pytest.raises(ValidationError):
            AssetSLBMetrics(
                market_value=10_000_000,
                proceeds=9_750_000,
                slb_rent=650_000,
                cap_rate=1.5,  # Invalid
            )


class TestAssetSelection:
    """Test AssetSelection model."""

    def test_valid_selection(self):
        """Valid selection should pass."""
        asset = Asset(
            asset_id="A001",
            asset_type=AssetType.STORE,
            market="Dallas, TX",
            noi=500_000,
            book_value=4_000_000,
            criticality=0.3,
            leaseability_score=0.8,
        )
        selection = AssetSelection(
            asset=asset,
            proceeds=9_500_000,
            slb_rent=500_000,
        )
        assert selection.asset.asset_id == "A001"
        assert selection.proceeds == 9_500_000


# =============================================================================
# Explanation Model Tests
# =============================================================================


class TestExplanationNode:
    """Test ExplanationNode model."""

    def test_valid_node(self):
        """Valid node should pass."""
        node = ExplanationNode(
            id="constraint_leverage",
            label="Leverage constraint binding",
            severity="warning",
            category="constraint",
            metric="leverage",
            baseline_value=4.0,
            post_value=3.5,
            threshold=4.0,
        )
        assert node.severity == "warning"
        assert node.category == "constraint"

    def test_node_severity_literal(self):
        """Severity must be one of the allowed values."""
        with pytest.raises(ValidationError):
            ExplanationNode(
                id="test",
                label="Test",
                severity="critical",  # Invalid
                category="constraint",
            )

    def test_node_category_literal(self):
        """Category must be one of the allowed values."""
        with pytest.raises(ValidationError):
            ExplanationNode(
                id="test",
                label="Test",
                severity="info",
                category="unknown",  # Invalid
            )


class TestExplanation:
    """Test Explanation model."""

    def test_valid_explanation(self):
        """Valid explanation should pass."""
        node = ExplanationNode(
            id="driver_low_crit",
            label="Low criticality assets selected",
            severity="info",
            category="driver",
        )
        explanation = Explanation(
            summary="Successfully structured a $485M SLB program.",
            nodes=[node],
        )
        assert len(explanation.nodes) == 1

    def test_explanation_empty_nodes(self):
        """Empty nodes list is valid."""
        explanation = Explanation(summary="No selection made.")
        assert explanation.nodes == []


# =============================================================================
# API Model Tests
# =============================================================================


class TestProgramRequest:
    """Test ProgramRequest model."""

    def test_valid_request(self):
        """Valid request should pass."""
        asset = Asset(
            asset_id="A001",
            asset_type=AssetType.STORE,
            market="Dallas, TX",
            noi=500_000,
            book_value=4_000_000,
            criticality=0.3,
            leaseability_score=0.8,
        )
        state = CorporateState(
            net_debt=2_000_000_000,
            ebitda=500_000_000,
            interest_expense=100_000_000,
        )
        request = ProgramRequest(
            assets=[asset],
            corporate_state=state,
            program_type=ProgramType.SLB,
            program_description="Raise ~$500M via SLB",
        )
        assert len(request.assets) == 1

    def test_request_empty_assets_rejected(self):
        """Empty asset list should be rejected."""
        state = CorporateState(
            net_debt=2_000_000_000,
            ebitda=500_000_000,
            interest_expense=100_000_000,
        )
        with pytest.raises(ValidationError):
            ProgramRequest(
                assets=[],  # Invalid
                corporate_state=state,
                program_type=ProgramType.SLB,
                program_description="Raise ~$500M via SLB",
            )

    def test_request_empty_description_rejected(self):
        """Empty description should be rejected."""
        asset = Asset(
            asset_id="A001",
            asset_type=AssetType.STORE,
            market="Dallas, TX",
            noi=500_000,
            book_value=4_000_000,
            criticality=0.3,
            leaseability_score=0.8,
        )
        state = CorporateState(
            net_debt=2_000_000_000,
            ebitda=500_000_000,
            interest_expense=100_000_000,
        )
        with pytest.raises(ValidationError):
            ProgramRequest(
                assets=[asset],
                corporate_state=state,
                program_type=ProgramType.SLB,
                program_description="",  # Invalid
            )


# =============================================================================
# Serialization Tests
# =============================================================================


class TestSerialization:
    """Test model serialization to/from JSON."""

    def test_asset_roundtrip(self):
        """Asset should serialize and deserialize correctly."""
        asset = Asset(
            asset_id="A001",
            asset_type=AssetType.STORE,
            market="Dallas, TX",
            market_tier=MarketTier.TIER_2,
            noi=500_000,
            book_value=4_000_000,
            criticality=0.3,
            leaseability_score=0.8,
        )
        json_str = asset.model_dump_json()
        restored = Asset.model_validate_json(json_str)
        assert restored.asset_id == asset.asset_id
        assert restored.market_tier == MarketTier.TIER_2

    def test_selector_spec_roundtrip(self):
        """SelectorSpec should serialize and deserialize correctly."""
        spec = SelectorSpec(
            program_type=ProgramType.SLB,
            objective=Objective.BALANCED,
            target_amount=500_000_000,
            hard_constraints=HardConstraints(max_net_leverage=4.0),
            soft_preferences=SoftPreferences(),
            asset_filters=AssetFilters(
                include_asset_types=[AssetType.STORE],
            ),
        )
        json_str = spec.model_dump_json()
        restored = SelectorSpec.model_validate_json(json_str)
        assert restored.target_amount == 500_000_000
        assert restored.hard_constraints.max_net_leverage == 4.0
