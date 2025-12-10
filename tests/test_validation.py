"""
Tests for app/validation.py

Covers:
- Asset validation (individual and list)
- CorporateState validation
- SelectorSpec validation
- ValidationError exception
"""

import pytest

from app.models import (
    Asset,
    AssetFilters,
    AssetType,
    CorporateState,
    HardConstraints,
    Objective,
    ProgramType,
    SelectorSpec,
    SoftPreferences,
)
from app.validation import (
    ValidationError,
    validate_and_raise,
    validate_asset,
    validate_assets,
    validate_corporate_state,
    validate_spec,
)


# =============================================================================
# Test Fixtures
# =============================================================================


def make_valid_asset(asset_id: str = "A001") -> Asset:
    """Create a valid asset for testing."""
    return Asset(
        asset_id=asset_id,
        asset_type=AssetType.STORE,
        market="Dallas, TX",
        noi=500_000,
        book_value=4_000_000,
        criticality=0.3,
        leaseability_score=0.8,
    )


def make_valid_corporate_state() -> CorporateState:
    """Create a valid corporate state for testing."""
    return CorporateState(
        net_debt=2_000_000_000,
        ebitda=500_000_000,
        interest_expense=100_000_000,
    )


def make_valid_spec() -> SelectorSpec:
    """Create a valid selector spec for testing."""
    return SelectorSpec(
        program_type=ProgramType.SLB,
        objective=Objective.BALANCED,
        target_amount=500_000_000,
        hard_constraints=HardConstraints(max_net_leverage=4.0),
        soft_preferences=SoftPreferences(),
        asset_filters=AssetFilters(),
    )


# =============================================================================
# ValidationError Tests
# =============================================================================


class TestValidationError:
    """Test ValidationError exception."""

    def test_single_error(self):
        """Single error should be in message."""
        error = ValidationError(["Field X is invalid"])
        assert "Field X is invalid" in str(error)
        assert len(error.errors) == 1

    def test_multiple_errors(self):
        """Multiple errors should be joined."""
        error = ValidationError(["Error 1", "Error 2"])
        assert "Error 1" in str(error)
        assert "Error 2" in str(error)
        assert len(error.errors) == 2

    def test_empty_errors(self):
        """Empty errors list should have fallback message."""
        error = ValidationError([])
        assert "Validation failed" in str(error)


# =============================================================================
# Asset Validation Tests
# =============================================================================


class TestValidateAsset:
    """Test validate_asset function."""

    def test_valid_asset(self):
        """Valid asset should return no errors."""
        asset = make_valid_asset()
        errors = validate_asset(asset)
        assert errors == []

    def test_empty_asset_id(self):
        """Empty asset_id should be caught."""
        # Create asset directly with model_construct to bypass Pydantic
        asset = Asset.model_construct(
            asset_id="",
            asset_type=AssetType.STORE,
            market="Dallas, TX",
            noi=500_000,
            book_value=4_000_000,
            criticality=0.3,
            leaseability_score=0.8,
        )
        errors = validate_asset(asset)
        assert any("empty asset_id" in e for e in errors)

    def test_empty_market(self):
        """Empty market should be caught."""
        asset = Asset.model_construct(
            asset_id="A001",
            asset_type=AssetType.STORE,
            market="   ",  # Whitespace only
            noi=500_000,
            book_value=4_000_000,
            criticality=0.3,
            leaseability_score=0.8,
        )
        errors = validate_asset(asset)
        assert any("market must not be empty" in e for e in errors)

    def test_negative_noi(self):
        """Negative NOI should be caught."""
        asset = Asset.model_construct(
            asset_id="A001",
            asset_type=AssetType.STORE,
            market="Dallas, TX",
            noi=-100,
            book_value=4_000_000,
            criticality=0.3,
            leaseability_score=0.8,
        )
        errors = validate_asset(asset)
        assert any("noi must be > 0" in e for e in errors)

    def test_zero_book_value(self):
        """Zero book_value should be caught."""
        asset = Asset.model_construct(
            asset_id="A001",
            asset_type=AssetType.STORE,
            market="Dallas, TX",
            noi=500_000,
            book_value=0,
            criticality=0.3,
            leaseability_score=0.8,
        )
        errors = validate_asset(asset)
        assert any("book_value must be > 0" in e for e in errors)

    def test_criticality_out_of_bounds(self):
        """Criticality > 1 should be caught."""
        asset = Asset.model_construct(
            asset_id="A001",
            asset_type=AssetType.STORE,
            market="Dallas, TX",
            noi=500_000,
            book_value=4_000_000,
            criticality=1.5,
            leaseability_score=0.8,
        )
        errors = validate_asset(asset)
        assert any("criticality must be in [0, 1]" in e for e in errors)

    def test_leaseability_out_of_bounds(self):
        """Leaseability < 0 should be caught."""
        asset = Asset.model_construct(
            asset_id="A001",
            asset_type=AssetType.STORE,
            market="Dallas, TX",
            noi=500_000,
            book_value=4_000_000,
            criticality=0.3,
            leaseability_score=-0.1,
        )
        errors = validate_asset(asset)
        assert any("leaseability_score must be in [0, 1]" in e for e in errors)


class TestValidateAssets:
    """Test validate_assets function."""

    def test_valid_assets(self):
        """Valid asset list should return no errors."""
        assets = [make_valid_asset("A001"), make_valid_asset("A002")]
        errors = validate_assets(assets)
        assert errors == []

    def test_empty_list(self):
        """Empty list should return error."""
        errors = validate_assets([])
        assert any("cannot be empty" in e for e in errors)

    def test_duplicate_ids(self):
        """Duplicate asset IDs should be caught."""
        assets = [
            make_valid_asset("A001"),
            make_valid_asset("A001"),  # Duplicate
            make_valid_asset("A002"),
        ]
        errors = validate_assets(assets)
        assert any("must be unique" in e for e in errors)
        assert any("A001" in e for e in errors)

    def test_invalid_asset_in_list(self):
        """Invalid asset in list should be caught."""
        valid_asset = make_valid_asset("A001")
        invalid_asset = Asset.model_construct(
            asset_id="A002",
            asset_type=AssetType.STORE,
            market="Dallas, TX",
            noi=-100,  # Invalid
            book_value=4_000_000,
            criticality=0.3,
            leaseability_score=0.8,
        )
        errors = validate_assets([valid_asset, invalid_asset])
        assert any("A002" in e and "noi" in e for e in errors)

    def test_multiple_errors(self):
        """Multiple errors should all be collected."""
        invalid1 = Asset.model_construct(
            asset_id="A001",
            asset_type=AssetType.STORE,
            market="Dallas, TX",
            noi=-100,
            book_value=4_000_000,
            criticality=0.3,
            leaseability_score=0.8,
        )
        invalid2 = Asset.model_construct(
            asset_id="A002",
            asset_type=AssetType.STORE,
            market="",
            noi=500_000,
            book_value=0,
            criticality=0.3,
            leaseability_score=0.8,
        )
        errors = validate_assets([invalid1, invalid2])
        # Should have errors from both assets
        assert len(errors) >= 2


# =============================================================================
# CorporateState Validation Tests
# =============================================================================


class TestValidateCorporateState:
    """Test validate_corporate_state function."""

    def test_valid_state(self):
        """Valid state should return no errors."""
        state = make_valid_corporate_state()
        errors = validate_corporate_state(state)
        assert errors == []

    def test_negative_net_debt(self):
        """Negative net_debt should be caught."""
        state = CorporateState.model_construct(
            net_debt=-1_000_000,
            ebitda=500_000_000,
            interest_expense=100_000_000,
        )
        errors = validate_corporate_state(state)
        assert any("net_debt must be >= 0" in e for e in errors)

    def test_negative_ebitda_allowed(self):
        """Negative EBITDA is allowed (distressed company)."""
        state = CorporateState(
            net_debt=2_000_000_000,
            ebitda=-100_000_000,  # Distressed
            interest_expense=100_000_000,
        )
        errors = validate_corporate_state(state)
        assert errors == []

    def test_negative_interest_expense(self):
        """Negative interest_expense should be caught."""
        state = CorporateState.model_construct(
            net_debt=2_000_000_000,
            ebitda=500_000_000,
            interest_expense=-50_000,
        )
        errors = validate_corporate_state(state)
        assert any("interest_expense must be >= 0" in e for e in errors)

    def test_negative_lease_expense(self):
        """Negative lease_expense should be caught."""
        state = CorporateState.model_construct(
            net_debt=2_000_000_000,
            ebitda=500_000_000,
            interest_expense=100_000_000,
            lease_expense=-10_000,
        )
        errors = validate_corporate_state(state)
        assert any("lease_expense must be >= 0" in e for e in errors)

    def test_none_lease_expense_ok(self):
        """None lease_expense is valid."""
        state = CorporateState(
            net_debt=2_000_000_000,
            ebitda=500_000_000,
            interest_expense=100_000_000,
            lease_expense=None,
        )
        errors = validate_corporate_state(state)
        assert errors == []

    def test_zero_values_ok(self):
        """Zero net_debt and interest_expense are valid."""
        state = CorporateState(
            net_debt=0,
            ebitda=500_000_000,
            interest_expense=0,
        )
        errors = validate_corporate_state(state)
        assert errors == []


# =============================================================================
# SelectorSpec Validation Tests
# =============================================================================


class TestValidateSpec:
    """Test validate_spec function."""

    def test_valid_spec(self):
        """Valid spec should return no errors."""
        spec = make_valid_spec()
        errors = validate_spec(spec)
        assert errors == []

    def test_zero_target_amount(self):
        """Zero target_amount should be caught."""
        spec = SelectorSpec.model_construct(
            program_type=ProgramType.SLB,
            objective=Objective.BALANCED,
            target_amount=0,
            hard_constraints=HardConstraints(),
            soft_preferences=SoftPreferences(),
            asset_filters=AssetFilters(),
            max_iterations=3,
        )
        errors = validate_spec(spec)
        assert any("target_amount must be positive" in e for e in errors)

    def test_leverage_out_of_bounds(self):
        """max_net_leverage >= 10 should be caught."""
        spec = SelectorSpec.model_construct(
            program_type=ProgramType.SLB,
            objective=Objective.BALANCED,
            target_amount=100_000_000,
            hard_constraints=HardConstraints.model_construct(
                max_net_leverage=15,
                min_interest_coverage=None,
                min_fixed_charge_coverage=None,
                max_critical_fraction=None,
            ),
            soft_preferences=SoftPreferences(),
            asset_filters=AssetFilters(),
            max_iterations=3,
        )
        errors = validate_spec(spec)
        assert any("max_net_leverage must be in (0, 10)" in e for e in errors)

    def test_interest_coverage_out_of_bounds(self):
        """min_interest_coverage >= 50 should be caught."""
        spec = SelectorSpec.model_construct(
            program_type=ProgramType.SLB,
            objective=Objective.BALANCED,
            target_amount=100_000_000,
            hard_constraints=HardConstraints.model_construct(
                max_net_leverage=None,
                min_interest_coverage=60,
                min_fixed_charge_coverage=None,
                max_critical_fraction=None,
            ),
            soft_preferences=SoftPreferences(),
            asset_filters=AssetFilters(),
            max_iterations=3,
        )
        errors = validate_spec(spec)
        assert any("min_interest_coverage must be in (0, 50)" in e for e in errors)

    def test_fixed_charge_coverage_out_of_bounds(self):
        """min_fixed_charge_coverage >= 20 should be caught."""
        spec = SelectorSpec.model_construct(
            program_type=ProgramType.SLB,
            objective=Objective.BALANCED,
            target_amount=100_000_000,
            hard_constraints=HardConstraints.model_construct(
                max_net_leverage=None,
                min_interest_coverage=None,
                min_fixed_charge_coverage=25,
                max_critical_fraction=None,
            ),
            soft_preferences=SoftPreferences(),
            asset_filters=AssetFilters(),
            max_iterations=3,
        )
        errors = validate_spec(spec)
        assert any("min_fixed_charge_coverage must be in (0, 20)" in e for e in errors)

    def test_critical_fraction_out_of_bounds(self):
        """max_critical_fraction > 1 should be caught."""
        spec = SelectorSpec.model_construct(
            program_type=ProgramType.SLB,
            objective=Objective.BALANCED,
            target_amount=100_000_000,
            hard_constraints=HardConstraints.model_construct(
                max_net_leverage=None,
                min_interest_coverage=None,
                min_fixed_charge_coverage=None,
                max_critical_fraction=1.5,
            ),
            soft_preferences=SoftPreferences(),
            asset_filters=AssetFilters(),
            max_iterations=3,
        )
        errors = validate_spec(spec)
        assert any("max_critical_fraction must be in (0, 1]" in e for e in errors)

    def test_critical_fraction_zero(self):
        """max_critical_fraction = 0 should be caught."""
        spec = SelectorSpec.model_construct(
            program_type=ProgramType.SLB,
            objective=Objective.BALANCED,
            target_amount=100_000_000,
            hard_constraints=HardConstraints.model_construct(
                max_net_leverage=None,
                min_interest_coverage=None,
                min_fixed_charge_coverage=None,
                max_critical_fraction=0,
            ),
            soft_preferences=SoftPreferences(),
            asset_filters=AssetFilters(),
            max_iterations=3,
        )
        errors = validate_spec(spec)
        assert any("max_critical_fraction must be in (0, 1]" in e for e in errors)

    def test_negative_weight_criticality(self):
        """Negative weight_criticality should be caught."""
        spec = SelectorSpec.model_construct(
            program_type=ProgramType.SLB,
            objective=Objective.BALANCED,
            target_amount=100_000_000,
            hard_constraints=HardConstraints(),
            soft_preferences=SoftPreferences.model_construct(
                prefer_low_criticality=True,
                prefer_high_leaseability=True,
                weight_criticality=-1,
                weight_leaseability=1,
            ),
            asset_filters=AssetFilters(),
            max_iterations=3,
        )
        errors = validate_spec(spec)
        assert any("weight_criticality must be non-negative" in e for e in errors)

    def test_negative_weight_leaseability(self):
        """Negative weight_leaseability should be caught."""
        spec = SelectorSpec.model_construct(
            program_type=ProgramType.SLB,
            objective=Objective.BALANCED,
            target_amount=100_000_000,
            hard_constraints=HardConstraints(),
            soft_preferences=SoftPreferences.model_construct(
                prefer_low_criticality=True,
                prefer_high_leaseability=True,
                weight_criticality=1,
                weight_leaseability=-0.5,
            ),
            asset_filters=AssetFilters(),
            max_iterations=3,
        )
        errors = validate_spec(spec)
        assert any("weight_leaseability must be non-negative" in e for e in errors)

    def test_filter_leaseability_out_of_bounds(self):
        """min_leaseability_score > 1 should be caught."""
        spec = SelectorSpec.model_construct(
            program_type=ProgramType.SLB,
            objective=Objective.BALANCED,
            target_amount=100_000_000,
            hard_constraints=HardConstraints(),
            soft_preferences=SoftPreferences(),
            asset_filters=AssetFilters.model_construct(
                include_asset_types=None,
                exclude_asset_types=None,
                exclude_markets=None,
                min_leaseability_score=1.5,
                max_criticality=None,
            ),
            max_iterations=3,
        )
        errors = validate_spec(spec)
        assert any("min_leaseability_score must be in [0, 1]" in e for e in errors)

    def test_filter_criticality_out_of_bounds(self):
        """max_criticality < 0 should be caught."""
        spec = SelectorSpec.model_construct(
            program_type=ProgramType.SLB,
            objective=Objective.BALANCED,
            target_amount=100_000_000,
            hard_constraints=HardConstraints(),
            soft_preferences=SoftPreferences(),
            asset_filters=AssetFilters.model_construct(
                include_asset_types=None,
                exclude_asset_types=None,
                exclude_markets=None,
                min_leaseability_score=None,
                max_criticality=-0.1,
            ),
            max_iterations=3,
        )
        errors = validate_spec(spec)
        assert any("max_criticality must be in [0, 1]" in e for e in errors)

    def test_max_iterations_too_low(self):
        """max_iterations < 1 should be caught."""
        spec = SelectorSpec.model_construct(
            program_type=ProgramType.SLB,
            objective=Objective.BALANCED,
            target_amount=100_000_000,
            hard_constraints=HardConstraints(),
            soft_preferences=SoftPreferences(),
            asset_filters=AssetFilters(),
            max_iterations=0,
        )
        errors = validate_spec(spec)
        assert any("max_iterations must be at least 1" in e for e in errors)

    def test_max_iterations_too_high(self):
        """max_iterations > 10 should be caught."""
        spec = SelectorSpec.model_construct(
            program_type=ProgramType.SLB,
            objective=Objective.BALANCED,
            target_amount=100_000_000,
            hard_constraints=HardConstraints(),
            soft_preferences=SoftPreferences(),
            asset_filters=AssetFilters(),
            max_iterations=15,
        )
        errors = validate_spec(spec)
        assert any("max_iterations should not exceed 10" in e for e in errors)

    def test_none_constraints_ok(self):
        """None values for optional constraints are valid."""
        spec = SelectorSpec(
            program_type=ProgramType.SLB,
            objective=Objective.BALANCED,
            target_amount=100_000_000,
            hard_constraints=HardConstraints(),  # All None
            soft_preferences=SoftPreferences(),
            asset_filters=AssetFilters(),  # All None
        )
        errors = validate_spec(spec)
        assert errors == []


# =============================================================================
# validate_and_raise Tests
# =============================================================================


class TestValidateAndRaise:
    """Test validate_and_raise convenience function."""

    def test_all_valid(self):
        """Valid inputs should not raise."""
        assets = [make_valid_asset()]
        state = make_valid_corporate_state()
        spec = make_valid_spec()

        # Should not raise
        validate_and_raise(assets=assets, corporate_state=state, spec=spec)

    def test_invalid_assets_raises(self):
        """Invalid assets should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_and_raise(assets=[])
        assert "cannot be empty" in str(exc_info.value)

    def test_invalid_state_raises(self):
        """Invalid corporate state should raise ValidationError."""
        state = CorporateState.model_construct(
            net_debt=-1,
            ebitda=100,
            interest_expense=10,
        )
        with pytest.raises(ValidationError) as exc_info:
            validate_and_raise(corporate_state=state)
        assert "net_debt" in str(exc_info.value)

    def test_invalid_spec_raises(self):
        """Invalid spec should raise ValidationError."""
        spec = SelectorSpec.model_construct(
            program_type=ProgramType.SLB,
            objective=Objective.BALANCED,
            target_amount=-100,
            hard_constraints=HardConstraints(),
            soft_preferences=SoftPreferences(),
            asset_filters=AssetFilters(),
            max_iterations=3,
        )
        with pytest.raises(ValidationError) as exc_info:
            validate_and_raise(spec=spec)
        assert "target_amount" in str(exc_info.value)

    def test_multiple_errors_collected(self):
        """Multiple validation errors should all be collected."""
        invalid_asset = Asset.model_construct(
            asset_id="A001",
            asset_type=AssetType.STORE,
            market="Dallas, TX",
            noi=-100,
            book_value=4_000_000,
            criticality=0.3,
            leaseability_score=0.8,
        )
        invalid_state = CorporateState.model_construct(
            net_debt=-1,
            ebitda=100,
            interest_expense=10,
        )

        with pytest.raises(ValidationError) as exc_info:
            validate_and_raise(assets=[invalid_asset], corporate_state=invalid_state)

        # Should have errors from both
        assert len(exc_info.value.errors) >= 2

    def test_none_inputs_ok(self):
        """None inputs should be skipped."""
        # Should not raise
        validate_and_raise(assets=None, corporate_state=None, spec=None)
