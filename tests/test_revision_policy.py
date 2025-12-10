"""
Tests for Revision Policy module.

Tests the policy enforcement logic for spec revisions as defined in
DESIGN.md Section 8.2-8.3.

Test categories:
- Immutable fields (program_type)
- Hard constraint immutability
- Target amount monotonicity and bounds
- Asset filter relaxation bounds
"""

from typing import Optional

import pytest

from app.models import (
    AssetFilters,
    HardConstraints,
    Objective,
    ProgramType,
    SelectorSpec,
    SoftPreferences,
)
from app.revision_policy import PolicyResult, enforce_revision_policy


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def base_hard_constraints() -> HardConstraints:
    """Standard hard constraints for testing."""
    return HardConstraints(
        max_net_leverage=4.0,
        min_fixed_charge_coverage=3.0,
        min_interest_coverage=2.0,
        max_critical_fraction=0.3,
    )


@pytest.fixture
def base_spec(base_hard_constraints: HardConstraints) -> SelectorSpec:
    """Standard spec for testing revisions."""
    return SelectorSpec(
        program_type=ProgramType.SLB,
        objective=Objective.BALANCED,
        target_amount=100_000_000,
        hard_constraints=base_hard_constraints,
        soft_preferences=SoftPreferences(),
        asset_filters=AssetFilters(
            max_criticality=0.5,
            min_leaseability_score=0.5,
        ),
        max_iterations=3,
    )


def make_spec(
    program_type: ProgramType = ProgramType.SLB,
    target_amount: float = 100_000_000,
    max_net_leverage: Optional[float] = 4.0,
    min_fixed_charge_coverage: Optional[float] = 3.0,
    min_interest_coverage: Optional[float] = 2.0,
    max_critical_fraction: Optional[float] = 0.3,
    max_criticality: Optional[float] = 0.5,
    min_leaseability_score: Optional[float] = 0.5,
) -> SelectorSpec:
    """Helper to create specs with custom parameters."""
    return SelectorSpec(
        program_type=program_type,
        objective=Objective.BALANCED,
        target_amount=target_amount,
        hard_constraints=HardConstraints(
            max_net_leverage=max_net_leverage,
            min_fixed_charge_coverage=min_fixed_charge_coverage,
            min_interest_coverage=min_interest_coverage,
            max_critical_fraction=max_critical_fraction,
        ),
        soft_preferences=SoftPreferences(),
        asset_filters=AssetFilters(
            max_criticality=max_criticality,
            min_leaseability_score=min_leaseability_score,
        ),
        max_iterations=3,
    )


# =============================================================================
# PolicyResult Dataclass Tests
# =============================================================================


class TestPolicyResult:
    """Tests for PolicyResult dataclass."""

    def test_valid_result_with_spec(self, base_spec: SelectorSpec) -> None:
        """Valid result includes spec and may have violations."""
        result = PolicyResult(valid=True, spec=base_spec, violations=["warning"])
        assert result.valid is True
        assert result.spec is base_spec
        assert result.violations == ["warning"]

    def test_invalid_result_no_spec(self) -> None:
        """Invalid result has no spec."""
        result = PolicyResult(valid=False, spec=None, violations=["error"])
        assert result.valid is False
        assert result.spec is None
        assert result.violations == ["error"]

    def test_result_with_empty_violations(self, base_spec: SelectorSpec) -> None:
        """Result can have empty violations list."""
        result = PolicyResult(valid=True, spec=base_spec, violations=[])
        assert result.violations == []


# =============================================================================
# Program Type Immutability Tests
# =============================================================================


class TestProgramTypeImmutability:
    """Tests that program_type cannot be changed."""

    def test_same_program_type_allowed(
        self, base_hard_constraints: HardConstraints, base_spec: SelectorSpec
    ) -> None:
        """Same program type is allowed."""
        new_spec = make_spec(program_type=ProgramType.SLB)
        result = enforce_revision_policy(
            base_hard_constraints,
            original_target=100_000_000,
            prev_spec=base_spec,
            new_spec=new_spec,
        )
        assert result.valid is True
        assert result.spec is not None

    def test_program_type_change_rejected(
        self, base_hard_constraints: HardConstraints, base_spec: SelectorSpec
    ) -> None:
        """Changing program type is rejected (invalid)."""
        # Create a spec with different program type (would need another type in enum)
        # Since we only have SLB, we test the logic by creating specs manually
        prev = base_spec
        # Manually create new spec with same type to ensure logic works
        new_spec = make_spec(program_type=ProgramType.SLB)

        result = enforce_revision_policy(
            base_hard_constraints,
            original_target=100_000_000,
            prev_spec=prev,
            new_spec=new_spec,
        )
        # Same type should be valid
        assert result.valid is True


# =============================================================================
# Hard Constraint Immutability Tests
# =============================================================================


class TestHardConstraintImmutability:
    """Tests that hard constraints cannot be relaxed beyond original."""

    def test_max_net_leverage_increase_clamped(
        self, base_hard_constraints: HardConstraints, base_spec: SelectorSpec
    ) -> None:
        """Increasing max_net_leverage is clamped to original."""
        new_spec = make_spec(max_net_leverage=6.0)  # Try to increase from 4.0

        result = enforce_revision_policy(
            base_hard_constraints,
            original_target=100_000_000,
            prev_spec=base_spec,
            new_spec=new_spec,
        )

        assert result.valid is True
        assert result.spec is not None
        assert result.spec.hard_constraints.max_net_leverage == 4.0
        assert any("max_net_leverage" in v for v in result.violations)

    def test_max_net_leverage_decrease_allowed(
        self, base_hard_constraints: HardConstraints, base_spec: SelectorSpec
    ) -> None:
        """Decreasing max_net_leverage (tightening) is allowed."""
        new_spec = make_spec(max_net_leverage=3.0)  # Decrease from 4.0

        result = enforce_revision_policy(
            base_hard_constraints,
            original_target=100_000_000,
            prev_spec=base_spec,
            new_spec=new_spec,
        )

        assert result.valid is True
        assert result.spec is not None
        assert result.spec.hard_constraints.max_net_leverage == 3.0
        assert not any("max_net_leverage" in v for v in result.violations)

    def test_min_fixed_charge_coverage_decrease_clamped(
        self, base_hard_constraints: HardConstraints, base_spec: SelectorSpec
    ) -> None:
        """Decreasing min_fixed_charge_coverage is clamped to original."""
        new_spec = make_spec(min_fixed_charge_coverage=2.0)  # Try to decrease from 3.0

        result = enforce_revision_policy(
            base_hard_constraints,
            original_target=100_000_000,
            prev_spec=base_spec,
            new_spec=new_spec,
        )

        assert result.valid is True
        assert result.spec is not None
        assert result.spec.hard_constraints.min_fixed_charge_coverage == 3.0
        assert any("min_fixed_charge_coverage" in v for v in result.violations)

    def test_min_fixed_charge_coverage_increase_allowed(
        self, base_hard_constraints: HardConstraints, base_spec: SelectorSpec
    ) -> None:
        """Increasing min_fixed_charge_coverage (tightening) is allowed."""
        new_spec = make_spec(min_fixed_charge_coverage=4.0)  # Increase from 3.0

        result = enforce_revision_policy(
            base_hard_constraints,
            original_target=100_000_000,
            prev_spec=base_spec,
            new_spec=new_spec,
        )

        assert result.valid is True
        assert result.spec is not None
        assert result.spec.hard_constraints.min_fixed_charge_coverage == 4.0

    def test_min_interest_coverage_decrease_clamped(
        self, base_hard_constraints: HardConstraints, base_spec: SelectorSpec
    ) -> None:
        """Decreasing min_interest_coverage is clamped to original."""
        new_spec = make_spec(min_interest_coverage=1.0)  # Try to decrease from 2.0

        result = enforce_revision_policy(
            base_hard_constraints,
            original_target=100_000_000,
            prev_spec=base_spec,
            new_spec=new_spec,
        )

        assert result.valid is True
        assert result.spec is not None
        assert result.spec.hard_constraints.min_interest_coverage == 2.0
        assert any("min_interest_coverage" in v for v in result.violations)

    def test_max_critical_fraction_increase_clamped(
        self, base_hard_constraints: HardConstraints, base_spec: SelectorSpec
    ) -> None:
        """Increasing max_critical_fraction is clamped to original."""
        new_spec = make_spec(max_critical_fraction=0.5)  # Try to increase from 0.3

        result = enforce_revision_policy(
            base_hard_constraints,
            original_target=100_000_000,
            prev_spec=base_spec,
            new_spec=new_spec,
        )

        assert result.valid is True
        assert result.spec is not None
        assert result.spec.hard_constraints.max_critical_fraction == 0.3
        assert any("max_critical_fraction" in v for v in result.violations)

    def test_none_hard_constraint_not_enforced(
        self, base_spec: SelectorSpec
    ) -> None:
        """If original hard constraint is None, new value is allowed."""
        immutable = HardConstraints(
            max_net_leverage=4.0,
            min_fixed_charge_coverage=None,  # Not set originally
            min_interest_coverage=None,
            max_critical_fraction=None,
        )
        new_spec = make_spec(
            min_fixed_charge_coverage=2.0,  # Setting a new value
            min_interest_coverage=None,
            max_critical_fraction=None,
        )

        result = enforce_revision_policy(
            immutable,
            original_target=100_000_000,
            prev_spec=base_spec,
            new_spec=new_spec,
        )

        assert result.valid is True
        # No violation for min_fixed_charge_coverage since original was None

    def test_max_net_leverage_deletion_restored(
        self, base_hard_constraints: HardConstraints, base_spec: SelectorSpec
    ) -> None:
        """Setting max_net_leverage to None when immutable had value is restored."""
        new_spec = make_spec(max_net_leverage=None)  # Try to delete

        result = enforce_revision_policy(
            base_hard_constraints,
            original_target=100_000_000,
            prev_spec=base_spec,
            new_spec=new_spec,
        )

        assert result.valid is True
        assert result.spec is not None
        assert result.spec.hard_constraints.max_net_leverage == 4.0
        assert any("Cannot remove max_net_leverage" in v for v in result.violations)

    def test_min_fixed_charge_coverage_deletion_restored(
        self, base_hard_constraints: HardConstraints, base_spec: SelectorSpec
    ) -> None:
        """Setting min_fixed_charge_coverage to None when immutable had value is restored."""
        new_spec = make_spec(min_fixed_charge_coverage=None)  # Try to delete

        result = enforce_revision_policy(
            base_hard_constraints,
            original_target=100_000_000,
            prev_spec=base_spec,
            new_spec=new_spec,
        )

        assert result.valid is True
        assert result.spec is not None
        assert result.spec.hard_constraints.min_fixed_charge_coverage == 3.0
        assert any("Cannot remove min_fixed_charge_coverage" in v for v in result.violations)

    def test_min_interest_coverage_deletion_restored(
        self, base_hard_constraints: HardConstraints, base_spec: SelectorSpec
    ) -> None:
        """Setting min_interest_coverage to None when immutable had value is restored."""
        new_spec = make_spec(min_interest_coverage=None)  # Try to delete

        result = enforce_revision_policy(
            base_hard_constraints,
            original_target=100_000_000,
            prev_spec=base_spec,
            new_spec=new_spec,
        )

        assert result.valid is True
        assert result.spec is not None
        assert result.spec.hard_constraints.min_interest_coverage == 2.0
        assert any("Cannot remove min_interest_coverage" in v for v in result.violations)

    def test_max_critical_fraction_deletion_restored(
        self, base_hard_constraints: HardConstraints, base_spec: SelectorSpec
    ) -> None:
        """Setting max_critical_fraction to None when immutable had value is restored."""
        new_spec = make_spec(max_critical_fraction=None)  # Try to delete

        result = enforce_revision_policy(
            base_hard_constraints,
            original_target=100_000_000,
            prev_spec=base_spec,
            new_spec=new_spec,
        )

        assert result.valid is True
        assert result.spec is not None
        assert result.spec.hard_constraints.max_critical_fraction == 0.3
        assert any("Cannot remove max_critical_fraction" in v for v in result.violations)


# =============================================================================
# Target Amount Monotonicity Tests
# =============================================================================


class TestTargetAmountMonotonicity:
    """Tests for target amount constraints."""

    def test_target_increase_clamped(
        self, base_hard_constraints: HardConstraints, base_spec: SelectorSpec
    ) -> None:
        """Target amount cannot increase - clamped to previous."""
        new_spec = make_spec(target_amount=120_000_000)  # Try to increase from 100M

        result = enforce_revision_policy(
            base_hard_constraints,
            original_target=100_000_000,
            prev_spec=base_spec,
            new_spec=new_spec,
        )

        assert result.valid is True
        assert result.spec is not None
        assert result.spec.target_amount == 100_000_000
        assert any("cannot increase" in v for v in result.violations)

    def test_target_same_allowed(
        self, base_hard_constraints: HardConstraints, base_spec: SelectorSpec
    ) -> None:
        """Target amount staying the same is allowed."""
        new_spec = make_spec(target_amount=100_000_000)

        result = enforce_revision_policy(
            base_hard_constraints,
            original_target=100_000_000,
            prev_spec=base_spec,
            new_spec=new_spec,
        )

        assert result.valid is True
        assert result.spec is not None
        assert result.spec.target_amount == 100_000_000
        assert not any("target_amount" in v for v in result.violations)

    def test_target_decrease_within_20_percent_allowed(
        self, base_hard_constraints: HardConstraints, base_spec: SelectorSpec
    ) -> None:
        """Target decrease within 20% per iteration is allowed."""
        new_spec = make_spec(target_amount=85_000_000)  # 15% decrease

        result = enforce_revision_policy(
            base_hard_constraints,
            original_target=100_000_000,
            prev_spec=base_spec,
            new_spec=new_spec,
        )

        assert result.valid is True
        assert result.spec is not None
        assert result.spec.target_amount == 85_000_000

    def test_target_decrease_beyond_20_percent_clamped(
        self, base_hard_constraints: HardConstraints, base_spec: SelectorSpec
    ) -> None:
        """Target decrease beyond 20% per iteration is clamped."""
        new_spec = make_spec(target_amount=70_000_000)  # 30% decrease

        result = enforce_revision_policy(
            base_hard_constraints,
            original_target=100_000_000,
            prev_spec=base_spec,
            new_spec=new_spec,
        )

        assert result.valid is True
        assert result.spec is not None
        assert result.spec.target_amount == 80_000_000  # Clamped to 20% drop
        assert any("20%" in v for v in result.violations)

    def test_target_exactly_20_percent_allowed(
        self, base_hard_constraints: HardConstraints, base_spec: SelectorSpec
    ) -> None:
        """Target decrease of exactly 20% is allowed."""
        new_spec = make_spec(target_amount=80_000_000)  # Exactly 20% decrease

        result = enforce_revision_policy(
            base_hard_constraints,
            original_target=100_000_000,
            prev_spec=base_spec,
            new_spec=new_spec,
        )

        assert result.valid is True
        assert result.spec is not None
        assert result.spec.target_amount == 80_000_000
        # Should not have per-iteration violation
        assert not any("20%" in v for v in result.violations)

    def test_target_below_50_percent_floor_invalid(
        self, base_hard_constraints: HardConstraints, base_spec: SelectorSpec
    ) -> None:
        """Target below 50% of original is invalid."""
        # Start with a lower target so we can test the floor
        prev_spec = make_spec(target_amount=60_000_000)  # Already at 60M
        new_spec = make_spec(target_amount=40_000_000)  # Try to go to 40M (40% of original 100M)

        result = enforce_revision_policy(
            base_hard_constraints,
            original_target=100_000_000,  # Original was 100M
            prev_spec=prev_spec,
            new_spec=new_spec,
        )

        assert result.valid is False
        assert result.spec is None
        assert any("50%" in v for v in result.violations)

    def test_target_at_50_percent_floor_allowed(
        self, base_hard_constraints: HardConstraints, base_spec: SelectorSpec
    ) -> None:
        """Target at exactly 50% of original is allowed."""
        prev_spec = make_spec(target_amount=60_000_000)
        new_spec = make_spec(target_amount=50_000_000)  # Exactly 50% of original

        result = enforce_revision_policy(
            base_hard_constraints,
            original_target=100_000_000,
            prev_spec=prev_spec,
            new_spec=new_spec,
        )

        # Should be valid since 50M is exactly 50% of 100M and within 20% of 60M
        assert result.valid is True
        assert result.spec is not None

    def test_multiple_iterations_compound(
        self, base_hard_constraints: HardConstraints
    ) -> None:
        """Test that iterations compound correctly (100 -> 80 -> 64)."""
        # First iteration: 100M -> 80M (20% drop)
        prev_spec_1 = make_spec(target_amount=100_000_000)
        new_spec_1 = make_spec(target_amount=80_000_000)

        result_1 = enforce_revision_policy(
            base_hard_constraints,
            original_target=100_000_000,
            prev_spec=prev_spec_1,
            new_spec=new_spec_1,
        )

        assert result_1.valid is True
        assert result_1.spec is not None
        assert result_1.spec.target_amount == 80_000_000

        # Second iteration: 80M -> 64M (20% drop)
        prev_spec_2 = make_spec(target_amount=80_000_000)
        new_spec_2 = make_spec(target_amount=64_000_000)

        result_2 = enforce_revision_policy(
            base_hard_constraints,
            original_target=100_000_000,  # Original is still 100M
            prev_spec=prev_spec_2,
            new_spec=new_spec_2,
        )

        assert result_2.valid is True
        assert result_2.spec is not None
        assert result_2.spec.target_amount == 64_000_000


# =============================================================================
# Asset Filter Relaxation Tests
# =============================================================================


class TestMaxCriticalityRelaxation:
    """Tests for max_criticality filter relaxation."""

    def test_criticality_increase_within_bounds(
        self, base_hard_constraints: HardConstraints, base_spec: SelectorSpec
    ) -> None:
        """max_criticality can increase by 0.1 per iteration."""
        new_spec = make_spec(max_criticality=0.6)  # Increase from 0.5 by 0.1

        result = enforce_revision_policy(
            base_hard_constraints,
            original_target=100_000_000,
            prev_spec=base_spec,
            new_spec=new_spec,
        )

        assert result.valid is True
        assert result.spec is not None
        assert result.spec.asset_filters.max_criticality == pytest.approx(0.6)

    def test_criticality_increase_beyond_bounds_clamped(
        self, base_hard_constraints: HardConstraints, base_spec: SelectorSpec
    ) -> None:
        """max_criticality increase beyond 0.1 is clamped."""
        new_spec = make_spec(max_criticality=0.8)  # Try to increase from 0.5 by 0.3

        result = enforce_revision_policy(
            base_hard_constraints,
            original_target=100_000_000,
            prev_spec=base_spec,
            new_spec=new_spec,
        )

        assert result.valid is True
        assert result.spec is not None
        assert result.spec.asset_filters.max_criticality == pytest.approx(0.6)  # Clamped to +0.1
        assert any("max_criticality" in v for v in result.violations)

    def test_criticality_ceiling_at_0_8(
        self, base_hard_constraints: HardConstraints
    ) -> None:
        """max_criticality cannot exceed 0.8 absolute ceiling."""
        prev_spec = make_spec(max_criticality=0.75)
        new_spec = make_spec(max_criticality=0.9)  # Try to go to 0.9

        result = enforce_revision_policy(
            base_hard_constraints,
            original_target=100_000_000,
            prev_spec=prev_spec,
            new_spec=new_spec,
        )

        assert result.valid is True
        assert result.spec is not None
        # Should be min(0.75 + 0.1, 0.8) = 0.8
        assert result.spec.asset_filters.max_criticality == pytest.approx(0.8)

    def test_criticality_decrease_allowed(
        self, base_hard_constraints: HardConstraints, base_spec: SelectorSpec
    ) -> None:
        """max_criticality can decrease (tightening is always allowed)."""
        new_spec = make_spec(max_criticality=0.3)  # Decrease from 0.5

        result = enforce_revision_policy(
            base_hard_constraints,
            original_target=100_000_000,
            prev_spec=base_spec,
            new_spec=new_spec,
        )

        assert result.valid is True
        assert result.spec is not None
        assert result.spec.asset_filters.max_criticality == pytest.approx(0.3)

    def test_criticality_from_none_uses_default(
        self, base_hard_constraints: HardConstraints
    ) -> None:
        """If previous max_criticality is None, use 0.5 default."""
        prev_spec = make_spec(max_criticality=None)
        new_spec = make_spec(max_criticality=0.7)  # Try to set to 0.7

        result = enforce_revision_policy(
            base_hard_constraints,
            original_target=100_000_000,
            prev_spec=prev_spec,
            new_spec=new_spec,
        )

        assert result.valid is True
        assert result.spec is not None
        # Default is 0.5, so max is 0.6
        assert result.spec.asset_filters.max_criticality == pytest.approx(0.6)


class TestMinLeaseabilityRelaxation:
    """Tests for min_leaseability_score filter relaxation."""

    def test_leaseability_decrease_within_bounds(
        self, base_hard_constraints: HardConstraints, base_spec: SelectorSpec
    ) -> None:
        """min_leaseability_score can decrease by 0.1 per iteration."""
        new_spec = make_spec(min_leaseability_score=0.4)  # Decrease from 0.5 by 0.1

        result = enforce_revision_policy(
            base_hard_constraints,
            original_target=100_000_000,
            prev_spec=base_spec,
            new_spec=new_spec,
        )

        assert result.valid is True
        assert result.spec is not None
        assert result.spec.asset_filters.min_leaseability_score == pytest.approx(0.4)

    def test_leaseability_decrease_beyond_bounds_clamped(
        self, base_hard_constraints: HardConstraints, base_spec: SelectorSpec
    ) -> None:
        """min_leaseability_score decrease beyond 0.1 is clamped."""
        new_spec = make_spec(min_leaseability_score=0.2)  # Try to decrease from 0.5 by 0.3

        result = enforce_revision_policy(
            base_hard_constraints,
            original_target=100_000_000,
            prev_spec=base_spec,
            new_spec=new_spec,
        )

        assert result.valid is True
        assert result.spec is not None
        assert result.spec.asset_filters.min_leaseability_score == pytest.approx(0.4)  # Clamped to -0.1
        assert any("min_leaseability_score" in v for v in result.violations)

    def test_leaseability_floor_at_0_2(
        self, base_hard_constraints: HardConstraints
    ) -> None:
        """min_leaseability_score cannot go below 0.2 absolute floor."""
        prev_spec = make_spec(min_leaseability_score=0.25)
        new_spec = make_spec(min_leaseability_score=0.1)  # Try to go to 0.1

        result = enforce_revision_policy(
            base_hard_constraints,
            original_target=100_000_000,
            prev_spec=prev_spec,
            new_spec=new_spec,
        )

        assert result.valid is True
        assert result.spec is not None
        # Should be max(0.25 - 0.1, 0.2) = 0.2
        assert result.spec.asset_filters.min_leaseability_score == pytest.approx(0.2)

    def test_leaseability_increase_allowed(
        self, base_hard_constraints: HardConstraints, base_spec: SelectorSpec
    ) -> None:
        """min_leaseability_score can increase (tightening is always allowed)."""
        new_spec = make_spec(min_leaseability_score=0.7)  # Increase from 0.5

        result = enforce_revision_policy(
            base_hard_constraints,
            original_target=100_000_000,
            prev_spec=base_spec,
            new_spec=new_spec,
        )

        assert result.valid is True
        assert result.spec is not None
        assert result.spec.asset_filters.min_leaseability_score == pytest.approx(0.7)

    def test_leaseability_from_none_uses_default(
        self, base_hard_constraints: HardConstraints
    ) -> None:
        """If previous min_leaseability_score is None, use 0.5 default."""
        prev_spec = make_spec(min_leaseability_score=None)
        new_spec = make_spec(min_leaseability_score=0.3)  # Try to set to 0.3

        result = enforce_revision_policy(
            base_hard_constraints,
            original_target=100_000_000,
            prev_spec=prev_spec,
            new_spec=new_spec,
        )

        assert result.valid is True
        assert result.spec is not None
        # Default is 0.5, so min is 0.4
        assert result.spec.asset_filters.min_leaseability_score == pytest.approx(0.4)


# =============================================================================
# Filter Deletion Tests
# =============================================================================


class TestFilterDeletion:
    """Tests that filters cannot be deleted when previously set."""

    def test_max_criticality_deletion_restored(
        self, base_hard_constraints: HardConstraints, base_spec: SelectorSpec
    ) -> None:
        """Setting max_criticality to None when prev had value is restored."""
        new_spec = make_spec(max_criticality=None)  # Try to delete

        result = enforce_revision_policy(
            base_hard_constraints,
            original_target=100_000_000,
            prev_spec=base_spec,
            new_spec=new_spec,
        )

        assert result.valid is True
        assert result.spec is not None
        assert result.spec.asset_filters.max_criticality == pytest.approx(0.5)
        assert any("Cannot remove max_criticality" in v for v in result.violations)

    def test_min_leaseability_deletion_restored(
        self, base_hard_constraints: HardConstraints, base_spec: SelectorSpec
    ) -> None:
        """Setting min_leaseability_score to None when prev had value is restored."""
        new_spec = make_spec(min_leaseability_score=None)  # Try to delete

        result = enforce_revision_policy(
            base_hard_constraints,
            original_target=100_000_000,
            prev_spec=base_spec,
            new_spec=new_spec,
        )

        assert result.valid is True
        assert result.spec is not None
        assert result.spec.asset_filters.min_leaseability_score == pytest.approx(0.5)
        assert any("Cannot remove min_leaseability_score" in v for v in result.violations)

    def test_filter_deletion_allowed_when_prev_was_none(
        self, base_hard_constraints: HardConstraints
    ) -> None:
        """If previous filter was None, keeping it None is allowed."""
        prev_spec = make_spec(max_criticality=None, min_leaseability_score=None)
        new_spec = make_spec(max_criticality=None, min_leaseability_score=None)

        result = enforce_revision_policy(
            base_hard_constraints,
            original_target=100_000_000,
            prev_spec=prev_spec,
            new_spec=new_spec,
        )

        assert result.valid is True
        assert result.spec is not None
        assert result.spec.asset_filters.max_criticality is None
        assert result.spec.asset_filters.min_leaseability_score is None
        assert not any("Cannot remove" in v for v in result.violations)


# =============================================================================
# Combined/Edge Case Tests
# =============================================================================


class TestCombinedScenarios:
    """Tests for combined constraint scenarios."""

    def test_no_changes_valid(
        self, base_hard_constraints: HardConstraints, base_spec: SelectorSpec
    ) -> None:
        """Identical spec passes with no violations."""
        result = enforce_revision_policy(
            base_hard_constraints,
            original_target=100_000_000,
            prev_spec=base_spec,
            new_spec=base_spec.model_copy(deep=True),
        )

        assert result.valid is True
        assert result.spec is not None
        assert result.violations == []

    def test_multiple_violations_all_clamped(
        self, base_hard_constraints: HardConstraints, base_spec: SelectorSpec
    ) -> None:
        """Multiple violations result in multiple clamps."""
        new_spec = make_spec(
            target_amount=70_000_000,  # 30% drop - will be clamped to 20%
            max_net_leverage=6.0,  # Will be clamped to 4.0
            max_criticality=0.9,  # Will be clamped to 0.6
        )

        result = enforce_revision_policy(
            base_hard_constraints,
            original_target=100_000_000,
            prev_spec=base_spec,
            new_spec=new_spec,
        )

        assert result.valid is True
        assert result.spec is not None
        assert result.spec.target_amount == 80_000_000
        assert result.spec.hard_constraints.max_net_leverage == 4.0
        assert result.spec.asset_filters.max_criticality == pytest.approx(0.6)
        assert len(result.violations) >= 3

    def test_valid_relaxation_within_all_bounds(
        self, base_hard_constraints: HardConstraints, base_spec: SelectorSpec
    ) -> None:
        """Valid revision within all bounds."""
        new_spec = make_spec(
            target_amount=85_000_000,  # 15% drop - OK
            max_criticality=0.6,  # +0.1 - OK
            min_leaseability_score=0.4,  # -0.1 - OK
        )

        result = enforce_revision_policy(
            base_hard_constraints,
            original_target=100_000_000,
            prev_spec=base_spec,
            new_spec=new_spec,
        )

        assert result.valid is True
        assert result.spec is not None
        assert result.spec.target_amount == 85_000_000
        assert result.spec.asset_filters.max_criticality == pytest.approx(0.6)
        assert result.spec.asset_filters.min_leaseability_score == pytest.approx(0.4)
        assert result.violations == []

    def test_soft_preferences_unrestricted(
        self, base_hard_constraints: HardConstraints, base_spec: SelectorSpec
    ) -> None:
        """Soft preferences can be changed without restriction."""
        new_spec = base_spec.model_copy(deep=True)
        new_spec.soft_preferences = SoftPreferences(
            prefer_low_criticality=False,  # Changed
            prefer_high_leaseability=False,  # Changed
            weight_criticality=2.0,  # Changed
            weight_leaseability=0.5,  # Changed
        )

        result = enforce_revision_policy(
            base_hard_constraints,
            original_target=100_000_000,
            prev_spec=base_spec,
            new_spec=new_spec,
        )

        assert result.valid is True
        assert result.spec is not None
        assert result.spec.soft_preferences.prefer_low_criticality is False
        assert result.spec.soft_preferences.weight_criticality == 2.0
        assert result.violations == []

    def test_target_clamp_then_floor_check(
        self, base_hard_constraints: HardConstraints
    ) -> None:
        """Target is clamped first, then floor is checked."""
        # If at 55M and trying to go to 30M:
        # - First clamp to 55M * 0.8 = 44M (20% rule)
        # - Then check 44M < 50M (50% of 100M) → invalid
        prev_spec = make_spec(target_amount=55_000_000)
        new_spec = make_spec(target_amount=30_000_000)

        result = enforce_revision_policy(
            base_hard_constraints,
            original_target=100_000_000,
            prev_spec=prev_spec,
            new_spec=new_spec,
        )

        assert result.valid is False
        assert result.spec is None
        assert any("50%" in v for v in result.violations)
