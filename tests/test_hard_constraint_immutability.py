"""
Tests for Hard Constraint Immutability in the orchestrator.

Tests that hard constraints captured from the initial spec remain locked
across all revision iterations. The LLM cannot relax these constraints.

Test categories:
- Individual constraint immutability (max_net_leverage, min_interest_coverage, etc.)
- All four constraints together
- Multi-iteration scenarios
- Policy enforcement preserves immutability
"""

import pytest

from app.llm.mock import MockLLMClient
from app.models import (
    Asset,
    AssetFilters,
    AssetType,
    CorporateState,
    HardConstraints,
    Objective,
    ProgramRequest,
    ProgramType,
    SelectionStatus,
    SelectorSpec,
    SoftPreferences,
)
from app.orchestrator import run_program


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_assets() -> list[Asset]:
    """Sample asset portfolio for testing."""
    return [
        Asset(
            asset_id="A001",
            asset_type=AssetType.STORE,
            market="Dallas, TX",
            noi=5_000_000,
            book_value=50_000_000,
            criticality=0.3,
            leaseability_score=0.8,
        ),
        Asset(
            asset_id="A002",
            asset_type=AssetType.DISTRIBUTION_CENTER,
            market="Chicago, IL",
            noi=8_000_000,
            book_value=80_000_000,
            criticality=0.4,
            leaseability_score=0.7,
        ),
        Asset(
            asset_id="A003",
            asset_type=AssetType.STORE,
            market="Houston, TX",
            noi=3_000_000,
            book_value=30_000_000,
            criticality=0.2,
            leaseability_score=0.9,
        ),
    ]


@pytest.fixture
def corporate_state() -> CorporateState:
    """Corporate state for testing."""
    return CorporateState(
        net_debt=200_000_000,
        ebitda=100_000_000,
        interest_expense=10_000_000,
        lease_expense=5_000_000,
    )


@pytest.fixture
def base_request(
    sample_assets: list[Asset], corporate_state: CorporateState
) -> ProgramRequest:
    """Base program request for testing."""
    return ProgramRequest(
        assets=sample_assets,
        corporate_state=corporate_state,
        program_type=ProgramType.SLB,
        program_description="Raise $50M via SLB to reduce debt",
    )


def make_spec_with_constraints(
    max_leverage: float = 5.0,
    min_interest_coverage: float = 3.0,
    min_fixed_charge_coverage: float = 2.0,
    max_critical_fraction: float = 0.3,
    target: float = 50_000_000,
    max_iterations: int = 3,
) -> SelectorSpec:
    """Create a spec with all four hard constraints set."""
    return SelectorSpec(
        program_type=ProgramType.SLB,
        objective=Objective.BALANCED,
        target_amount=target,
        hard_constraints=HardConstraints(
            max_net_leverage=max_leverage,
            min_interest_coverage=min_interest_coverage,
            min_fixed_charge_coverage=min_fixed_charge_coverage,
            max_critical_fraction=max_critical_fraction,
        ),
        soft_preferences=SoftPreferences(),
        asset_filters=AssetFilters(),
        max_iterations=max_iterations,
    )


# =============================================================================
# Individual Constraint Immutability Tests
# =============================================================================


class TestLeverageStaysLocked:
    """Tests that max_net_leverage stays locked across iterations."""

    def test_leverage_stays_locked_across_iterations(
        self, base_request: ProgramRequest
    ) -> None:
        """max_net_leverage from initial spec is preserved through revisions."""
        initial_leverage = 5.0
        spec = make_spec_with_constraints(
            max_leverage=initial_leverage,
            target=500_000_000,  # High target to trigger revisions
            max_iterations=3,
        )
        mock = MockLLMClient(
            custom_spec=spec,
            revision_target_reduction=0.2,
        )

        response = run_program(base_request, mock)

        # Final spec should preserve the initial leverage constraint
        assert response.selector_spec.hard_constraints.max_net_leverage == initial_leverage

    def test_leverage_preserved_even_with_infeasible_outcome(
        self, base_request: ProgramRequest
    ) -> None:
        """max_net_leverage is preserved even when outcome is infeasible."""
        initial_leverage = 3.0  # Tight constraint
        spec = SelectorSpec(
            program_type=ProgramType.SLB,
            objective=Objective.BALANCED,
            target_amount=500_000_000,  # Impossible target
            hard_constraints=HardConstraints(
                max_net_leverage=initial_leverage,
                min_fixed_charge_coverage=1.0,
            ),
            soft_preferences=SoftPreferences(),
            asset_filters=AssetFilters(),
            max_iterations=1,  # No revisions
        )
        mock = MockLLMClient(custom_spec=spec)

        response = run_program(base_request, mock)

        # Even infeasible outcome should preserve constraint
        assert response.selector_spec.hard_constraints.max_net_leverage == initial_leverage
        assert response.outcome.status == SelectionStatus.INFEASIBLE


class TestInterestCoverageStaysLocked:
    """Tests that min_interest_coverage stays locked across iterations."""

    def test_interest_coverage_stays_locked(
        self, base_request: ProgramRequest
    ) -> None:
        """min_interest_coverage from initial spec is preserved through revisions."""
        initial_coverage = 4.0
        spec = make_spec_with_constraints(
            min_interest_coverage=initial_coverage,
            target=500_000_000,
            max_iterations=3,
        )
        mock = MockLLMClient(
            custom_spec=spec,
            revision_target_reduction=0.2,
        )

        response = run_program(base_request, mock)

        assert (
            response.selector_spec.hard_constraints.min_interest_coverage
            == initial_coverage
        )


class TestFixedChargeCoverageStaysLocked:
    """Tests that min_fixed_charge_coverage stays locked across iterations."""

    def test_fixed_charge_coverage_stays_locked(
        self, base_request: ProgramRequest
    ) -> None:
        """min_fixed_charge_coverage from initial spec is preserved through revisions."""
        initial_fcc = 2.5
        spec = make_spec_with_constraints(
            min_fixed_charge_coverage=initial_fcc,
            target=500_000_000,
            max_iterations=3,
        )
        mock = MockLLMClient(
            custom_spec=spec,
            revision_target_reduction=0.2,
        )

        response = run_program(base_request, mock)

        assert (
            response.selector_spec.hard_constraints.min_fixed_charge_coverage
            == initial_fcc
        )


class TestCriticalFractionStaysLocked:
    """Tests that max_critical_fraction stays locked across iterations."""

    def test_critical_fraction_stays_locked(
        self, base_request: ProgramRequest
    ) -> None:
        """max_critical_fraction from initial spec is preserved through revisions."""
        initial_critical = 0.25
        spec = make_spec_with_constraints(
            max_critical_fraction=initial_critical,
            target=500_000_000,
            max_iterations=3,
        )
        mock = MockLLMClient(
            custom_spec=spec,
            revision_target_reduction=0.2,
        )

        response = run_program(base_request, mock)

        assert (
            response.selector_spec.hard_constraints.max_critical_fraction
            == initial_critical
        )


# =============================================================================
# All Constraints Together Tests
# =============================================================================


class TestAllFourConstraintsImmutable:
    """Tests that all four hard constraints stay locked together."""

    def test_all_four_constraints_immutable(
        self, base_request: ProgramRequest
    ) -> None:
        """All four hard constraints remain locked across multiple iterations."""
        initial_leverage = 4.5
        initial_interest = 3.5
        initial_fcc = 2.0
        initial_critical = 0.35

        spec = make_spec_with_constraints(
            max_leverage=initial_leverage,
            min_interest_coverage=initial_interest,
            min_fixed_charge_coverage=initial_fcc,
            max_critical_fraction=initial_critical,
            target=500_000_000,
            max_iterations=5,
        )
        mock = MockLLMClient(
            custom_spec=spec,
            revision_target_reduction=0.15,
        )

        response = run_program(base_request, mock)

        hc = response.selector_spec.hard_constraints
        assert hc.max_net_leverage == initial_leverage
        assert hc.min_interest_coverage == initial_interest
        assert hc.min_fixed_charge_coverage == initial_fcc
        assert hc.max_critical_fraction == initial_critical

    def test_constraints_immutable_with_successful_outcome(
        self, base_request: ProgramRequest
    ) -> None:
        """Constraints remain immutable even with successful (OK) outcome."""
        initial_leverage = 6.0
        initial_fcc = 1.5

        spec = SelectorSpec(
            program_type=ProgramType.SLB,
            objective=Objective.BALANCED,
            target_amount=50_000_000,  # Feasible target
            hard_constraints=HardConstraints(
                max_net_leverage=initial_leverage,
                min_fixed_charge_coverage=initial_fcc,
            ),
            soft_preferences=SoftPreferences(),
            asset_filters=AssetFilters(),
            max_iterations=3,
        )
        mock = MockLLMClient(custom_spec=spec)

        response = run_program(base_request, mock)

        assert response.outcome.status == SelectionStatus.OK
        assert response.selector_spec.hard_constraints.max_net_leverage == initial_leverage
        assert (
            response.selector_spec.hard_constraints.min_fixed_charge_coverage
            == initial_fcc
        )


# =============================================================================
# Multi-Iteration Tests
# =============================================================================


class TestMultiIterationImmutability:
    """Tests for constraint immutability across multiple iterations."""

    def test_constraints_preserved_through_multiple_revisions(
        self, base_request: ProgramRequest
    ) -> None:
        """Hard constraints stay the same through 3+ revision iterations."""
        initial_leverage = 5.5
        initial_fcc = 2.2

        spec = SelectorSpec(
            program_type=ProgramType.SLB,
            objective=Objective.BALANCED,
            target_amount=800_000_000,  # Very high target to force max revisions
            hard_constraints=HardConstraints(
                max_net_leverage=initial_leverage,
                min_fixed_charge_coverage=initial_fcc,
            ),
            soft_preferences=SoftPreferences(),
            asset_filters=AssetFilters(),
            max_iterations=5,
        )
        mock = MockLLMClient(
            custom_spec=spec,
            revision_target_reduction=0.1,  # Small reduction to ensure multiple iterations
        )

        response = run_program(base_request, mock)

        # Verify constraints unchanged regardless of number of iterations
        assert response.selector_spec.hard_constraints.max_net_leverage == initial_leverage
        assert (
            response.selector_spec.hard_constraints.min_fixed_charge_coverage
            == initial_fcc
        )

    def test_audit_trace_shows_constraints_unchanged(
        self, base_request: ProgramRequest
    ) -> None:
        """Audit trace entries all show same hard constraints."""
        initial_leverage = 4.0
        initial_fcc = 1.8

        spec = SelectorSpec(
            program_type=ProgramType.SLB,
            objective=Objective.BALANCED,
            target_amount=400_000_000,
            hard_constraints=HardConstraints(
                max_net_leverage=initial_leverage,
                min_fixed_charge_coverage=initial_fcc,
            ),
            soft_preferences=SoftPreferences(),
            asset_filters=AssetFilters(),
            max_iterations=4,
        )
        mock = MockLLMClient(
            custom_spec=spec,
            revision_target_reduction=0.15,
        )

        response = run_program(base_request, mock)

        # Every audit trace entry should show the same constraints
        for entry in response.audit_trace.entries:
            snapshot = entry.spec_snapshot
            assert snapshot.max_net_leverage == initial_leverage
            assert snapshot.min_fixed_charge_coverage == initial_fcc


# =============================================================================
# Policy Enforcement Tests
# =============================================================================


class TestPolicyEnforcesImmutability:
    """Tests that revision policy enforces hard constraint immutability."""

    def test_policy_blocks_leverage_relaxation(
        self, base_request: ProgramRequest
    ) -> None:
        """Revision policy blocks attempts to relax max_net_leverage."""
        # This test verifies the orchestrator correctly uses enforce_revision_policy
        # to block any LLM attempt to relax constraints
        initial_leverage = 4.0

        spec = SelectorSpec(
            program_type=ProgramType.SLB,
            objective=Objective.BALANCED,
            target_amount=300_000_000,
            hard_constraints=HardConstraints(
                max_net_leverage=initial_leverage,
                min_fixed_charge_coverage=1.5,
            ),
            soft_preferences=SoftPreferences(),
            asset_filters=AssetFilters(),
            max_iterations=3,
        )
        mock = MockLLMClient(
            custom_spec=spec,
            revision_target_reduction=0.2,
        )

        response = run_program(base_request, mock)

        # Final spec must have original leverage (policy enforcement)
        assert response.selector_spec.hard_constraints.max_net_leverage == initial_leverage

    def test_policy_allows_target_reduction_while_preserving_constraints(
        self, base_request: ProgramRequest
    ) -> None:
        """Policy allows target amount reduction while keeping constraints locked."""
        initial_leverage = 5.0
        initial_fcc = 2.0
        initial_target = 300_000_000

        spec = SelectorSpec(
            program_type=ProgramType.SLB,
            objective=Objective.BALANCED,
            target_amount=initial_target,
            hard_constraints=HardConstraints(
                max_net_leverage=initial_leverage,
                min_fixed_charge_coverage=initial_fcc,
            ),
            soft_preferences=SoftPreferences(),
            asset_filters=AssetFilters(),
            max_iterations=3,
        )
        mock = MockLLMClient(
            custom_spec=spec,
            revision_target_reduction=0.2,
        )

        response = run_program(base_request, mock)

        # Constraints unchanged
        assert response.selector_spec.hard_constraints.max_net_leverage == initial_leverage
        assert (
            response.selector_spec.hard_constraints.min_fixed_charge_coverage
            == initial_fcc
        )

        # Target may have been reduced (revisions allowed)
        # But constraints are always preserved


# =============================================================================
# Edge Cases
# =============================================================================


class TestConstraintImmutabilityEdgeCases:
    """Edge cases for hard constraint immutability."""

    def test_none_constraints_stay_none(
        self, base_request: ProgramRequest
    ) -> None:
        """Constraints that start as None remain None (not filled in by LLM)."""
        spec = SelectorSpec(
            program_type=ProgramType.SLB,
            objective=Objective.BALANCED,
            target_amount=50_000_000,
            hard_constraints=HardConstraints(
                max_net_leverage=5.0,
                # min_interest_coverage=None (not set)
                min_fixed_charge_coverage=1.5,
                # max_critical_fraction=None (not set)
            ),
            soft_preferences=SoftPreferences(),
            asset_filters=AssetFilters(),
            max_iterations=2,
        )
        mock = MockLLMClient(custom_spec=spec)

        response = run_program(base_request, mock)

        # None constraints should stay None
        assert response.selector_spec.hard_constraints.min_interest_coverage is None
        assert response.selector_spec.hard_constraints.max_critical_fraction is None

    def test_single_iteration_preserves_constraints(
        self, base_request: ProgramRequest
    ) -> None:
        """Even with max_iterations=1, constraints are preserved."""
        initial_leverage = 3.5

        spec = SelectorSpec(
            program_type=ProgramType.SLB,
            objective=Objective.BALANCED,
            target_amount=500_000_000,  # Will be infeasible
            hard_constraints=HardConstraints(
                max_net_leverage=initial_leverage,
                min_fixed_charge_coverage=1.0,
            ),
            soft_preferences=SoftPreferences(),
            asset_filters=AssetFilters(),
            max_iterations=1,  # Single iteration only
        )
        mock = MockLLMClient(custom_spec=spec)

        response = run_program(base_request, mock)

        assert response.selector_spec.hard_constraints.max_net_leverage == initial_leverage
