"""
Tests for Audit Trace integration in the orchestrator.

Tests that the orchestrator correctly builds and returns audit traces
with proper target_source, floor_fraction, and entry tracking.

Test categories:
- Audit trace presence in response
- Target source determination (user_override vs llm_extraction)
- Floor fraction correctness
- Entry tracking across iterations
- Last entry matches final outcome
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
    PolicyViolationCode,
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
def healthy_corporate_state() -> CorporateState:
    """Corporate state with healthy metrics."""
    return CorporateState(
        net_debt=200_000_000,
        ebitda=100_000_000,
        interest_expense=10_000_000,
        lease_expense=5_000_000,
    )


@pytest.fixture
def base_request(
    sample_assets: list[Asset], healthy_corporate_state: CorporateState
) -> ProgramRequest:
    """Base program request for testing (no overrides - LLM extraction)."""
    return ProgramRequest(
        assets=sample_assets,
        corporate_state=healthy_corporate_state,
        program_type=ProgramType.SLB,
        program_description="Raise $50M via SLB to reduce debt",
    )


@pytest.fixture
def override_request(
    sample_assets: list[Asset], healthy_corporate_state: CorporateState
) -> ProgramRequest:
    """Program request with explicit target override (user_override)."""
    return ProgramRequest(
        assets=sample_assets,
        corporate_state=healthy_corporate_state,
        program_type=ProgramType.SLB,
        program_description="Raise funds via SLB",
        target_amount_override=50_000_000,  # User specifies exact target
    )


def make_feasible_spec(target: float = 50_000_000) -> SelectorSpec:
    """Create a spec that should be feasible with sample assets."""
    return SelectorSpec(
        program_type=ProgramType.SLB,
        objective=Objective.BALANCED,
        target_amount=target,
        hard_constraints=HardConstraints(
            max_net_leverage=9.0,
            min_fixed_charge_coverage=1.0,
        ),
        soft_preferences=SoftPreferences(),
        asset_filters=AssetFilters(),
        max_iterations=3,
    )


def make_infeasible_spec(target: float = 500_000_000) -> SelectorSpec:
    """Create a spec that will be infeasible (target too high)."""
    return SelectorSpec(
        program_type=ProgramType.SLB,
        objective=Objective.BALANCED,
        target_amount=target,
        hard_constraints=HardConstraints(
            max_net_leverage=9.0,
            min_fixed_charge_coverage=1.0,
        ),
        soft_preferences=SoftPreferences(),
        asset_filters=AssetFilters(),
        max_iterations=3,
    )


# =============================================================================
# Audit Trace Presence Tests
# =============================================================================


class TestAuditTracePresence:
    """Tests that audit trace is always present in response."""

    def test_audit_trace_present_in_response(self, base_request: ProgramRequest) -> None:
        """Audit trace is always present (not None) in response."""
        mock = MockLLMClient(custom_spec=make_feasible_spec())

        response = run_program(base_request, mock)

        assert response.audit_trace is not None

    def test_audit_trace_has_entries(self, base_request: ProgramRequest) -> None:
        """Audit trace has at least one entry (the initial iteration)."""
        mock = MockLLMClient(custom_spec=make_feasible_spec())

        response = run_program(base_request, mock)

        assert len(response.audit_trace.entries) >= 1

    def test_audit_trace_has_timestamps(self, base_request: ProgramRequest) -> None:
        """Audit trace has started_at and completed_at timestamps."""
        mock = MockLLMClient(custom_spec=make_feasible_spec())

        response = run_program(base_request, mock)

        assert response.audit_trace.started_at is not None
        assert response.audit_trace.completed_at is not None
        # completed_at should be after or equal to started_at
        assert response.audit_trace.completed_at >= response.audit_trace.started_at


# =============================================================================
# Target Source Tests
# =============================================================================


class TestTargetSource:
    """Tests for target_source determination (user_override vs llm_extraction)."""

    def test_llm_extraction_when_no_override(self, base_request: ProgramRequest) -> None:
        """target_source is 'llm_extraction' when no override provided."""
        mock = MockLLMClient(custom_spec=make_feasible_spec())

        response = run_program(base_request, mock)

        assert response.audit_trace.target_source == "llm_extraction"

    def test_user_override_when_override_provided(
        self, override_request: ProgramRequest
    ) -> None:
        """target_source is 'user_override' when override provided."""
        mock = MockLLMClient(custom_spec=make_feasible_spec())

        response = run_program(override_request, mock)

        assert response.audit_trace.target_source == "user_override"


# =============================================================================
# Floor Fraction Tests
# =============================================================================


class TestFloorFraction:
    """Tests for floor_fraction correctness."""

    def test_llm_extraction_floor_is_75_percent(
        self, base_request: ProgramRequest
    ) -> None:
        """floor_fraction is 0.75 for llm_extraction (no override)."""
        mock = MockLLMClient(custom_spec=make_feasible_spec(target=100_000_000))

        response = run_program(base_request, mock)

        assert response.audit_trace.floor_fraction == 0.75
        # floor_target should be 75% of original_target
        expected_floor = response.audit_trace.original_target * 0.75
        assert response.audit_trace.floor_target == pytest.approx(expected_floor)

    def test_user_override_floor_is_100_percent(
        self, override_request: ProgramRequest
    ) -> None:
        """floor_fraction is 1.0 for user_override."""
        mock = MockLLMClient(custom_spec=make_feasible_spec())

        response = run_program(override_request, mock)

        assert response.audit_trace.floor_fraction == 1.0
        # floor_target should equal original_target
        assert response.audit_trace.floor_target == response.audit_trace.original_target

    def test_original_target_matches_spec(self, base_request: ProgramRequest) -> None:
        """original_target in audit trace matches the initial spec's target."""
        spec = make_feasible_spec(target=75_000_000)
        mock = MockLLMClient(custom_spec=spec)

        response = run_program(base_request, mock)

        assert response.audit_trace.original_target == 75_000_000


# =============================================================================
# Entry Tracking Tests
# =============================================================================


class TestEntryTracking:
    """Tests for audit trace entry tracking across iterations."""

    def test_single_iteration_has_one_entry(self, base_request: ProgramRequest) -> None:
        """Successful first attempt has exactly one entry."""
        mock = MockLLMClient(custom_spec=make_feasible_spec())

        response = run_program(base_request, mock)

        assert response.outcome.status == SelectionStatus.OK
        assert len(response.audit_trace.entries) == 1

    def test_first_entry_is_initial_phase(self, base_request: ProgramRequest) -> None:
        """First entry has phase='initial' and iteration=0."""
        mock = MockLLMClient(custom_spec=make_feasible_spec())

        response = run_program(base_request, mock)

        first_entry = response.audit_trace.entries[0]
        assert first_entry.phase == "initial"
        assert first_entry.iteration == 0

    def test_initial_entry_has_no_target_before(
        self, base_request: ProgramRequest
    ) -> None:
        """First entry has target_before=None."""
        mock = MockLLMClient(custom_spec=make_feasible_spec())

        response = run_program(base_request, mock)

        first_entry = response.audit_trace.entries[0]
        assert first_entry.target_before is None

    def test_revision_entries_have_revision_phase(
        self, base_request: ProgramRequest
    ) -> None:
        """Revision entries have phase='revision'."""
        # Use infeasible spec to trigger revisions
        infeasible_spec = make_infeasible_spec(target=400_000_000)
        mock = MockLLMClient(
            custom_spec=infeasible_spec,
            revision_target_reduction=0.15,
        )

        response = run_program(base_request, mock)

        # Should have at least one revision entry
        if len(response.audit_trace.entries) > 1:
            for entry in response.audit_trace.entries[1:]:
                assert entry.phase == "revision"
                assert entry.iteration > 0

    def test_revision_entries_have_target_before(
        self, base_request: ProgramRequest
    ) -> None:
        """Revision entries have target_before set from previous iteration."""
        infeasible_spec = make_infeasible_spec(target=400_000_000)
        mock = MockLLMClient(
            custom_spec=infeasible_spec,
            revision_target_reduction=0.15,
        )

        response = run_program(base_request, mock)

        # If we have revision entries, check target_before
        if len(response.audit_trace.entries) > 1:
            for i, entry in enumerate(response.audit_trace.entries[1:], start=1):
                # target_before should match previous entry's target_after
                prev_entry = response.audit_trace.entries[i - 1]
                assert entry.target_before == prev_entry.target_after

    def test_entry_spec_snapshot_uses_factory(
        self, base_request: ProgramRequest
    ) -> None:
        """Entry spec_snapshot is created via from_spec factory."""
        mock = MockLLMClient(custom_spec=make_feasible_spec(target=60_000_000))

        response = run_program(base_request, mock)

        entry = response.audit_trace.entries[0]
        # Verify spec snapshot fields match spec
        assert entry.spec_snapshot.target_amount == response.selector_spec.target_amount

    def test_entry_outcome_snapshot_uses_factory(
        self, base_request: ProgramRequest
    ) -> None:
        """Entry outcome_snapshot is created via from_outcome factory."""
        mock = MockLLMClient(custom_spec=make_feasible_spec())

        response = run_program(base_request, mock)

        # Last entry's outcome should match response outcome
        last_entry = response.audit_trace.entries[-1]
        assert last_entry.outcome_snapshot.status == response.outcome.status
        assert last_entry.outcome_snapshot.proceeds == response.outcome.proceeds


# =============================================================================
# Last Entry Matches Outcome Tests
# =============================================================================


class TestLastEntryMatchesOutcome:
    """Tests that the last audit trace entry matches the final response outcome."""

    def test_last_entry_status_matches_outcome(
        self, base_request: ProgramRequest
    ) -> None:
        """Last entry's outcome_snapshot.status matches response.outcome.status."""
        mock = MockLLMClient(custom_spec=make_feasible_spec())

        response = run_program(base_request, mock)

        last_entry = response.audit_trace.entries[-1]
        assert last_entry.outcome_snapshot.status == response.outcome.status

    def test_last_entry_proceeds_matches_outcome(
        self, base_request: ProgramRequest
    ) -> None:
        """Last entry's outcome_snapshot.proceeds matches response.outcome.proceeds."""
        mock = MockLLMClient(custom_spec=make_feasible_spec())

        response = run_program(base_request, mock)

        last_entry = response.audit_trace.entries[-1]
        assert last_entry.outcome_snapshot.proceeds == response.outcome.proceeds

    def test_last_entry_violations_match_outcome(
        self, base_request: ProgramRequest
    ) -> None:
        """Last entry's outcome_snapshot.violations matches response.outcome.violations."""
        # Use infeasible spec to get violations
        spec = make_infeasible_spec(target=10_000_000_000)
        spec = SelectorSpec(
            program_type=spec.program_type,
            objective=spec.objective,
            target_amount=spec.target_amount,
            hard_constraints=spec.hard_constraints,
            soft_preferences=spec.soft_preferences,
            asset_filters=spec.asset_filters,
            max_iterations=1,  # No revisions
        )
        mock = MockLLMClient(custom_spec=spec)

        response = run_program(base_request, mock)

        last_entry = response.audit_trace.entries[-1]
        assert last_entry.outcome_snapshot.violations == response.outcome.violations

    def test_infeasible_outcome_recorded_correctly(
        self, base_request: ProgramRequest
    ) -> None:
        """Infeasible outcome is correctly recorded in last entry."""
        # Very high target that can't be met
        spec = SelectorSpec(
            program_type=ProgramType.SLB,
            objective=Objective.BALANCED,
            target_amount=10_000_000_000,  # $10B - impossible
            hard_constraints=HardConstraints(
                max_net_leverage=9.0,
                min_fixed_charge_coverage=1.0,
            ),
            soft_preferences=SoftPreferences(),
            asset_filters=AssetFilters(),
            max_iterations=1,  # No revisions
        )
        mock = MockLLMClient(custom_spec=spec)

        response = run_program(base_request, mock)

        assert response.outcome.status == SelectionStatus.INFEASIBLE
        last_entry = response.audit_trace.entries[-1]
        assert last_entry.outcome_snapshot.status == SelectionStatus.INFEASIBLE


# =============================================================================
# Integration Tests
# =============================================================================


class TestAuditTraceIntegration:
    """Integration tests for audit trace with full orchestration flow."""

    def test_full_flow_with_revisions_tracks_all_entries(
        self, base_request: ProgramRequest
    ) -> None:
        """Full revision flow correctly tracks all iterations."""
        # Start with infeasible target, revisions should reduce it
        infeasible_spec = make_infeasible_spec(target=400_000_000)
        mock = MockLLMClient(
            custom_spec=infeasible_spec,
            revision_target_reduction=0.15,  # 15% reduction per revision
        )

        response = run_program(base_request, mock)

        # Should have multiple entries (initial + revisions)
        assert len(response.audit_trace.entries) >= 1

        # All entries should have valid timestamps
        for entry in response.audit_trace.entries:
            assert entry.timestamp is not None

        # Iteration numbers should be sequential
        for i, entry in enumerate(response.audit_trace.entries):
            assert entry.iteration == i

    def test_audit_trace_serializes_correctly(
        self, base_request: ProgramRequest
    ) -> None:
        """Audit trace can be serialized to JSON."""
        mock = MockLLMClient(custom_spec=make_feasible_spec())

        response = run_program(base_request, mock)

        # Should be able to convert to dict (for JSON serialization)
        audit_dict = response.audit_trace.model_dump()
        assert "entries" in audit_dict
        assert "original_target" in audit_dict
        assert "floor_fraction" in audit_dict
        assert "target_source" in audit_dict
