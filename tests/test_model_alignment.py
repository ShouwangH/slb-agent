"""
Tests for audit trace model alignment.

Ensures that derived models (SpecSnapshot, OutcomeSnapshot) stay in sync
with their source models (SelectorSpec, ProgramOutcome). These tests
prevent field drift between related types.
"""

import pytest
from pydantic import ValidationError

from app.models import (
    AssetFilters,
    AuditTrace,
    AuditTraceEntry,
    ConstraintViolation,
    HardConstraints,
    Objective,
    OutcomeSnapshot,
    PolicyViolation,
    PolicyViolationCode,
    ProgramOutcome,
    ProgramType,
    SelectionStatus,
    SelectorSpec,
    SoftPreferences,
    SpecSnapshot,
)


# =============================================================================
# PolicyViolationCode Tests
# =============================================================================


class TestPolicyViolationCode:
    """Test PolicyViolationCode enum values."""

    def test_target_violation_codes(self):
        """Target violation codes should exist."""
        assert PolicyViolationCode.TARGET_INCREASED.value == "target_increased"
        assert PolicyViolationCode.TARGET_DROP_EXCEEDED.value == "target_drop_exceeded"
        assert PolicyViolationCode.TARGET_BELOW_FLOOR.value == "target_below_floor"

    def test_hard_constraint_violation_codes(self):
        """Hard constraint violation codes should exist."""
        assert PolicyViolationCode.LEVERAGE_RELAXED.value == "leverage_relaxed"
        assert PolicyViolationCode.INTEREST_COVERAGE_RELAXED.value == "interest_coverage_relaxed"
        assert PolicyViolationCode.FIXED_CHARGE_COVERAGE_RELAXED.value == "fixed_charge_coverage_relaxed"
        assert PolicyViolationCode.CRITICAL_FRACTION_RELAXED.value == "critical_fraction_relaxed"
        assert PolicyViolationCode.CONSTRAINT_DELETED.value == "constraint_deleted"

    def test_filter_violation_codes(self):
        """Filter violation codes should exist."""
        assert PolicyViolationCode.CRITICALITY_STEP_EXCEEDED.value == "criticality_step_exceeded"
        assert PolicyViolationCode.LEASEABILITY_STEP_EXCEEDED.value == "leaseability_step_exceeded"
        assert PolicyViolationCode.FILTER_DELETED.value == "filter_deleted"

    def test_immutable_field_violation_codes(self):
        """Immutable field violation codes should exist."""
        assert PolicyViolationCode.PROGRAM_TYPE_CHANGED.value == "program_type_changed"


# =============================================================================
# PolicyViolation Tests
# =============================================================================


class TestPolicyViolation:
    """Test PolicyViolation model."""

    def test_valid_violation_with_all_fields(self):
        """Valid violation with all fields should pass."""
        v = PolicyViolation(
            code=PolicyViolationCode.TARGET_INCREASED,
            detail="target_amount cannot increase",
            field="target_amount",
            attempted=60_000_000,
            limit=50_000_000,
            adjusted_to=50_000_000,
        )
        assert v.code == PolicyViolationCode.TARGET_INCREASED
        assert v.field == "target_amount"
        assert v.adjusted_to == 50_000_000

    def test_violation_with_optional_fields_none(self):
        """Violation with optional fields as None should pass."""
        v = PolicyViolation(
            code=PolicyViolationCode.PROGRAM_TYPE_CHANGED,
            detail="Cannot change program_type",
            field="program_type",
            # attempted, limit, adjusted_to all None
        )
        assert v.attempted is None
        assert v.limit is None
        assert v.adjusted_to is None

    def test_rejection_violation(self):
        """Rejection (invalid) violation has adjusted_to=None."""
        v = PolicyViolation(
            code=PolicyViolationCode.TARGET_BELOW_FLOOR,
            detail="target_amount below 75% floor",
            field="target_amount",
            attempted=30_000_000,
            limit=37_500_000,
            adjusted_to=None,  # Could not adjust - invalid
        )
        assert v.adjusted_to is None

    def test_violation_serialization(self):
        """Violation should serialize correctly."""
        v = PolicyViolation(
            code=PolicyViolationCode.LEVERAGE_RELAXED,
            detail="Cannot increase max_net_leverage beyond 4.0x",
            field="max_net_leverage",
            attempted=5.0,
            limit=4.0,
            adjusted_to=4.0,
        )
        data = v.model_dump()
        assert data["code"] == "leverage_relaxed"
        assert data["field"] == "max_net_leverage"


# =============================================================================
# SpecSnapshot Tests
# =============================================================================


class TestSpecSnapshot:
    """Test SpecSnapshot model and factory method."""

    @pytest.fixture
    def sample_spec(self):
        """Create a sample SelectorSpec for testing."""
        return SelectorSpec(
            program_type=ProgramType.SLB,
            objective=Objective.BALANCED,
            target_amount=50_000_000,
            hard_constraints=HardConstraints(
                max_net_leverage=4.0,
                min_interest_coverage=2.0,
                min_fixed_charge_coverage=3.0,
                max_critical_fraction=0.3,
            ),
            soft_preferences=SoftPreferences(),
            asset_filters=AssetFilters(
                max_criticality=0.5,
                min_leaseability_score=0.6,
            ),
        )

    def test_from_spec_copies_target_amount(self, sample_spec):
        """from_spec should copy target_amount."""
        snapshot = SpecSnapshot.from_spec(sample_spec)
        assert snapshot.target_amount == sample_spec.target_amount

    def test_from_spec_copies_asset_filters(self, sample_spec):
        """from_spec should copy asset filter values."""
        snapshot = SpecSnapshot.from_spec(sample_spec)
        assert snapshot.max_criticality == sample_spec.asset_filters.max_criticality
        assert snapshot.min_leaseability_score == sample_spec.asset_filters.min_leaseability_score

    def test_from_spec_copies_all_hard_constraints(self, sample_spec):
        """from_spec should copy all 4 hard constraints."""
        snapshot = SpecSnapshot.from_spec(sample_spec)
        assert snapshot.max_net_leverage == sample_spec.hard_constraints.max_net_leverage
        assert snapshot.min_interest_coverage == sample_spec.hard_constraints.min_interest_coverage
        assert snapshot.min_fixed_charge_coverage == sample_spec.hard_constraints.min_fixed_charge_coverage
        assert snapshot.max_critical_fraction == sample_spec.hard_constraints.max_critical_fraction

    def test_from_spec_handles_none_values(self):
        """from_spec should handle None values in source."""
        spec = SelectorSpec(
            program_type=ProgramType.SLB,
            objective=Objective.BALANCED,
            target_amount=50_000_000,
            hard_constraints=HardConstraints(),  # All None
            soft_preferences=SoftPreferences(),
            asset_filters=AssetFilters(),  # All None
        )
        snapshot = SpecSnapshot.from_spec(spec)

        assert snapshot.max_criticality is None
        assert snapshot.min_leaseability_score is None
        assert snapshot.max_net_leverage is None
        assert snapshot.min_interest_coverage is None
        assert snapshot.min_fixed_charge_coverage is None
        assert snapshot.max_critical_fraction is None

    def test_snapshot_serialization(self, sample_spec):
        """Snapshot should serialize correctly."""
        snapshot = SpecSnapshot.from_spec(sample_spec)
        data = snapshot.model_dump()

        assert data["target_amount"] == 50_000_000
        assert data["max_net_leverage"] == 4.0
        assert data["min_interest_coverage"] == 2.0


# =============================================================================
# OutcomeSnapshot Tests
# =============================================================================


class TestOutcomeSnapshot:
    """Test OutcomeSnapshot model and factory method."""

    @pytest.fixture
    def sample_outcome(self):
        """Create a sample ProgramOutcome for testing."""
        return ProgramOutcome(
            status=SelectionStatus.OK,
            selected_assets=[],
            proceeds=45_000_000,
            leverage_before=4.0,
            leverage_after=3.5,
            interest_coverage_before=5.0,
            interest_coverage_after=5.5,
            fixed_charge_coverage_before=3.0,
            fixed_charge_coverage_after=2.8,
            critical_fraction=0.15,
            violations=[],
            warnings=[],
        )

    def test_from_outcome_copies_status(self, sample_outcome):
        """from_outcome should copy status."""
        snapshot = OutcomeSnapshot.from_outcome(sample_outcome)
        assert snapshot.status == sample_outcome.status

    def test_from_outcome_copies_proceeds(self, sample_outcome):
        """from_outcome should copy proceeds."""
        snapshot = OutcomeSnapshot.from_outcome(sample_outcome)
        assert snapshot.proceeds == sample_outcome.proceeds

    def test_from_outcome_copies_post_metrics(self, sample_outcome):
        """from_outcome should copy post-transaction metrics."""
        snapshot = OutcomeSnapshot.from_outcome(sample_outcome)
        assert snapshot.leverage_after == sample_outcome.leverage_after
        assert snapshot.interest_coverage_after == sample_outcome.interest_coverage_after
        assert snapshot.fixed_charge_coverage_after == sample_outcome.fixed_charge_coverage_after

    def test_from_outcome_copies_critical_fraction(self, sample_outcome):
        """from_outcome should copy critical_fraction."""
        snapshot = OutcomeSnapshot.from_outcome(sample_outcome)
        assert snapshot.critical_fraction == sample_outcome.critical_fraction

    def test_from_outcome_copies_violations(self):
        """from_outcome should copy violations list."""
        violation = ConstraintViolation(
            code="MAX_NET_LEVERAGE",
            detail="Leverage 4.5x exceeds limit",
            actual=4.5,
            limit=4.0,
        )
        outcome = ProgramOutcome(
            status=SelectionStatus.INFEASIBLE,
            proceeds=0,
            violations=[violation],
        )
        snapshot = OutcomeSnapshot.from_outcome(outcome)

        assert len(snapshot.violations) == 1
        assert snapshot.violations[0].code == "MAX_NET_LEVERAGE"

    def test_from_outcome_handles_none_metrics(self):
        """from_outcome should handle None metric values."""
        outcome = ProgramOutcome(
            status=SelectionStatus.OK,
            proceeds=1_000_000,
            leverage_after=None,  # e.g., ebitda ≈ 0
            interest_coverage_after=None,
            fixed_charge_coverage_after=None,
        )
        snapshot = OutcomeSnapshot.from_outcome(outcome)

        assert snapshot.leverage_after is None
        assert snapshot.interest_coverage_after is None
        assert snapshot.fixed_charge_coverage_after is None

    def test_snapshot_serialization(self, sample_outcome):
        """Snapshot should serialize correctly."""
        snapshot = OutcomeSnapshot.from_outcome(sample_outcome)
        data = snapshot.model_dump()

        assert data["status"] == "ok"
        assert data["proceeds"] == 45_000_000
        assert data["leverage_after"] == 3.5


# =============================================================================
# Field Alignment Tests
# =============================================================================


class TestFieldAlignment:
    """Ensure snapshot fields are subsets of source model fields."""

    def test_outcome_snapshot_fields_exist_in_program_outcome(self):
        """
        OutcomeSnapshot fields (except complex types) must exist in ProgramOutcome.

        This test catches field drift - if you add a field to OutcomeSnapshot
        that doesn't exist in ProgramOutcome, the test fails.
        """
        snapshot_fields = set(OutcomeSnapshot.model_fields.keys())
        outcome_fields = set(ProgramOutcome.model_fields.keys())

        # These fields exist in both but may have different types
        # (e.g., violations is copied directly)
        for field in snapshot_fields:
            assert field in outcome_fields, (
                f"OutcomeSnapshot.{field} not found in ProgramOutcome - "
                f"add to ProgramOutcome or remove from snapshot"
            )

    def test_spec_snapshot_target_amount_matches_spec(self):
        """SpecSnapshot.target_amount must match SelectorSpec.target_amount type."""
        snapshot_field = SpecSnapshot.model_fields["target_amount"]
        spec_field = SelectorSpec.model_fields["target_amount"]

        # Both should be float (required)
        assert snapshot_field.annotation == spec_field.annotation

    def test_from_spec_factory_is_authoritative(self):
        """
        Verify from_spec() is the single derivation point.

        This test documents that direct construction should be avoided.
        """
        spec = SelectorSpec(
            program_type=ProgramType.SLB,
            objective=Objective.BALANCED,
            target_amount=100_000_000,
            hard_constraints=HardConstraints(max_net_leverage=4.0),
            soft_preferences=SoftPreferences(),
            asset_filters=AssetFilters(max_criticality=0.5),
        )

        # Using factory (correct)
        snapshot_via_factory = SpecSnapshot.from_spec(spec)

        # Direct construction (discouraged but possible)
        snapshot_direct = SpecSnapshot(
            target_amount=spec.target_amount,
            max_criticality=spec.asset_filters.max_criticality,
            max_net_leverage=spec.hard_constraints.max_net_leverage,
        )

        # Factory should produce complete snapshot
        assert snapshot_via_factory.target_amount == 100_000_000
        assert snapshot_via_factory.max_criticality == 0.5
        assert snapshot_via_factory.max_net_leverage == 4.0

        # Direct construction may miss fields (demonstrates why factory is preferred)
        # This is intentionally incomplete to show the risk
        assert snapshot_direct.min_fixed_charge_coverage is None  # Missing!

    def test_from_outcome_factory_is_authoritative(self):
        """
        Verify from_outcome() is the single derivation point.

        This test documents that direct construction should be avoided.
        """
        outcome = ProgramOutcome(
            status=SelectionStatus.OK,
            proceeds=50_000_000,
            leverage_after=3.5,
            interest_coverage_after=5.5,
            fixed_charge_coverage_after=2.8,
            critical_fraction=0.2,
        )

        # Using factory (correct)
        snapshot_via_factory = OutcomeSnapshot.from_outcome(outcome)

        # Factory produces complete snapshot
        assert snapshot_via_factory.status == SelectionStatus.OK
        assert snapshot_via_factory.proceeds == 50_000_000
        assert snapshot_via_factory.leverage_after == 3.5
        assert snapshot_via_factory.interest_coverage_after == 5.5
        assert snapshot_via_factory.fixed_charge_coverage_after == 2.8
        assert snapshot_via_factory.critical_fraction == 0.2


# =============================================================================
# AuditTraceEntry Tests
# =============================================================================


class TestAuditTraceEntry:
    """Test AuditTraceEntry model."""

    @pytest.fixture
    def sample_entry(self):
        """Create a sample entry for testing."""
        spec_snapshot = SpecSnapshot(
            target_amount=50_000_000,
            max_criticality=0.5,
            max_net_leverage=4.0,
        )
        outcome_snapshot = OutcomeSnapshot(
            status=SelectionStatus.OK,
            proceeds=45_000_000,
            leverage_after=3.5,
            critical_fraction=0.15,
        )
        return AuditTraceEntry(
            iteration=0,
            phase="initial",
            spec_snapshot=spec_snapshot,
            outcome_snapshot=outcome_snapshot,
            policy_violations=[],
            target_before=None,
            target_after=50_000_000,
            timestamp="2024-01-15T10:30:00Z",
        )

    def test_valid_initial_entry(self, sample_entry):
        """Valid initial entry should pass."""
        assert sample_entry.iteration == 0
        assert sample_entry.phase == "initial"
        assert sample_entry.target_before is None

    def test_revision_entry(self):
        """Revision entry should have target_before set."""
        spec_snapshot = SpecSnapshot(target_amount=45_000_000)
        outcome_snapshot = OutcomeSnapshot(
            status=SelectionStatus.INFEASIBLE,
            proceeds=40_000_000,
            critical_fraction=0.2,
        )
        entry = AuditTraceEntry(
            iteration=1,
            phase="revision",
            spec_snapshot=spec_snapshot,
            outcome_snapshot=outcome_snapshot,
            policy_violations=[],
            target_before=50_000_000,
            target_after=45_000_000,
            timestamp="2024-01-15T10:31:00Z",
        )
        assert entry.phase == "revision"
        assert entry.target_before == 50_000_000

    def test_entry_with_policy_violations(self):
        """Entry can have policy violations."""
        violation = PolicyViolation(
            code=PolicyViolationCode.TARGET_DROP_EXCEEDED,
            detail="Drop exceeded 20% limit",
            field="target_amount",
            attempted=35_000_000,
            limit=40_000_000,
            adjusted_to=40_000_000,
        )
        spec_snapshot = SpecSnapshot(target_amount=40_000_000)
        outcome_snapshot = OutcomeSnapshot(
            status=SelectionStatus.INFEASIBLE,
            proceeds=35_000_000,
            critical_fraction=0.1,
        )
        entry = AuditTraceEntry(
            iteration=2,
            phase="revision",
            spec_snapshot=spec_snapshot,
            outcome_snapshot=outcome_snapshot,
            policy_violations=[violation],
            target_before=50_000_000,
            target_after=40_000_000,
            timestamp="2024-01-15T10:32:00Z",
        )
        assert len(entry.policy_violations) == 1
        assert entry.policy_violations[0].code == PolicyViolationCode.TARGET_DROP_EXCEEDED

    def test_phase_literal_validation(self):
        """Phase must be 'initial' or 'revision'."""
        spec_snapshot = SpecSnapshot(target_amount=50_000_000)
        outcome_snapshot = OutcomeSnapshot(
            status=SelectionStatus.OK,
            proceeds=45_000_000,
            critical_fraction=0.15,
        )
        with pytest.raises(ValidationError):
            AuditTraceEntry(
                iteration=0,
                phase="unknown",  # Invalid
                spec_snapshot=spec_snapshot,
                outcome_snapshot=outcome_snapshot,
                target_after=50_000_000,
                timestamp="2024-01-15T10:30:00Z",
            )


# =============================================================================
# AuditTrace Tests
# =============================================================================


class TestAuditTrace:
    """Test AuditTrace model."""

    def test_valid_audit_trace_user_override(self):
        """Valid audit trace with user override source."""
        trace = AuditTrace(
            entries=[],
            original_target=50_000_000,
            floor_target=50_000_000,  # 100% floor for override
            floor_fraction=1.0,
            target_source="user_override",
            started_at="2024-01-15T10:30:00Z",
            completed_at=None,
        )
        assert trace.target_source == "user_override"
        assert trace.floor_fraction == 1.0
        assert trace.floor_target == trace.original_target

    def test_valid_audit_trace_llm_extraction(self):
        """Valid audit trace with LLM extraction source."""
        trace = AuditTrace(
            entries=[],
            original_target=50_000_000,
            floor_target=37_500_000,  # 75% floor
            floor_fraction=0.75,
            target_source="llm_extraction",
            started_at="2024-01-15T10:30:00Z",
        )
        assert trace.target_source == "llm_extraction"
        assert trace.floor_fraction == 0.75
        assert trace.floor_target == trace.original_target * 0.75

    def test_audit_trace_with_entries(self):
        """Audit trace with iteration entries."""
        entry = AuditTraceEntry(
            iteration=0,
            phase="initial",
            spec_snapshot=SpecSnapshot(target_amount=50_000_000),
            outcome_snapshot=OutcomeSnapshot(
                status=SelectionStatus.OK,
                proceeds=45_000_000,
                critical_fraction=0.15,
            ),
            target_after=50_000_000,
            timestamp="2024-01-15T10:30:00Z",
        )
        trace = AuditTrace(
            entries=[entry],
            original_target=50_000_000,
            floor_target=50_000_000,
            floor_fraction=1.0,
            target_source="user_override",
            started_at="2024-01-15T10:30:00Z",
            completed_at="2024-01-15T10:30:05Z",
        )
        assert len(trace.entries) == 1
        assert trace.completed_at is not None

    def test_target_source_literal_validation(self):
        """target_source must be valid literal."""
        with pytest.raises(ValidationError):
            AuditTrace(
                entries=[],
                original_target=50_000_000,
                floor_target=50_000_000,
                floor_fraction=1.0,
                target_source="unknown",  # Invalid
                started_at="2024-01-15T10:30:00Z",
            )

    def test_floor_fraction_bounds(self):
        """floor_fraction must be in [0, 1]."""
        with pytest.raises(ValidationError):
            AuditTrace(
                entries=[],
                original_target=50_000_000,
                floor_target=50_000_000,
                floor_fraction=1.5,  # Invalid
                target_source="user_override",
                started_at="2024-01-15T10:30:00Z",
            )

    def test_original_target_must_be_positive(self):
        """original_target must be > 0."""
        with pytest.raises(ValidationError):
            AuditTrace(
                entries=[],
                original_target=0,  # Invalid
                floor_target=0,
                floor_fraction=1.0,
                target_source="user_override",
                started_at="2024-01-15T10:30:00Z",
            )

    def test_serialization_roundtrip(self):
        """Audit trace should serialize and deserialize correctly."""
        entry = AuditTraceEntry(
            iteration=0,
            phase="initial",
            spec_snapshot=SpecSnapshot(
                target_amount=50_000_000,
                max_net_leverage=4.0,
            ),
            outcome_snapshot=OutcomeSnapshot(
                status=SelectionStatus.OK,
                proceeds=45_000_000,
                critical_fraction=0.15,
            ),
            target_after=50_000_000,
            timestamp="2024-01-15T10:30:00Z",
        )
        trace = AuditTrace(
            entries=[entry],
            original_target=50_000_000,
            floor_target=50_000_000,
            floor_fraction=1.0,
            target_source="user_override",
            started_at="2024-01-15T10:30:00Z",
            completed_at="2024-01-15T10:30:05Z",
        )

        json_str = trace.model_dump_json()
        restored = AuditTrace.model_validate_json(json_str)

        assert restored.original_target == 50_000_000
        assert restored.target_source == "user_override"
        assert len(restored.entries) == 1
        assert restored.entries[0].phase == "initial"
