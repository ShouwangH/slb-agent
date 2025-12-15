"""Tests for LLM Interface & Mock Client (PR7).

Tests cover:
- MockLLMClient implements LLMClient protocol
- generate_selector_spec returns valid structured data
- revise_selector_spec reduces target and relaxes filters
- generate_explanation_summary produces narrative
- Call tracking works correctly
"""

import pytest

from app.llm import LLMClient, MockLLMClient
from app.models import (
    AssetFilters,
    ConstraintViolation,
    ExplanationNode,
    HardConstraints,
    Objective,
    ProgramOutcome,
    ProgramType,
    ScenarioKind,
    SelectionStatus,
    SelectorSpec,
    SoftPreferences,
)


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


class TestLLMClientProtocol:
    """Tests for LLMClient protocol compliance."""

    def test_mock_implements_protocol(self):
        """MockLLMClient implements LLMClient protocol."""
        mock = MockLLMClient()
        assert isinstance(mock, LLMClient)

    def test_protocol_methods_exist(self):
        """MockLLMClient has all required protocol methods."""
        mock = MockLLMClient()

        assert hasattr(mock, "generate_selector_spec")
        assert callable(mock.generate_selector_spec)

        assert hasattr(mock, "revise_selector_spec")
        assert callable(mock.revise_selector_spec)

        assert hasattr(mock, "generate_explanation_summary")
        assert callable(mock.generate_explanation_summary)

        assert hasattr(mock, "generate_scenario_definitions")
        assert callable(mock.generate_scenario_definitions)


# =============================================================================
# generate_selector_spec Tests
# =============================================================================


class TestGenerateSelectorSpec:
    """Tests for generate_selector_spec method."""

    def test_returns_valid_selector_spec(self):
        """Returns a valid SelectorSpec object."""
        mock = MockLLMClient()

        spec = mock.generate_selector_spec(
            program_type=ProgramType.SLB,
            program_description="Raise capital via SLB",
            asset_summary="10 assets, $500M total value",
        )

        assert isinstance(spec, SelectorSpec)
        assert spec.program_type == ProgramType.SLB
        assert spec.target_amount > 0
        assert spec.hard_constraints is not None
        assert spec.soft_preferences is not None
        assert spec.asset_filters is not None

    def test_uses_default_target(self):
        """Uses configured default target when not in description."""
        mock = MockLLMClient(default_target_amount=50_000_000)

        spec = mock.generate_selector_spec(
            program_type=ProgramType.SLB,
            program_description="Raise capital via SLB",
            asset_summary="10 assets",
        )

        assert spec.target_amount == 50_000_000

    def test_extracts_target_from_description_millions(self):
        """Extracts target amount from description (millions)."""
        mock = MockLLMClient()

        spec = mock.generate_selector_spec(
            program_type=ProgramType.SLB,
            program_description="Raise $150M via sale-leaseback",
            asset_summary="10 assets",
        )

        assert spec.target_amount == 150_000_000

    def test_extracts_target_from_description_million_word(self):
        """Extracts target with 'million' spelled out."""
        mock = MockLLMClient()

        spec = mock.generate_selector_spec(
            program_type=ProgramType.SLB,
            program_description="Need $75 million for debt paydown",
            asset_summary="10 assets",
        )

        assert spec.target_amount == 75_000_000

    def test_determines_maximize_objective(self):
        """Detects MAXIMIZE_PROCEEDS objective from keywords."""
        mock = MockLLMClient()

        spec = mock.generate_selector_spec(
            program_type=ProgramType.SLB,
            program_description="Maximize proceeds from SLB program",
            asset_summary="10 assets",
        )

        assert spec.objective == Objective.MAXIMIZE_PROCEEDS

    def test_determines_minimize_risk_objective(self):
        """Detects MINIMIZE_RISK objective from keywords."""
        mock = MockLLMClient()

        spec = mock.generate_selector_spec(
            program_type=ProgramType.SLB,
            program_description="Conservative SLB to minimize risk",
            asset_summary="10 assets",
        )

        assert spec.objective == Objective.MINIMIZE_RISK

    def test_defaults_to_balanced_objective(self):
        """Defaults to BALANCED objective when no keywords."""
        mock = MockLLMClient()

        spec = mock.generate_selector_spec(
            program_type=ProgramType.SLB,
            program_description="Raise capital for operations",
            asset_summary="10 assets",
        )

        assert spec.objective == Objective.BALANCED

    def test_custom_spec_overrides_generation(self):
        """Custom spec is returned instead of generated one."""
        custom = SelectorSpec(
            program_type=ProgramType.SLB,
            objective=Objective.MAXIMIZE_PROCEEDS,
            target_amount=999_000_000,
            hard_constraints=HardConstraints(max_net_leverage=5.0),
            soft_preferences=SoftPreferences(),
            asset_filters=AssetFilters(),
        )
        mock = MockLLMClient(custom_spec=custom)

        spec = mock.generate_selector_spec(
            program_type=ProgramType.SLB,
            program_description="Raise $100M",
            asset_summary="10 assets",
        )

        assert spec.target_amount == 999_000_000
        assert spec.hard_constraints.max_net_leverage == 5.0

    def test_increments_call_count(self):
        """Call count is incremented."""
        mock = MockLLMClient()

        assert mock.call_counts["generate_selector_spec"] == 0

        mock.generate_selector_spec(ProgramType.SLB, "desc", "summary")
        assert mock.call_counts["generate_selector_spec"] == 1

        mock.generate_selector_spec(ProgramType.SLB, "desc", "summary")
        assert mock.call_counts["generate_selector_spec"] == 2

    def test_records_call_history(self):
        """Call history is recorded."""
        mock = MockLLMClient()

        mock.generate_selector_spec(
            program_type=ProgramType.SLB,
            program_description="Test description",
            asset_summary="Test summary",
        )

        assert len(mock.call_history) == 1
        assert mock.call_history[0]["method"] == "generate_selector_spec"
        assert mock.call_history[0]["program_description"] == "Test description"


# =============================================================================
# revise_selector_spec Tests
# =============================================================================


class TestReviseSelectorSpec:
    """Tests for revise_selector_spec method."""

    def _make_prev_spec(self, target: float = 100_000_000) -> SelectorSpec:
        """Create a previous spec for revision tests."""
        return SelectorSpec(
            program_type=ProgramType.SLB,
            objective=Objective.BALANCED,
            target_amount=target,
            hard_constraints=HardConstraints(
                max_net_leverage=4.0,
                min_fixed_charge_coverage=3.0,
            ),
            soft_preferences=SoftPreferences(),
            asset_filters=AssetFilters(
                min_leaseability_score=0.5,
                max_criticality=0.7,
            ),
        )

    def _make_infeasible_outcome(self) -> ProgramOutcome:
        """Create an infeasible outcome for revision tests."""
        return ProgramOutcome(
            status=SelectionStatus.INFEASIBLE,
            selected_assets=[],
            proceeds=0,
            violations=[
                ConstraintViolation(
                    code="TARGET_NOT_MET",
                    detail="Proceeds below target",
                    actual=80_000_000,
                    limit=100_000_000,
                )
            ],
        )

    def test_returns_valid_selector_spec(self):
        """Returns a valid SelectorSpec object."""
        mock = MockLLMClient()
        prev_spec = self._make_prev_spec()
        outcome = self._make_infeasible_outcome()

        revised = mock.revise_selector_spec(
            original_description="Raise capital",
            previous_spec=prev_spec,
            outcome=outcome,
        )

        assert isinstance(revised, SelectorSpec)
        assert revised.program_type == ProgramType.SLB

    def test_reduces_target_amount(self):
        """Target is reduced by configured percentage."""
        mock = MockLLMClient(revision_target_reduction=0.15)  # 15% reduction
        prev_spec = self._make_prev_spec(target=100_000_000)
        outcome = self._make_infeasible_outcome()

        revised = mock.revise_selector_spec(
            original_description="Raise capital",
            previous_spec=prev_spec,
            outcome=outcome,
        )

        expected = 100_000_000 * 0.85  # 15% reduction
        assert revised.target_amount == expected

    def test_preserves_hard_constraints(self):
        """Hard constraints are preserved (immutable)."""
        mock = MockLLMClient()
        prev_spec = self._make_prev_spec()
        outcome = self._make_infeasible_outcome()

        revised = mock.revise_selector_spec(
            original_description="Raise capital",
            previous_spec=prev_spec,
            outcome=outcome,
        )

        assert revised.hard_constraints.max_net_leverage == 4.0
        assert revised.hard_constraints.min_fixed_charge_coverage == 3.0

    def test_relaxes_min_leaseability(self):
        """min_leaseability_score is relaxed (decreased)."""
        mock = MockLLMClient()
        prev_spec = self._make_prev_spec()
        prev_spec.asset_filters.min_leaseability_score = 0.5
        outcome = self._make_infeasible_outcome()

        revised = mock.revise_selector_spec(
            original_description="Raise capital",
            previous_spec=prev_spec,
            outcome=outcome,
        )

        # Should be reduced by 0.1
        assert revised.asset_filters.min_leaseability_score == 0.4

    def test_relaxes_max_criticality(self):
        """max_criticality is relaxed (increased)."""
        mock = MockLLMClient()
        prev_spec = self._make_prev_spec()
        prev_spec.asset_filters.max_criticality = 0.7
        outcome = self._make_infeasible_outcome()

        revised = mock.revise_selector_spec(
            original_description="Raise capital",
            previous_spec=prev_spec,
            outcome=outcome,
        )

        # Should be increased by 0.1
        assert revised.asset_filters.max_criticality == pytest.approx(0.8)

    def test_min_threshold_does_not_go_negative(self):
        """min_leaseability_score doesn't go below 0."""
        mock = MockLLMClient()
        prev_spec = self._make_prev_spec()
        prev_spec.asset_filters.min_leaseability_score = 0.05
        outcome = self._make_infeasible_outcome()

        revised = mock.revise_selector_spec(
            original_description="Raise capital",
            previous_spec=prev_spec,
            outcome=outcome,
        )

        assert revised.asset_filters.min_leaseability_score == 0.0

    def test_max_threshold_does_not_exceed_one(self):
        """max_criticality doesn't exceed 1.0."""
        mock = MockLLMClient()
        prev_spec = self._make_prev_spec()
        prev_spec.asset_filters.max_criticality = 0.95
        outcome = self._make_infeasible_outcome()

        revised = mock.revise_selector_spec(
            original_description="Raise capital",
            previous_spec=prev_spec,
            outcome=outcome,
        )

        assert revised.asset_filters.max_criticality == 1.0

    def test_increments_call_count(self):
        """Call count is incremented."""
        mock = MockLLMClient()
        prev_spec = self._make_prev_spec()
        outcome = self._make_infeasible_outcome()

        assert mock.call_counts["revise_selector_spec"] == 0

        mock.revise_selector_spec("desc", prev_spec, outcome)
        assert mock.call_counts["revise_selector_spec"] == 1


# =============================================================================
# generate_explanation_summary Tests
# =============================================================================


class TestGenerateExplanationSummary:
    """Tests for generate_explanation_summary method."""

    def test_returns_string(self):
        """Returns a string summary."""
        mock = MockLLMClient()

        nodes = [
            ExplanationNode(
                id="driver_1",
                label="Target Achieved",
                severity="info",
                category="driver",
            )
        ]

        summary = mock.generate_explanation_summary(nodes)

        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_handles_empty_nodes(self):
        """Returns summary even with no nodes."""
        mock = MockLLMClient()

        summary = mock.generate_explanation_summary([])

        assert isinstance(summary, str)
        assert "Successfully" in summary

    def test_handles_error_nodes(self):
        """Produces appropriate summary for error nodes."""
        mock = MockLLMClient()

        nodes = [
            ExplanationNode(
                id="constraint_1",
                label="Target Not Met",
                severity="error",
                category="constraint",
            )
        ]

        summary = mock.generate_explanation_summary(nodes)

        assert "unable" in summary.lower() or "issue" in summary.lower()
        assert "Target Not Met" in summary

    def test_includes_driver_info(self):
        """Includes driver information in summary."""
        mock = MockLLMClient()

        nodes = [
            ExplanationNode(
                id="driver_1",
                label="Leverage Reduced",
                severity="info",
                category="driver",
            )
        ]

        summary = mock.generate_explanation_summary(nodes)

        assert "Leverage Reduced" in summary

    def test_includes_risk_info(self):
        """Includes risk information in summary."""
        mock = MockLLMClient()

        nodes = [
            ExplanationNode(
                id="driver_1",
                label="Target Achieved",
                severity="info",
                category="driver",
            ),
            ExplanationNode(
                id="risk_1",
                label="Market Concentration",
                severity="warning",
                category="risk",
            ),
        ]

        summary = mock.generate_explanation_summary(nodes)

        assert "Market Concentration" in summary

    def test_custom_summary_overrides(self):
        """Custom summary is returned instead of generated one."""
        custom = "This is a custom summary."
        mock = MockLLMClient(custom_summary=custom)

        nodes = [
            ExplanationNode(
                id="driver_1",
                label="Something",
                severity="info",
                category="driver",
            )
        ]

        summary = mock.generate_explanation_summary(nodes)

        assert summary == custom

    def test_increments_call_count(self):
        """Call count is incremented."""
        mock = MockLLMClient()

        assert mock.call_counts["generate_explanation_summary"] == 0

        mock.generate_explanation_summary([])
        assert mock.call_counts["generate_explanation_summary"] == 1


# =============================================================================
# Call Tracking Tests
# =============================================================================


class TestCallTracking:
    """Tests for call tracking functionality."""

    def test_get_total_calls(self):
        """get_total_calls returns sum of all calls."""
        mock = MockLLMClient()

        mock.generate_selector_spec(ProgramType.SLB, "desc", "summary")
        mock.generate_selector_spec(ProgramType.SLB, "desc", "summary")
        mock.generate_explanation_summary([])

        assert mock.get_total_calls() == 3

    def test_reset_counts(self):
        """reset_counts clears all counters and history."""
        mock = MockLLMClient()

        mock.generate_selector_spec(ProgramType.SLB, "desc", "summary")
        mock.generate_explanation_summary([])

        assert mock.get_total_calls() == 2
        assert len(mock.call_history) == 2

        mock.reset_counts()

        assert mock.get_total_calls() == 0
        assert len(mock.call_history) == 0

    def test_call_history_contains_details(self):
        """Call history contains method details."""
        mock = MockLLMClient()

        mock.generate_selector_spec(
            program_type=ProgramType.SLB,
            program_description="Raise $100M",
            asset_summary="10 assets",
        )

        assert len(mock.call_history) == 1
        record = mock.call_history[0]
        assert record["method"] == "generate_selector_spec"
        assert record["program_type"] == ProgramType.SLB
        assert record["program_description"] == "Raise $100M"
        assert record["asset_summary"] == "10 assets"


# =============================================================================
# Integration Tests
# =============================================================================


class TestMockLLMIntegration:
    """Integration tests for MockLLMClient."""

    def test_multiple_revision_iterations(self):
        """Mock handles multiple revision iterations correctly."""
        mock = MockLLMClient(
            default_target_amount=100_000_000,
            revision_target_reduction=0.2,  # 20% reduction
        )

        # Initial spec
        spec = mock.generate_selector_spec(
            ProgramType.SLB,
            "Raise capital",
            "10 assets",
        )
        assert spec.target_amount == 100_000_000

        # First revision
        outcome = ProgramOutcome(
            status=SelectionStatus.INFEASIBLE,
            selected_assets=[],
            proceeds=0,
            violations=[
                ConstraintViolation(
                    code="TARGET_NOT_MET",
                    detail="Proceeds below target",
                    actual=50_000_000,
                    limit=100_000_000,
                )
            ],
        )
        spec = mock.revise_selector_spec("Raise capital", spec, outcome)
        assert spec.target_amount == 80_000_000  # 20% reduction

        # Second revision
        spec = mock.revise_selector_spec("Raise capital", spec, outcome)
        assert spec.target_amount == 64_000_000  # Another 20% reduction

        # Call counts
        assert mock.call_counts["generate_selector_spec"] == 1
        assert mock.call_counts["revise_selector_spec"] == 2

    def test_deterministic_behavior(self):
        """Mock produces same results for same inputs."""
        mock1 = MockLLMClient(default_target_amount=100_000_000)
        mock2 = MockLLMClient(default_target_amount=100_000_000)

        spec1 = mock1.generate_selector_spec(
            ProgramType.SLB,
            "Raise $50M via SLB",
            "10 assets",
        )
        spec2 = mock2.generate_selector_spec(
            ProgramType.SLB,
            "Raise $50M via SLB",
            "10 assets",
        )

        assert spec1.target_amount == spec2.target_amount
        assert spec1.objective == spec2.objective
        assert spec1.hard_constraints.max_net_leverage == spec2.hard_constraints.max_net_leverage


# =============================================================================
# generate_scenario_definitions Tests
# =============================================================================


class TestGenerateScenarioDefinitions:
    """Tests for generate_scenario_definitions method."""

    def test_returns_list_of_scenario_definitions(self):
        """Returns a list of ScenarioDefinition objects."""
        mock = MockLLMClient()

        scenarios = mock.generate_scenario_definitions(
            brief="Raise $100M via SLB",
            asset_summary="10 assets, $500M total value",
            num_scenarios=3,
        )

        assert isinstance(scenarios, list)
        assert len(scenarios) == 3

    def test_first_scenario_is_base(self):
        """First scenario MUST be kind=BASE."""
        mock = MockLLMClient()

        scenarios = mock.generate_scenario_definitions(
            brief="Raise $100M",
            asset_summary="10 assets",
            num_scenarios=3,
        )

        assert scenarios[0].kind == ScenarioKind.BASE
        assert scenarios[0].label == "Base Case"

    def test_returns_requested_number_of_scenarios(self):
        """Returns exactly the requested number of scenarios (up to 5)."""
        mock = MockLLMClient()

        for num in [1, 2, 3, 4, 5]:
            scenarios = mock.generate_scenario_definitions(
                brief="Raise capital",
                asset_summary="10 assets",
                num_scenarios=num,
            )
            assert len(scenarios) == num

    def test_max_five_scenarios(self):
        """Returns at most 5 scenarios even if more requested."""
        mock = MockLLMClient()

        scenarios = mock.generate_scenario_definitions(
            brief="Raise capital",
            asset_summary="10 assets",
            num_scenarios=10,
        )

        assert len(scenarios) == 5

    def test_extracts_target_from_brief(self):
        """Base scenario target is extracted from brief."""
        mock = MockLLMClient()

        scenarios = mock.generate_scenario_definitions(
            brief="Raise $80M via SLB",
            asset_summary="10 assets",
            num_scenarios=3,
        )

        # Base scenario should have extracted target
        assert scenarios[0].target_amount == 80_000_000

    def test_uses_default_target_when_not_in_brief(self):
        """Uses default target when brief doesn't specify amount."""
        mock = MockLLMClient(default_target_amount=50_000_000)

        scenarios = mock.generate_scenario_definitions(
            brief="Raise capital for operations",
            asset_summary="10 assets",
            num_scenarios=3,
        )

        assert scenarios[0].target_amount == 50_000_000

    def test_scenario_targets_vary(self):
        """Different scenarios have different targets."""
        mock = MockLLMClient()

        scenarios = mock.generate_scenario_definitions(
            brief="Raise $100M",
            asset_summary="10 assets",
            num_scenarios=3,
        )

        targets = [s.target_amount for s in scenarios]
        # All targets should be unique
        assert len(set(targets)) == 3

    def test_scenarios_have_required_fields(self):
        """All scenarios have required fields populated."""
        mock = MockLLMClient()

        scenarios = mock.generate_scenario_definitions(
            brief="Raise $100M",
            asset_summary="10 assets",
            num_scenarios=3,
        )

        for s in scenarios:
            assert s.label and len(s.label) > 0
            assert s.kind is not None
            assert s.rationale and len(s.rationale) > 0
            assert s.target_amount > 0

    def test_scenarios_have_unique_labels(self):
        """All scenarios have unique labels."""
        mock = MockLLMClient()

        scenarios = mock.generate_scenario_definitions(
            brief="Raise $100M",
            asset_summary="10 assets",
            num_scenarios=5,
        )

        labels = [s.label for s in scenarios]
        assert len(set(labels)) == 5

    def test_includes_conservative_scenario(self):
        """Includes a RISK_OFF (conservative) scenario."""
        mock = MockLLMClient()

        scenarios = mock.generate_scenario_definitions(
            brief="Raise $100M",
            asset_summary="10 assets",
            num_scenarios=3,
        )

        risk_off_scenarios = [s for s in scenarios if s.kind == ScenarioKind.RISK_OFF]
        assert len(risk_off_scenarios) >= 1

    def test_includes_aggressive_scenario(self):
        """Includes an AGGRESSIVE scenario."""
        mock = MockLLMClient()

        scenarios = mock.generate_scenario_definitions(
            brief="Raise $100M",
            asset_summary="10 assets",
            num_scenarios=3,
        )

        aggressive_scenarios = [s for s in scenarios if s.kind == ScenarioKind.AGGRESSIVE]
        assert len(aggressive_scenarios) >= 1

    def test_increments_call_count(self):
        """Call count is incremented."""
        mock = MockLLMClient()

        mock.generate_scenario_definitions("brief", "summary", 3)
        assert mock.call_counts.get("generate_scenario_definitions", 0) == 1

        mock.generate_scenario_definitions("brief", "summary", 2)
        assert mock.call_counts.get("generate_scenario_definitions", 0) == 2

    def test_records_call_history(self):
        """Call history is recorded."""
        mock = MockLLMClient()

        mock.generate_scenario_definitions(
            brief="Raise $100M",
            asset_summary="10 assets total",
            num_scenarios=3,
        )

        assert len(mock.call_history) == 1
        record = mock.call_history[0]
        assert record["method"] == "generate_scenario_definitions"
        assert record["brief"] == "Raise $100M"
        assert record["asset_summary"] == "10 assets total"
        assert record["num_scenarios"] == 3
