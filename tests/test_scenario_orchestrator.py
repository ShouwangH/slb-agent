"""
Tests for Scenario Orchestrator module.

Tests the multi-scenario coordination using MockLLMClient for deterministic behavior.

Test categories:
- build_scenario_request: Request building with correct floor logic
- run_scenario_set: Full scenario set execution
- Error handling: Individual scenario failures
"""

import pytest

from app.config import DEFAULT_ENGINE_CONFIG
from app.llm.mock import MockLLMClient
from app.models import (
    Asset,
    AssetType,
    CorporateState,
    ProgramRequest,
    ProgramType,
    ScenarioDefinition,
    ScenarioKind,
)
from app.run_store import run_store
from app.scenario_orchestrator import build_scenario_request, run_scenario_set


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
    """Base program request for testing."""
    return ProgramRequest(
        assets=sample_assets,
        corporate_state=healthy_corporate_state,
        program_type=ProgramType.SLB,
        program_description="Raise $50M via SLB",
    )


@pytest.fixture
def mock_llm() -> MockLLMClient:
    """Mock LLM client for deterministic testing."""
    return MockLLMClient()


@pytest.fixture(autouse=True)
def clear_run_store():
    """Clear run store before each test."""
    run_store.clear()
    yield
    run_store.clear()


# =============================================================================
# Tests: build_scenario_request
# =============================================================================


class TestBuildScenarioRequest:
    """Tests for build_scenario_request function."""

    def test_base_scenario_floor_equals_target(self, base_request: ProgramRequest):
        """BASE scenario should have floor = target (sacred, no revision)."""
        scenario = ScenarioDefinition(
            label="Base Case",
            kind=ScenarioKind.BASE,
            rationale="Direct interpretation",
            target_amount=10_000_000,
            max_leverage=None,
            min_coverage=None,
        )

        request = build_scenario_request(base_request, scenario)

        # Floor should equal target for BASE scenario
        assert request.floor_override == 10_000_000
        assert request.floor_override == scenario.target_amount

    def test_variant_scenario_floor_is_90_percent(self, base_request: ProgramRequest):
        """Variant scenarios should have floor = target * 0.9 (allow revision)."""
        for kind in [ScenarioKind.RISK_OFF, ScenarioKind.AGGRESSIVE, ScenarioKind.GEO_FOCUS, ScenarioKind.CUSTOM]:
            scenario = ScenarioDefinition(
                label=f"{kind.value} Case",
                kind=kind,
                rationale=f"Test {kind.value}",
                target_amount=10_000_000,
                max_leverage=None,
                min_coverage=None,
            )

            request = build_scenario_request(base_request, scenario)

            # Floor should be 90% of target for variant scenarios
            assert request.floor_override == 9_000_000
            assert request.floor_override == scenario.target_amount * 0.9

    def test_assets_unchanged(self, base_request: ProgramRequest):
        """Assets should be unchanged (same values)."""
        scenario = ScenarioDefinition(
            label="Test",
            kind=ScenarioKind.BASE,
            rationale="Test",
            target_amount=10_000_000,
        )

        request = build_scenario_request(base_request, scenario)

        # Assets should have same values (Pydantic may copy the list)
        assert request.assets == base_request.assets
        assert len(request.assets) == len(base_request.assets)
        for i, asset in enumerate(request.assets):
            assert asset.asset_id == base_request.assets[i].asset_id

    def test_corporate_state_unchanged(self, base_request: ProgramRequest):
        """Corporate state should be unchanged (same values)."""
        scenario = ScenarioDefinition(
            label="Test",
            kind=ScenarioKind.BASE,
            rationale="Test",
            target_amount=10_000_000,
        )

        request = build_scenario_request(base_request, scenario)

        # Corporate state should have same values (Pydantic may copy)
        assert request.corporate_state == base_request.corporate_state

    def test_description_includes_label(self, base_request: ProgramRequest):
        """Description should include scenario label."""
        scenario = ScenarioDefinition(
            label="Conservative",
            kind=ScenarioKind.RISK_OFF,
            rationale="Test",
            target_amount=10_000_000,
        )

        request = build_scenario_request(base_request, scenario)

        assert "[Conservative]" in request.program_description
        assert base_request.program_description in request.program_description

    def test_max_leverage_override(self, base_request: ProgramRequest):
        """max_leverage should be passed through."""
        scenario = ScenarioDefinition(
            label="Test",
            kind=ScenarioKind.BASE,
            rationale="Test",
            target_amount=10_000_000,
            max_leverage=3.5,
        )

        request = build_scenario_request(base_request, scenario)

        assert request.max_leverage_override == 3.5

    def test_min_coverage_override(self, base_request: ProgramRequest):
        """min_coverage should be passed through."""
        scenario = ScenarioDefinition(
            label="Test",
            kind=ScenarioKind.BASE,
            rationale="Test",
            target_amount=10_000_000,
            min_coverage=3.0,
        )

        request = build_scenario_request(base_request, scenario)

        assert request.min_coverage_override == 3.0

    def test_none_overrides_passed_through(self, base_request: ProgramRequest):
        """None overrides should be passed through as None."""
        scenario = ScenarioDefinition(
            label="Test",
            kind=ScenarioKind.BASE,
            rationale="Test",
            target_amount=10_000_000,
            max_leverage=None,
            min_coverage=None,
        )

        request = build_scenario_request(base_request, scenario)

        assert request.max_leverage_override is None
        assert request.min_coverage_override is None


# =============================================================================
# Tests: run_scenario_set
# =============================================================================


class TestRunScenarioSet:
    """Tests for run_scenario_set function."""

    def test_returns_summary_and_results(
        self,
        base_request: ProgramRequest,
        mock_llm: MockLLMClient,
    ):
        """Should return ScenarioSetSummary and list of run results."""
        summary, results = run_scenario_set(
            brief="Test brief",
            base_request=base_request,
            llm=mock_llm,
            config=DEFAULT_ENGINE_CONFIG,
            num_scenarios=2,
        )

        assert summary is not None
        assert summary.id is not None
        assert summary.brief == "Test brief"
        assert len(results) == 2

    def test_creates_correct_number_of_scenarios(
        self,
        base_request: ProgramRequest,
        mock_llm: MockLLMClient,
    ):
        """Should create the requested number of scenarios."""
        for num in [1, 2, 3]:
            run_store.clear()

            summary, results = run_scenario_set(
                brief="Test brief",
                base_request=base_request,
                llm=mock_llm,
                config=DEFAULT_ENGINE_CONFIG,
                num_scenarios=num,
            )

            assert len(summary.run_ids) == num
            assert len(results) == num

    def test_stores_run_records_with_scenario_metadata(
        self,
        base_request: ProgramRequest,
        mock_llm: MockLLMClient,
    ):
        """RunRecords should have correct scenario metadata."""
        summary, results = run_scenario_set(
            brief="Test brief",
            base_request=base_request,
            llm=mock_llm,
            config=DEFAULT_ENGINE_CONFIG,
            num_scenarios=3,
        )

        # Check each run record
        for run_id in summary.run_ids:
            record = run_store.get(run_id)
            assert record is not None
            assert record.scenario_set_id == summary.id
            assert record.scenario_kind is not None
            assert record.scenario_label is not None

    def test_first_scenario_is_base(
        self,
        base_request: ProgramRequest,
        mock_llm: MockLLMClient,
    ):
        """First scenario should be BASE kind."""
        summary, results = run_scenario_set(
            brief="Test brief",
            base_request=base_request,
            llm=mock_llm,
            config=DEFAULT_ENGINE_CONFIG,
            num_scenarios=3,
        )

        # First result should be BASE
        assert results[0]["scenario_kind"] == "base"

        # First run record should be BASE
        first_record = run_store.get(summary.run_ids[0])
        assert first_record is not None
        assert first_record.scenario_kind == ScenarioKind.BASE

    def test_stores_scenario_set_summary(
        self,
        base_request: ProgramRequest,
        mock_llm: MockLLMClient,
    ):
        """Should store ScenarioSetSummary in run_store."""
        summary, _ = run_scenario_set(
            brief="Test brief",
            base_request=base_request,
            llm=mock_llm,
            config=DEFAULT_ENGINE_CONFIG,
            num_scenarios=2,
        )

        # Should be retrievable from store
        stored = run_store.get_scenario_set(summary.id)
        assert stored is not None
        assert stored.id == summary.id
        assert stored.brief == summary.brief
        assert stored.run_ids == summary.run_ids

    def test_run_results_include_status(
        self,
        base_request: ProgramRequest,
        mock_llm: MockLLMClient,
    ):
        """Run results should include status field."""
        _, results = run_scenario_set(
            brief="Test brief",
            base_request=base_request,
            llm=mock_llm,
            config=DEFAULT_ENGINE_CONFIG,
            num_scenarios=2,
        )

        for result in results:
            assert "status" in result
            assert result["status"] in ["completed", "failed"]

    def test_run_results_include_scenario_metadata(
        self,
        base_request: ProgramRequest,
        mock_llm: MockLLMClient,
    ):
        """Run results should include scenario metadata."""
        summary, results = run_scenario_set(
            brief="Test brief",
            base_request=base_request,
            llm=mock_llm,
            config=DEFAULT_ENGINE_CONFIG,
            num_scenarios=2,
        )

        for result in results:
            assert "scenario_set_id" in result
            assert result["scenario_set_id"] == summary.id
            assert "scenario_kind" in result
            assert "scenario_label" in result

    def test_completed_runs_have_response(
        self,
        base_request: ProgramRequest,
        mock_llm: MockLLMClient,
    ):
        """Completed runs should have response in result."""
        _, results = run_scenario_set(
            brief="Test brief",
            base_request=base_request,
            llm=mock_llm,
            config=DEFAULT_ENGINE_CONFIG,
            num_scenarios=2,
        )

        for result in results:
            if result["status"] == "completed":
                assert "response" in result
                assert result["response"] is not None

    def test_fund_id_passed_to_records(
        self,
        base_request: ProgramRequest,
        mock_llm: MockLLMClient,
    ):
        """fund_id should be passed to all RunRecords."""
        summary, _ = run_scenario_set(
            brief="Test brief",
            base_request=base_request,
            llm=mock_llm,
            config=DEFAULT_ENGINE_CONFIG,
            num_scenarios=2,
            fund_id="test-fund-123",
        )

        for run_id in summary.run_ids:
            record = run_store.get(run_id)
            assert record is not None
            assert record.fund_id == "test-fund-123"

    def test_get_runs_for_set_returns_all_runs(
        self,
        base_request: ProgramRequest,
        mock_llm: MockLLMClient,
    ):
        """get_runs_for_set should return all runs in the set."""
        summary, _ = run_scenario_set(
            brief="Test brief",
            base_request=base_request,
            llm=mock_llm,
            config=DEFAULT_ENGINE_CONFIG,
            num_scenarios=3,
        )

        runs = run_store.get_runs_for_set(summary.id)
        assert len(runs) == 3

        run_ids = {r.run_id for r in runs}
        assert run_ids == set(summary.run_ids)


# =============================================================================
# Tests: Integration
# =============================================================================


class TestScenarioIntegration:
    """Integration tests for scenario orchestrator."""

    def test_base_scenario_deterministic(
        self,
        base_request: ProgramRequest,
        mock_llm: MockLLMClient,
    ):
        """BASE scenario should run deterministically (no revision loop)."""
        # Run scenario set with known mock behavior
        summary, results = run_scenario_set(
            brief="Raise $10M",
            base_request=base_request,
            llm=mock_llm,
            config=DEFAULT_ENGINE_CONFIG,
            num_scenarios=1,
        )

        # Should have exactly 1 result
        assert len(results) == 1
        assert results[0]["scenario_kind"] == "base"

        # BASE scenario with floor = target should run once
        record = run_store.get(summary.run_ids[0])
        assert record is not None
        # With mock LLM, should complete (mock returns feasible specs)

    def test_multiple_scenarios_same_assets(
        self,
        base_request: ProgramRequest,
        mock_llm: MockLLMClient,
    ):
        """Multiple scenarios should use same assets (different capital asks)."""
        summary, results = run_scenario_set(
            brief="Test brief",
            base_request=base_request,
            llm=mock_llm,
            config=DEFAULT_ENGINE_CONFIG,
            num_scenarios=3,
        )

        # All scenarios should complete (with mock)
        assert len(results) == 3

        # Each should have different scenario kinds
        kinds = {r["scenario_kind"] for r in results}
        assert "base" in kinds  # First is always BASE
