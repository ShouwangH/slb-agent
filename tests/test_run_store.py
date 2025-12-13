"""
Tests for run_store module, focusing on scenario set functionality.
"""

import pytest

from app.models import ScenarioKind, ScenarioSetSummary
from app.run_store import RunRecord, RunStore


class TestScenarioSetStorage:
    """Tests for scenario set CRUD operations."""

    @pytest.fixture
    def store(self) -> RunStore:
        """Create a fresh store for each test."""
        return RunStore()

    @pytest.fixture
    def sample_summary(self) -> ScenarioSetSummary:
        """Create a sample scenario set summary."""
        return ScenarioSetSummary(
            id="set-123",
            brief="Test brief for $10M SLB",
            created_at="2025-01-01T00:00:00Z",
            run_ids=["run-1", "run-2", "run-3"],
        )

    def test_create_scenario_set(self, store: RunStore, sample_summary: ScenarioSetSummary):
        """Test creating a scenario set."""
        store.create_scenario_set(sample_summary)

        result = store.get_scenario_set("set-123")
        assert result is not None
        assert result.id == "set-123"
        assert result.brief == "Test brief for $10M SLB"

    def test_get_scenario_set_not_found(self, store: RunStore):
        """Test getting a non-existent scenario set returns None."""
        result = store.get_scenario_set("nonexistent")
        assert result is None

    def test_list_scenario_sets_empty(self, store: RunStore):
        """Test listing scenario sets when empty."""
        result = store.list_scenario_sets()
        assert result == []

    def test_list_scenario_sets_returns_most_recent_first(self, store: RunStore):
        """Test that list_scenario_sets returns most recent first."""
        for i in range(3):
            store.create_scenario_set(
                ScenarioSetSummary(
                    id=f"set-{i}",
                    brief=f"Brief {i}",
                    created_at=f"2025-01-0{i+1}T00:00:00Z",
                    run_ids=[],
                )
            )

        result = store.list_scenario_sets()
        assert len(result) == 3
        # Most recent (last inserted) should be first
        assert result[0].id == "set-2"
        assert result[1].id == "set-1"
        assert result[2].id == "set-0"

    def test_list_scenario_sets_respects_limit(self, store: RunStore):
        """Test that list_scenario_sets respects the limit parameter."""
        for i in range(5):
            store.create_scenario_set(
                ScenarioSetSummary(
                    id=f"set-{i}",
                    brief=f"Brief {i}",
                    created_at=f"2025-01-0{i+1}T00:00:00Z",
                    run_ids=[],
                )
            )

        result = store.list_scenario_sets(limit=2)
        assert len(result) == 2

    def test_clear_clears_scenario_sets(self, store: RunStore, sample_summary: ScenarioSetSummary):
        """Test that clear() also clears scenario sets."""
        store.create_scenario_set(sample_summary)
        assert store.get_scenario_set("set-123") is not None

        store.clear()

        assert store.get_scenario_set("set-123") is None
        assert store.list_scenario_sets() == []


class TestGetRunsForSet:
    """Tests for get_runs_for_set method."""

    @pytest.fixture
    def store(self) -> RunStore:
        """Create a fresh store for each test."""
        return RunStore()

    def test_get_runs_for_set_empty(self, store: RunStore):
        """Test getting runs for a set with no runs."""
        result = store.get_runs_for_set("set-123")
        assert result == []

    def test_get_runs_for_set_filters_correctly(self, store: RunStore):
        """Test that get_runs_for_set only returns runs belonging to the set."""
        # Create runs for set-123
        for i in range(2):
            store.create(
                RunRecord(
                    run_id=f"run-set123-{i}",
                    fund_id=None,
                    program_description="Test",
                    response=None,
                    error=None,
                    created_at="2025-01-01T00:00:00Z",
                    scenario_set_id="set-123",
                    scenario_kind=ScenarioKind.BASE if i == 0 else ScenarioKind.AGGRESSIVE,
                    scenario_label=f"Scenario {i}",
                )
            )

        # Create a run for a different set
        store.create(
            RunRecord(
                run_id="run-other",
                fund_id=None,
                program_description="Test",
                response=None,
                error=None,
                created_at="2025-01-01T00:00:00Z",
                scenario_set_id="set-456",
                scenario_kind=ScenarioKind.BASE,
                scenario_label="Other",
            )
        )

        # Create a single-scenario run (no set)
        store.create(
            RunRecord(
                run_id="run-single",
                fund_id=None,
                program_description="Test",
                response=None,
                error=None,
                created_at="2025-01-01T00:00:00Z",
            )
        )

        result = store.get_runs_for_set("set-123")
        assert len(result) == 2
        assert all(r.scenario_set_id == "set-123" for r in result)

    def test_get_runs_for_set_returns_runs_with_metadata(self, store: RunStore):
        """Test that returned runs have correct scenario metadata."""
        store.create(
            RunRecord(
                run_id="run-1",
                fund_id=None,
                program_description="Test",
                response=None,
                error=None,
                created_at="2025-01-01T00:00:00Z",
                scenario_set_id="set-123",
                scenario_kind=ScenarioKind.RISK_OFF,
                scenario_label="Conservative",
            )
        )

        result = store.get_runs_for_set("set-123")
        assert len(result) == 1
        assert result[0].scenario_kind == ScenarioKind.RISK_OFF
        assert result[0].scenario_label == "Conservative"


class TestClearIncludesBoth:
    """Test that clear() clears both runs and scenario sets."""

    def test_clear_clears_runs_and_sets(self):
        """Test that clear() clears both runs and scenario sets."""
        store = RunStore()

        # Add a run
        store.create(
            RunRecord(
                run_id="run-1",
                fund_id=None,
                program_description="Test",
                response=None,
                error=None,
                created_at="2025-01-01T00:00:00Z",
            )
        )

        # Add a scenario set
        store.create_scenario_set(
            ScenarioSetSummary(
                id="set-1",
                brief="Test",
                created_at="2025-01-01T00:00:00Z",
                run_ids=[],
            )
        )

        # Verify both exist
        assert store.get("run-1") is not None
        assert store.get_scenario_set("set-1") is not None

        # Clear
        store.clear()

        # Verify both cleared
        assert store.get("run-1") is None
        assert store.get_scenario_set("set-1") is None
