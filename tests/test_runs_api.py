"""
Tests for the Runs API endpoints.

Tests the runs endpoints:
- POST /api/runs - Create and execute a program run
- GET /api/runs/{run_id} - Get a single run by ID
- GET /api/runs - List runs

Test categories:
- Create run happy path (201)
- Create run with validation error (201 with status: "failed")
- Create run with infrastructure error (503)
- Get run by ID (200 / 404)
- List runs with filtering
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.api import app
from app.exceptions import InfrastructureError
from app.run_store import run_store


# =============================================================================
# Test Client & Fixtures
# =============================================================================


@pytest.fixture
def client() -> TestClient:
    """Create test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_run_store():
    """Clear the run store before and after each test."""
    run_store.clear()
    yield
    run_store.clear()


# =============================================================================
# Test Data
# =============================================================================


def make_valid_request() -> dict:
    """Create a valid program request payload."""
    return {
        "assets": [
            {
                "asset_id": "A001",
                "asset_type": "store",
                "market": "Dallas, TX",
                "noi": 500_000,
                "book_value": 5_000_000,
                "criticality": 0.3,
                "leaseability_score": 0.8,
            },
            {
                "asset_id": "A002",
                "asset_type": "distribution_center",
                "market": "Chicago, IL",
                "noi": 1_000_000,
                "book_value": 12_000_000,
                "criticality": 0.5,
                "leaseability_score": 0.7,
            },
        ],
        "corporate_state": {
            "net_debt": 500_000_000,
            "ebitda": 200_000_000,
            "interest_expense": 25_000_000,
        },
        "program_type": "slb",
        "program_description": "Raise $10M via SLB with conservative constraints.",
    }


# =============================================================================
# Create Run Tests (POST /api/runs)
# =============================================================================


class TestCreateRunHappyPath:
    """Tests for successful run creation."""

    def test_create_run_returns_201(self, client: TestClient) -> None:
        """Valid request returns 201 Created."""
        request_data = make_valid_request()

        response = client.post("/api/runs", json=request_data)

        assert response.status_code == 201

    def test_create_run_returns_run_id(self, client: TestClient) -> None:
        """Response includes run_id in UUID format."""
        request_data = make_valid_request()

        response = client.post("/api/runs", json=request_data)

        assert response.status_code == 201
        data = response.json()
        assert "run_id" in data
        # UUID format: 8-4-4-4-12 hex characters
        run_id = data["run_id"]
        assert len(run_id) == 36
        assert run_id.count("-") == 4

    def test_create_run_returns_completed_status(self, client: TestClient) -> None:
        """Successful run has status 'completed'."""
        request_data = make_valid_request()

        response = client.post("/api/runs", json=request_data)

        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "completed"

    def test_create_run_includes_response(self, client: TestClient) -> None:
        """Successful run includes full ProgramResponse."""
        request_data = make_valid_request()

        response = client.post("/api/runs", json=request_data)

        assert response.status_code == 201
        data = response.json()
        assert "response" in data
        assert data["response"] is not None

        # Verify response structure
        resp = data["response"]
        assert "selector_spec" in resp
        assert "outcome" in resp
        assert "explanation" in resp
        assert "audit_trace" in resp

    def test_create_run_includes_audit_trace(self, client: TestClient) -> None:
        """Response includes audit_trace with proper structure."""
        request_data = make_valid_request()

        response = client.post("/api/runs", json=request_data)

        assert response.status_code == 201
        audit_trace = response.json()["response"]["audit_trace"]

        assert "entries" in audit_trace
        assert "original_target" in audit_trace
        assert "floor_fraction" in audit_trace
        assert "target_source" in audit_trace
        assert audit_trace["target_source"] in ["user_override", "llm_extraction"]

    def test_create_run_with_fund_id(self, client: TestClient) -> None:
        """Run can be created with optional fund_id."""
        request_data = make_valid_request()

        response = client.post("/api/runs?fund_id=FUND123", json=request_data)

        assert response.status_code == 201
        run_id = response.json()["run_id"]

        # Verify fund_id is stored
        get_response = client.get(f"/api/runs/{run_id}")
        assert get_response.json()["fund_id"] == "FUND123"

    def test_create_run_stores_in_run_store(self, client: TestClient) -> None:
        """Created run is stored in run_store."""
        request_data = make_valid_request()

        response = client.post("/api/runs", json=request_data)

        assert response.status_code == 201
        run_id = response.json()["run_id"]

        # Verify stored in run_store
        record = run_store.get(run_id)
        assert record is not None
        assert record.run_id == run_id
        assert record.response is not None


class TestCreateRunValidationError:
    """Tests for runs that fail due to validation errors."""

    def test_validation_error_returns_201(self, client: TestClient) -> None:
        """Validation error still returns 201 (run record created)."""
        request_data = make_valid_request()
        # Duplicate asset IDs trigger validation error
        request_data["assets"][1]["asset_id"] = "A001"

        response = client.post("/api/runs", json=request_data)

        # 201 because run record is created, even if it failed
        assert response.status_code == 201

    def test_validation_error_has_failed_status(self, client: TestClient) -> None:
        """Failed validation has status 'failed'."""
        request_data = make_valid_request()
        request_data["assets"][1]["asset_id"] = "A001"

        response = client.post("/api/runs", json=request_data)

        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "failed"

    def test_validation_error_includes_error_message(self, client: TestClient) -> None:
        """Failed run includes error message."""
        request_data = make_valid_request()
        request_data["assets"][1]["asset_id"] = "A001"

        response = client.post("/api/runs", json=request_data)

        assert response.status_code == 201
        data = response.json()
        assert "error" in data
        assert data["error"] is not None
        assert "duplicate" in data["error"].lower()

    def test_validation_error_no_response(self, client: TestClient) -> None:
        """Failed run has no response (only error)."""
        request_data = make_valid_request()
        request_data["assets"][1]["asset_id"] = "A001"

        response = client.post("/api/runs", json=request_data)

        assert response.status_code == 201
        # Note: The create_run endpoint returns {"run_id", "status", "error"}
        # for failed runs, not {"response": None}
        data = response.json()
        assert "response" not in data or data.get("response") is None


class TestCreateRunInfrastructureError:
    """Tests for infrastructure errors (503)."""

    def test_infrastructure_error_returns_503(self, client: TestClient) -> None:
        """Infrastructure error returns 503 Service Unavailable."""
        request_data = make_valid_request()

        with patch("app.api.run_program") as mock_run:
            mock_run.side_effect = InfrastructureError("LLM unreachable")

            response = client.post("/api/runs", json=request_data)

        assert response.status_code == 503

    def test_infrastructure_error_detail(self, client: TestClient) -> None:
        """Infrastructure error includes error detail."""
        request_data = make_valid_request()

        with patch("app.api.run_program") as mock_run:
            mock_run.side_effect = InfrastructureError("Cannot reach OpenAI API")

            response = client.post("/api/runs", json=request_data)

        assert response.status_code == 503
        detail = response.json()["detail"]
        assert detail["error"] == "Infrastructure error"
        assert detail["code"] == "LLM_UNAVAILABLE"
        assert "OpenAI" in detail["detail"]

    def test_infrastructure_error_no_run_record(self, client: TestClient) -> None:
        """Infrastructure error does not create run record."""
        request_data = make_valid_request()
        initial_count = len(run_store.list_runs())

        with patch("app.api.run_program") as mock_run:
            mock_run.side_effect = InfrastructureError("LLM down")

            client.post("/api/runs", json=request_data)

        # No run record should have been created
        assert len(run_store.list_runs()) == initial_count


class TestCreateRunPydanticError:
    """Tests for Pydantic validation errors (422)."""

    def test_missing_assets_returns_422(self, client: TestClient) -> None:
        """Missing required field returns 422."""
        request_data = {
            "corporate_state": {
                "net_debt": 500_000_000,
                "ebitda": 200_000_000,
                "interest_expense": 25_000_000,
            },
            "program_type": "slb",
            "program_description": "Raise capital",
        }

        response = client.post("/api/runs", json=request_data)

        assert response.status_code == 422


# =============================================================================
# Get Run Tests (GET /api/runs/{run_id})
# =============================================================================


class TestGetRun:
    """Tests for getting a single run by ID."""

    def test_get_run_returns_200(self, client: TestClient) -> None:
        """Get existing run returns 200."""
        # First create a run
        request_data = make_valid_request()
        create_response = client.post("/api/runs", json=request_data)
        run_id = create_response.json()["run_id"]

        # Then get it
        response = client.get(f"/api/runs/{run_id}")

        assert response.status_code == 200

    def test_get_run_returns_full_record(self, client: TestClient) -> None:
        """Get run returns full record including response."""
        request_data = make_valid_request()
        create_response = client.post("/api/runs", json=request_data)
        run_id = create_response.json()["run_id"]

        response = client.get(f"/api/runs/{run_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["run_id"] == run_id
        assert "program_description" in data
        assert "status" in data
        assert "response" in data
        assert "created_at" in data

    def test_get_run_includes_audit_trace(self, client: TestClient) -> None:
        """Get run includes audit_trace in response."""
        request_data = make_valid_request()
        create_response = client.post("/api/runs", json=request_data)
        run_id = create_response.json()["run_id"]

        response = client.get(f"/api/runs/{run_id}")

        assert response.status_code == 200
        audit_trace = response.json()["response"]["audit_trace"]
        assert audit_trace is not None
        assert "entries" in audit_trace

    def test_get_unknown_run_returns_404(self, client: TestClient) -> None:
        """Get non-existent run returns 404."""
        response = client.get("/api/runs/nonexistent-id")

        assert response.status_code == 404

    def test_get_failed_run(self, client: TestClient) -> None:
        """Get run that failed (validation error)."""
        request_data = make_valid_request()
        request_data["assets"][1]["asset_id"] = "A001"  # Duplicate

        create_response = client.post("/api/runs", json=request_data)
        run_id = create_response.json()["run_id"]

        response = client.get(f"/api/runs/{run_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert data["error"] is not None
        assert data["response"] is None


# =============================================================================
# List Runs Tests (GET /api/runs)
# =============================================================================


class TestListRuns:
    """Tests for listing runs."""

    def test_list_runs_empty(self, client: TestClient) -> None:
        """List runs returns empty list when no runs exist."""
        response = client.get("/api/runs")

        assert response.status_code == 200
        assert response.json() == []

    def test_list_runs_returns_runs(self, client: TestClient) -> None:
        """List runs returns created runs."""
        request_data = make_valid_request()
        client.post("/api/runs", json=request_data)
        client.post("/api/runs", json=request_data)

        response = client.get("/api/runs")

        assert response.status_code == 200
        runs = response.json()
        assert len(runs) == 2

    def test_list_runs_summary_format(self, client: TestClient) -> None:
        """List runs returns summary (not full response)."""
        request_data = make_valid_request()
        client.post("/api/runs", json=request_data)

        response = client.get("/api/runs")

        assert response.status_code == 200
        runs = response.json()
        assert len(runs) == 1

        run = runs[0]
        assert "run_id" in run
        assert "program_description" in run
        assert "status" in run
        assert "created_at" in run
        # Summary should NOT include full response
        assert "response" not in run

    def test_list_runs_filter_by_fund_id(self, client: TestClient) -> None:
        """List runs can filter by fund_id."""
        request_data = make_valid_request()
        client.post("/api/runs?fund_id=FUND_A", json=request_data)
        client.post("/api/runs?fund_id=FUND_A", json=request_data)
        client.post("/api/runs?fund_id=FUND_B", json=request_data)

        # Filter by FUND_A
        response = client.get("/api/runs?fund_id=FUND_A")

        assert response.status_code == 200
        runs = response.json()
        assert len(runs) == 2
        assert all(r["fund_id"] == "FUND_A" for r in runs)

    def test_list_runs_limit(self, client: TestClient) -> None:
        """List runs respects limit parameter."""
        request_data = make_valid_request()
        for _ in range(5):
            client.post("/api/runs", json=request_data)

        response = client.get("/api/runs?limit=3")

        assert response.status_code == 200
        runs = response.json()
        assert len(runs) == 3

    def test_list_runs_most_recent_first(self, client: TestClient) -> None:
        """List runs returns most recent first."""
        request_data = make_valid_request()

        # Create runs with different descriptions
        request_data["program_description"] = "First run"
        client.post("/api/runs", json=request_data)
        request_data["program_description"] = "Second run"
        client.post("/api/runs", json=request_data)
        request_data["program_description"] = "Third run"
        client.post("/api/runs", json=request_data)

        response = client.get("/api/runs")

        assert response.status_code == 200
        runs = response.json()
        assert runs[0]["program_description"] == "Third run"
        assert runs[2]["program_description"] == "First run"


# =============================================================================
# Integration Tests
# =============================================================================


class TestRunsIntegration:
    """Integration tests for the runs API."""

    def test_create_then_get_run(self, client: TestClient) -> None:
        """Create a run and then retrieve it."""
        request_data = make_valid_request()

        # Create
        create_response = client.post("/api/runs", json=request_data)
        assert create_response.status_code == 201
        run_id = create_response.json()["run_id"]

        # Get
        get_response = client.get(f"/api/runs/{run_id}")
        assert get_response.status_code == 200
        assert get_response.json()["run_id"] == run_id

    def test_create_multiple_then_list(self, client: TestClient) -> None:
        """Create multiple runs and list them."""
        request_data = make_valid_request()

        # Create 3 runs
        run_ids = []
        for _ in range(3):
            response = client.post("/api/runs", json=request_data)
            run_ids.append(response.json()["run_id"])

        # List
        list_response = client.get("/api/runs")
        assert list_response.status_code == 200
        runs = list_response.json()
        assert len(runs) == 3

        # All created run_ids should be in the list
        listed_ids = [r["run_id"] for r in runs]
        for run_id in run_ids:
            assert run_id in listed_ids

    def test_infeasible_outcome_is_completed_not_failed(
        self, client: TestClient
    ) -> None:
        """Engine returning INFEASIBLE is still status='completed'."""
        # Use a request that might result in INFEASIBLE
        request_data = make_valid_request()
        request_data["program_description"] = "Raise $1B"  # Very high target

        response = client.post("/api/runs", json=request_data)

        assert response.status_code == 201
        data = response.json()

        # Even if outcome is INFEASIBLE, run status should be 'completed'
        # (INFEASIBLE is a valid business outcome, not a failure)
        assert data["status"] == "completed"
        assert data["response"] is not None
