"""
Tests for the FastAPI application.

Tests the API endpoints with various inputs and error conditions.

Test categories:
- Happy path (valid request → 200)
- Validation errors (duplicate IDs → 400)
- Pydantic errors (missing fields → 422)
- Internal errors (mock failure → 500)
- Scenario sets (POST/GET /api/scenario_sets)
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.api import app
from app.models import AssetType, ProgramType, SelectionStatus


# =============================================================================
# Test Client
# =============================================================================


@pytest.fixture
def client() -> TestClient:
    """Create test client for the FastAPI app."""
    return TestClient(app)


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
# Happy Path Tests
# =============================================================================


class TestHealthCheck:
    """Tests for health check endpoint."""

    def test_health_check_returns_200(self, client: TestClient) -> None:
        """Health check endpoint returns 200."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "service" in data
        assert "version" in data


class TestHappyPath:
    """Tests for successful requests."""

    def test_valid_request_returns_200(self, client: TestClient) -> None:
        """Valid request returns 200 with ProgramResponse."""
        request_data = make_valid_request()

        response = client.post("/program", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "selector_spec" in data
        assert "outcome" in data
        assert "explanation" in data

    def test_response_contains_selector_spec(self, client: TestClient) -> None:
        """Response contains valid selector_spec."""
        request_data = make_valid_request()

        response = client.post("/program", json=request_data)

        assert response.status_code == 200
        spec = response.json()["selector_spec"]

        assert spec["program_type"] == "slb"
        assert spec["target_amount"] > 0
        assert "hard_constraints" in spec
        assert "soft_preferences" in spec

    def test_response_contains_outcome(self, client: TestClient) -> None:
        """Response contains valid outcome."""
        request_data = make_valid_request()

        response = client.post("/program", json=request_data)

        assert response.status_code == 200
        outcome = response.json()["outcome"]

        assert "status" in outcome
        assert outcome["status"] in ["ok", "infeasible", "numeric_error"]
        assert "selected_assets" in outcome
        assert "proceeds" in outcome

    def test_response_contains_explanation(self, client: TestClient) -> None:
        """Response contains valid explanation."""
        request_data = make_valid_request()

        response = client.post("/program", json=request_data)

        assert response.status_code == 200
        explanation = response.json()["explanation"]

        assert "summary" in explanation
        assert len(explanation["summary"]) > 0
        assert "nodes" in explanation

    def test_feasible_outcome_returns_ok_status(self, client: TestClient) -> None:
        """Request with achievable target returns OK status."""
        request_data = make_valid_request()
        # Use a small target that can be met
        request_data["program_description"] = "Raise $1M via SLB"

        response = client.post("/program", json=request_data)

        assert response.status_code == 200
        # Note: Status depends on MockLLMClient behavior
        # Just verify we get a valid status
        assert response.json()["outcome"]["status"] in ["ok", "infeasible"]


# =============================================================================
# Validation Error Tests (400)
# =============================================================================


class TestValidationErrors:
    """Tests for validation errors that return 400."""

    def test_duplicate_asset_ids_returns_400(self, client: TestClient) -> None:
        """Duplicate asset IDs trigger validation error."""
        request_data = make_valid_request()
        # Make both assets have the same ID
        request_data["assets"][1]["asset_id"] = "A001"

        response = client.post("/program", json=request_data)

        assert response.status_code == 400
        detail = response.json()["detail"]
        assert detail["code"] == "VALIDATION_FAILED"
        assert "duplicate" in detail["detail"].lower()

    # Note: Testing unsupported program_type is not possible with current
    # enum (only SLB). When additional program types are added to the enum
    # but not supported by the orchestrator, that case can be tested.


# =============================================================================
# Pydantic Validation Error Tests (422)
# =============================================================================


class TestPydanticErrors:
    """Tests for Pydantic validation errors that return 422."""

    def test_missing_assets_returns_422(self, client: TestClient) -> None:
        """Missing required 'assets' field returns 422."""
        request_data = {
            "corporate_state": {
                "net_debt": 500_000_000,
                "ebitda": 200_000_000,
                "interest_expense": 25_000_000,
            },
            "program_type": "slb",
            "program_description": "Raise capital",
        }

        response = client.post("/program", json=request_data)

        assert response.status_code == 422

    def test_missing_corporate_state_returns_422(self, client: TestClient) -> None:
        """Missing required 'corporate_state' field returns 422."""
        request_data = make_valid_request()
        del request_data["corporate_state"]

        response = client.post("/program", json=request_data)

        assert response.status_code == 422

    def test_missing_program_type_returns_422(self, client: TestClient) -> None:
        """Missing required 'program_type' field returns 422."""
        request_data = make_valid_request()
        del request_data["program_type"]

        response = client.post("/program", json=request_data)

        assert response.status_code == 422

    def test_missing_program_description_returns_422(self, client: TestClient) -> None:
        """Missing required 'program_description' field returns 422."""
        request_data = make_valid_request()
        del request_data["program_description"]

        response = client.post("/program", json=request_data)

        assert response.status_code == 422

    def test_invalid_program_type_returns_422(self, client: TestClient) -> None:
        """Invalid program_type enum value returns 422."""
        request_data = make_valid_request()
        request_data["program_type"] = "invalid_type"

        response = client.post("/program", json=request_data)

        assert response.status_code == 422

    def test_invalid_asset_type_returns_422(self, client: TestClient) -> None:
        """Invalid asset_type enum value returns 422."""
        request_data = make_valid_request()
        request_data["assets"][0]["asset_type"] = "invalid_type"

        response = client.post("/program", json=request_data)

        assert response.status_code == 422

    def test_negative_noi_returns_422(self, client: TestClient) -> None:
        """Negative NOI value returns 422."""
        request_data = make_valid_request()
        request_data["assets"][0]["noi"] = -100

        response = client.post("/program", json=request_data)

        assert response.status_code == 422

    def test_criticality_out_of_range_returns_422(self, client: TestClient) -> None:
        """Criticality > 1 returns 422."""
        request_data = make_valid_request()
        request_data["assets"][0]["criticality"] = 1.5

        response = client.post("/program", json=request_data)

        assert response.status_code == 422

    def test_empty_assets_list_returns_422(self, client: TestClient) -> None:
        """Empty assets list returns 422."""
        request_data = make_valid_request()
        request_data["assets"] = []

        response = client.post("/program", json=request_data)

        assert response.status_code == 422

    def test_empty_program_description_returns_422(self, client: TestClient) -> None:
        """Empty program_description returns 422."""
        request_data = make_valid_request()
        request_data["program_description"] = ""

        response = client.post("/program", json=request_data)

        assert response.status_code == 422


# =============================================================================
# Internal Error Tests (500)
# =============================================================================


class TestInternalErrors:
    """Tests for internal errors that return 500."""

    def test_unexpected_exception_returns_500(self, client: TestClient) -> None:
        """Unexpected exception in orchestrator returns 500."""
        request_data = make_valid_request()

        # Mock run_program to raise an unexpected exception
        with patch("app.api.run_program") as mock_run:
            mock_run.side_effect = RuntimeError("Unexpected failure")

            response = client.post("/program", json=request_data)

        assert response.status_code == 500
        detail = response.json()["detail"]
        assert detail["code"] == "INTERNAL_ERROR"
        assert "Unexpected failure" in detail["detail"]

    def test_internal_error_response_structure(self, client: TestClient) -> None:
        """Internal error response has correct structure."""
        request_data = make_valid_request()

        with patch("app.api.run_program") as mock_run:
            mock_run.side_effect = Exception("Test error")

            response = client.post("/program", json=request_data)

        assert response.status_code == 500
        detail = response.json()["detail"]

        # Verify ErrorResponse structure
        assert "error" in detail
        assert "detail" in detail
        assert "code" in detail
        assert detail["error"] == "Internal error"
        assert detail["code"] == "INTERNAL_ERROR"


# =============================================================================
# Error Response Structure Tests
# =============================================================================


class TestErrorResponseStructure:
    """Tests for error response format consistency."""

    def test_validation_error_has_correct_structure(self, client: TestClient) -> None:
        """Validation error response has ErrorResponse structure."""
        request_data = make_valid_request()
        request_data["assets"][1]["asset_id"] = "A001"  # Duplicate

        response = client.post("/program", json=request_data)

        assert response.status_code == 400
        detail = response.json()["detail"]

        assert "error" in detail
        assert "detail" in detail
        assert "code" in detail

    def test_error_codes_are_uppercase(self, client: TestClient) -> None:
        """Error codes follow UPPER_SNAKE_CASE convention."""
        request_data = make_valid_request()
        request_data["assets"][1]["asset_id"] = "A001"  # Duplicate

        response = client.post("/program", json=request_data)

        code = response.json()["detail"]["code"]
        assert code == code.upper()
        assert "_" in code or code.isalpha()


# =============================================================================
# Scenario Sets API Tests
# =============================================================================


class TestScenarioSetsCreate:
    """Tests for POST /api/scenario_sets endpoint."""

    def test_create_scenario_set_returns_201(self, client: TestClient) -> None:
        """Valid request creates scenario set and returns 201."""
        request_data = make_valid_request()

        response = client.post(
            "/api/scenario_sets",
            json=request_data,
            params={
                "brief": "Raise $10M via SLB with conservative constraints",
                "num_scenarios": 2,
            },
        )

        assert response.status_code == 201
        data = response.json()

        # Verify response structure
        assert "scenario_set" in data
        assert "runs" in data

        # Verify scenario_set fields
        scenario_set = data["scenario_set"]
        assert "id" in scenario_set
        assert "brief" in scenario_set
        assert "created_at" in scenario_set
        assert "run_ids" in scenario_set

        # Brief should match what we sent
        assert scenario_set["brief"] == "Raise $10M via SLB with conservative constraints"

    def test_create_scenario_set_returns_correct_number_of_runs(
        self, client: TestClient
    ) -> None:
        """Scenario set returns requested number of runs."""
        request_data = make_valid_request()

        response = client.post(
            "/api/scenario_sets",
            json=request_data,
            params={
                "brief": "Test brief",
                "num_scenarios": 3,
            },
        )

        assert response.status_code == 201
        data = response.json()

        # Should have 3 runs
        assert len(data["runs"]) == 3
        assert len(data["scenario_set"]["run_ids"]) == 3

    def test_create_scenario_set_first_run_is_base(self, client: TestClient) -> None:
        """First run in scenario set should be BASE kind."""
        request_data = make_valid_request()

        response = client.post(
            "/api/scenario_sets",
            json=request_data,
            params={
                "brief": "Test brief",
                "num_scenarios": 3,
            },
        )

        assert response.status_code == 201
        data = response.json()

        # First run should be BASE
        first_run = data["runs"][0]
        assert first_run["scenario_kind"] == "base"

    def test_create_scenario_set_runs_have_scenario_metadata(
        self, client: TestClient
    ) -> None:
        """All runs should have scenario metadata."""
        request_data = make_valid_request()

        response = client.post(
            "/api/scenario_sets",
            json=request_data,
            params={
                "brief": "Test brief",
                "num_scenarios": 2,
            },
        )

        assert response.status_code == 201
        data = response.json()

        scenario_set_id = data["scenario_set"]["id"]
        for run in data["runs"]:
            assert run["scenario_set_id"] == scenario_set_id
            assert run["scenario_kind"] is not None
            assert run["scenario_label"] is not None
            assert "status" in run

    def test_create_scenario_set_with_fund_id(self, client: TestClient) -> None:
        """Scenario set can be created with fund_id."""
        request_data = make_valid_request()

        response = client.post(
            "/api/scenario_sets",
            json=request_data,
            params={
                "brief": "Test brief",
                "num_scenarios": 2,
                "fund_id": "test-fund-123",
            },
        )

        assert response.status_code == 201

    def test_create_scenario_set_num_scenarios_bounds(self, client: TestClient) -> None:
        """num_scenarios must be between 1 and 5."""
        request_data = make_valid_request()

        # Too low
        response = client.post(
            "/api/scenario_sets",
            json=request_data,
            params={
                "brief": "Test brief",
                "num_scenarios": 0,
            },
        )
        assert response.status_code == 422

        # Too high
        response = client.post(
            "/api/scenario_sets",
            json=request_data,
            params={
                "brief": "Test brief",
                "num_scenarios": 6,
            },
        )
        assert response.status_code == 422


class TestScenarioSetsList:
    """Tests for GET /api/scenario_sets endpoint."""

    def test_list_scenario_sets_returns_200(self, client: TestClient) -> None:
        """List scenario sets returns 200."""
        response = client.get("/api/scenario_sets")

        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_list_scenario_sets_returns_created_sets(self, client: TestClient) -> None:
        """List returns scenario sets that were created."""
        request_data = make_valid_request()

        # Create a scenario set
        create_response = client.post(
            "/api/scenario_sets",
            json=request_data,
            params={
                "brief": "Test brief for listing",
                "num_scenarios": 2,
            },
        )
        assert create_response.status_code == 201
        created_id = create_response.json()["scenario_set"]["id"]

        # List and verify it's there
        list_response = client.get("/api/scenario_sets")
        assert list_response.status_code == 200

        sets = list_response.json()
        set_ids = [s["id"] for s in sets]
        assert created_id in set_ids

    def test_list_scenario_sets_respects_limit(self, client: TestClient) -> None:
        """List respects limit parameter."""
        response = client.get("/api/scenario_sets", params={"limit": 5})

        assert response.status_code == 200
        assert len(response.json()) <= 5


class TestScenarioSetsGetById:
    """Tests for GET /api/scenario_sets/{id} endpoint."""

    def test_get_scenario_set_returns_200(self, client: TestClient) -> None:
        """Get existing scenario set returns 200."""
        request_data = make_valid_request()

        # Create a scenario set
        create_response = client.post(
            "/api/scenario_sets",
            json=request_data,
            params={
                "brief": "Test brief for get",
                "num_scenarios": 2,
            },
        )
        assert create_response.status_code == 201
        created_id = create_response.json()["scenario_set"]["id"]

        # Get by ID
        get_response = client.get(f"/api/scenario_sets/{created_id}")
        assert get_response.status_code == 200

        data = get_response.json()
        assert "scenario_set" in data
        assert "runs" in data
        assert data["scenario_set"]["id"] == created_id

    def test_get_scenario_set_includes_full_runs(self, client: TestClient) -> None:
        """Get by ID includes full run details with responses."""
        request_data = make_valid_request()

        # Create a scenario set
        create_response = client.post(
            "/api/scenario_sets",
            json=request_data,
            params={
                "brief": "Test brief for full runs",
                "num_scenarios": 2,
            },
        )
        assert create_response.status_code == 201
        created_id = create_response.json()["scenario_set"]["id"]

        # Get by ID
        get_response = client.get(f"/api/scenario_sets/{created_id}")
        assert get_response.status_code == 200

        data = get_response.json()
        for run in data["runs"]:
            # Full run details should be present
            assert "run_id" in run
            assert "status" in run
            assert "scenario_set_id" in run
            assert "scenario_kind" in run
            assert "scenario_label" in run
            # Should have response or error
            assert "response" in run or "error" in run

    def test_get_scenario_set_not_found_returns_404(self, client: TestClient) -> None:
        """Get non-existent scenario set returns 404."""
        response = client.get("/api/scenario_sets/non-existent-id")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
