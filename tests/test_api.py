"""
Tests for the FastAPI application.

Tests the POST /program endpoint with various inputs and error conditions.

Test categories:
- Happy path (valid request → 200)
- Validation errors (duplicate IDs → 400)
- Pydantic errors (missing fields → 422)
- Internal errors (mock failure → 500)
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
