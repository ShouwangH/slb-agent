"""
Tests for numeric override functionality.

This module tests the numeric constraint overrides in ProgramRequest:
- max_leverage_override: Override the LLM-generated leverage constraint
- min_coverage_override: Override the LLM-generated coverage constraint
- floor_override: Allow the target to be reduced below the LLM-extracted amount

New semantics (post-flip):
- LLM extracts target from natural language description
- Target is SACRED by default (floor = 100%, cannot be reduced)
- floor_override allows flexibility if user is willing to accept less
"""

import pytest

from app.config import DEFAULT_ENGINE_CONFIG
from app.llm.mock import MockLLMClient
from app.models import (
    Asset,
    AssetType,
    CorporateState,
    MarketTier,
    ProgramRequest,
    ProgramType,
)
from app.orchestrator import run_program


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_assets() -> list[Asset]:
    """Simple asset pool for testing (total book value ~$500M)."""
    return [
        Asset(
            asset_id="A1",
            asset_type=AssetType.STORE,
            market="Dallas, TX",
            market_tier=MarketTier.TIER_1,
            noi=5_000_000,
            book_value=100_000_000,
            criticality=0.3,
            leaseability_score=0.8,
        ),
        Asset(
            asset_id="A2",
            asset_type=AssetType.DISTRIBUTION_CENTER,
            market="Austin, TX",
            market_tier=MarketTier.TIER_2,
            noi=4_000_000,
            book_value=80_000_000,
            criticality=0.4,
            leaseability_score=0.9,
        ),
        Asset(
            asset_id="A3",
            asset_type=AssetType.OFFICE,
            market="Houston, TX",
            market_tier=MarketTier.TIER_1,
            noi=6_000_000,
            book_value=120_000_000,
            criticality=0.2,
            leaseability_score=0.7,
        ),
        Asset(
            asset_id="A4",
            asset_type=AssetType.STORE,
            market="San Antonio, TX",
            market_tier=MarketTier.TIER_3,
            noi=3_000_000,
            book_value=60_000_000,
            criticality=0.5,
            leaseability_score=0.6,
        ),
        Asset(
            asset_id="A5",
            asset_type=AssetType.MIXED_USE,
            market="El Paso, TX",
            market_tier=MarketTier.TIER_2,
            noi=7_000_000,
            book_value=140_000_000,
            criticality=0.1,
            leaseability_score=0.95,
        ),
    ]


@pytest.fixture
def corporate_state() -> CorporateState:
    """Corporate state with high leverage that needs deleveraging."""
    return CorporateState(
        net_debt=800_000_000,
        ebitda=150_000_000,
        interest_expense=48_000_000,
        lease_expense=10_000_000,
    )


# =============================================================================
# Core Override Functionality Tests
# =============================================================================


class TestTargetIsSacred:
    """Test that LLM-extracted target is sacred by default (floor = 100%)."""

    def test_target_is_sacred_by_default(
        self, simple_assets: list[Asset], corporate_state: CorporateState
    ) -> None:
        """Without floor_override, the LLM target cannot be reduced (floor = 100%)."""
        mock_llm = MockLLMClient(default_target_amount=100_000_000)

        request = ProgramRequest(
            assets=simple_assets,
            corporate_state=corporate_state,
            program_type=ProgramType.SLB,
            program_description="Raise $100M via SLB to deleverage",
            # No floor_override - target is sacred
        )

        response = run_program(request, mock_llm, DEFAULT_ENGINE_CONFIG)

        # The final spec must keep the full target amount (floor = 100%)
        assert response.selector_spec.target_amount == 100_000_000
        # Verify the LLM was called
        assert mock_llm.call_counts["generate_selector_spec"] == 1

    def test_floor_override_allows_reduction(
        self, simple_assets: list[Asset], corporate_state: CorporateState
    ) -> None:
        """When floor_override is provided, target can be reduced to the floor."""
        mock_llm = MockLLMClient(
            default_target_amount=100_000_000,
            revision_target_reduction=0.30,  # Try to reduce by 30%
        )

        request = ProgramRequest(
            assets=simple_assets,
            corporate_state=corporate_state,
            program_type=ProgramType.SLB,
            program_description="Raise $100M via SLB",
            floor_override=75_000_000,  # Allow reduction to $75M
        )

        response = run_program(request, mock_llm, DEFAULT_ENGINE_CONFIG)

        # Target can be reduced but not below the floor
        assert response.selector_spec.target_amount >= 75_000_000

    def test_floor_override_cannot_exceed_target(
        self, simple_assets: list[Asset], corporate_state: CorporateState
    ) -> None:
        """floor_override cannot be greater than the LLM-extracted target."""
        from fastapi import HTTPException

        mock_llm = MockLLMClient(default_target_amount=50_000_000)

        request = ProgramRequest(
            assets=simple_assets,
            corporate_state=corporate_state,
            program_type=ProgramType.SLB,
            program_description="Raise $50M via SLB",
            floor_override=100_000_000,  # Floor > target - invalid
        )

        # Should raise an error
        with pytest.raises(HTTPException) as exc_info:
            run_program(request, mock_llm, DEFAULT_ENGINE_CONFIG)

        assert exc_info.value.status_code == 400
        assert "Floor" in str(exc_info.value.detail) or "floor" in str(exc_info.value.detail)


class TestLeverageOverride:
    """Test that max_leverage_override prevents LLM from setting leverage constraints."""

    def test_leverage_override_replaces_llm_value(
        self, simple_assets: list[Asset], corporate_state: CorporateState
    ) -> None:
        """When leverage override is provided, it replaces LLM's generated value."""
        mock_llm = MockLLMClient(default_target_amount=200_000_000)

        request = ProgramRequest(
            assets=simple_assets,
            corporate_state=corporate_state,
            program_type=ProgramType.SLB,
            program_description="Raise capital with leverage below 3.5x",
            max_leverage_override=3.5,  # Explicit override
        )

        response = run_program(request, mock_llm, DEFAULT_ENGINE_CONFIG)

        # Should use the override value
        assert response.selector_spec.hard_constraints.max_net_leverage == 3.5

    def test_leverage_override_becomes_immutable(
        self, simple_assets: list[Asset], corporate_state: CorporateState
    ) -> None:
        """Overridden leverage becomes part of immutable hard constraints."""
        mock_llm = MockLLMClient(
            default_target_amount=200_000_000,
            revision_target_reduction=0.20,
        )

        request = ProgramRequest(
            assets=simple_assets,
            corporate_state=corporate_state,
            program_type=ProgramType.SLB,
            program_description="Raise capital with max leverage 3.0x",
            max_leverage_override=3.0,
        )

        response = run_program(request, mock_llm, DEFAULT_ENGINE_CONFIG)

        # The leverage constraint should remain at 3.0 even if revision happened
        assert response.selector_spec.hard_constraints.max_net_leverage == 3.0


class TestCoverageOverride:
    """Test that min_coverage_override prevents LLM from setting coverage constraints."""

    def test_coverage_override_replaces_llm_value(
        self, simple_assets: list[Asset], corporate_state: CorporateState
    ) -> None:
        """When coverage override is provided, it replaces LLM's generated value."""
        mock_llm = MockLLMClient(default_target_amount=200_000_000)

        request = ProgramRequest(
            assets=simple_assets,
            corporate_state=corporate_state,
            program_type=ProgramType.SLB,
            program_description="Raise capital with coverage above 2.5x",
            min_coverage_override=2.5,  # Explicit override
        )

        response = run_program(request, mock_llm, DEFAULT_ENGINE_CONFIG)

        # Should use the override value
        assert response.selector_spec.hard_constraints.min_fixed_charge_coverage == 2.5


# =============================================================================
# Integration Tests
# =============================================================================


class TestOverrideIntegration:
    """Integration tests combining multiple overrides."""

    def test_constraint_overrides_together(
        self, simple_assets: list[Asset], corporate_state: CorporateState
    ) -> None:
        """Test leverage and coverage overrides working together."""
        mock_llm = MockLLMClient(default_target_amount=100_000_000)

        request = ProgramRequest(
            assets=simple_assets,
            corporate_state=corporate_state,
            program_type=ProgramType.SLB,
            program_description="Raise $100M with leverage below 3.5x and coverage above 2.8x",
            max_leverage_override=3.5,
            min_coverage_override=2.8,
        )

        response = run_program(request, mock_llm, DEFAULT_ENGINE_CONFIG)

        # Constraint overrides should be applied
        assert response.selector_spec.hard_constraints.max_net_leverage == 3.5
        assert response.selector_spec.hard_constraints.min_fixed_charge_coverage == 2.8
        # Target is sacred (from LLM)
        assert response.selector_spec.target_amount == 100_000_000

    def test_floor_override_with_constraint_overrides(
        self, simple_assets: list[Asset], corporate_state: CorporateState
    ) -> None:
        """Test floor_override works with constraint overrides."""
        mock_llm = MockLLMClient(default_target_amount=100_000_000)

        request = ProgramRequest(
            assets=simple_assets,
            corporate_state=corporate_state,
            program_type=ProgramType.SLB,
            program_description="Raise $100M via SLB",
            floor_override=80_000_000,  # Allow reduction to $80M
            max_leverage_override=3.5,
        )

        response = run_program(request, mock_llm, DEFAULT_ENGINE_CONFIG)

        # Constraint override should be applied
        assert response.selector_spec.hard_constraints.max_net_leverage == 3.5
        # Target should be at least floor (may not reduce if feasible)
        assert response.selector_spec.target_amount >= 80_000_000

    def test_partial_overrides(
        self, simple_assets: list[Asset], corporate_state: CorporateState
    ) -> None:
        """Test that some overrides can be provided while others use LLM values."""
        mock_llm = MockLLMClient(default_target_amount=100_000_000)

        request = ProgramRequest(
            assets=simple_assets,
            corporate_state=corporate_state,
            program_type=ProgramType.SLB,
            program_description="Raise capital via SLB with leverage below 3.5x",
            max_leverage_override=3.5,  # Only override leverage
            # Leave coverage to LLM, no floor_override (target is sacred)
        )

        response = run_program(request, mock_llm, DEFAULT_ENGINE_CONFIG)

        # Target is sacred (from LLM)
        assert response.selector_spec.target_amount == 100_000_000
        # Leverage uses override
        assert response.selector_spec.hard_constraints.max_net_leverage == 3.5
        # Coverage uses LLM default (3.0 from MockLLMClient)
        assert response.selector_spec.hard_constraints.min_fixed_charge_coverage == 3.0


# =============================================================================
# Behavioral Tests (Bug Fix Verification)
# =============================================================================


class TestBugFixBehavior:
    """Tests that verify target sacrosanctity behavior."""

    def test_llm_target_is_used_exactly(
        self, simple_assets: list[Asset], corporate_state: CorporateState
    ) -> None:
        """
        LLM-extracted target is used exactly (sacred by default).

        The LLM extracts the target from the natural language description,
        and that becomes the sacred target that cannot be reduced without
        an explicit floor_override.
        """
        mock_llm = MockLLMClient(default_target_amount=100_000_000)

        request = ProgramRequest(
            assets=simple_assets,
            corporate_state=corporate_state,
            program_type=ProgramType.SLB,
            program_description="Raise $100M via SLB",
            # No floor_override - target is sacred
        )

        response = run_program(request, mock_llm, DEFAULT_ENGINE_CONFIG)

        # The target should be exactly what the LLM extracted
        assert response.selector_spec.target_amount == 100_000_000

    def test_floor_override_allows_revision_within_bounds(
        self, simple_assets: list[Asset], corporate_state: CorporateState
    ) -> None:
        """
        With floor_override, the revision loop can reduce target to the floor.

        This test verifies that when the user provides a floor_override,
        the revision loop can reduce the target down to that floor.
        """
        mock_llm = MockLLMClient(
            default_target_amount=400_000_000,
            revision_target_reduction=0.20,  # 20% reduction per revision
        )

        request = ProgramRequest(
            assets=simple_assets,
            corporate_state=corporate_state,
            program_type=ProgramType.SLB,
            program_description="Raise $400M via SLB",
            floor_override=300_000_000,  # Allow reduction to $300M (75%)
        )

        response = run_program(request, mock_llm, DEFAULT_ENGINE_CONFIG)

        # The target cannot go below the floor
        assert response.selector_spec.target_amount >= 300_000_000

    def test_prompt_instructs_llm_not_to_freelance(self) -> None:
        """
        Verify that the prompt explicitly tells the LLM not to adjust user numbers.

        This is part of the defense-in-depth approach: both the prompt AND
        the orchestrator enforce the LLM-extracted target as sacred.
        """
        from app.llm.prompts import format_generate_spec_system

        prompt = format_generate_spec_system("slb")

        # Check for the critical instruction
        assert "DO NOT ADJUST USER-PROVIDED NUMBERS" in prompt
        assert "EXACT EXTRACTION" in prompt
        assert "feasibility is handled by a later revision process" in prompt.lower()
