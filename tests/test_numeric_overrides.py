"""
Tests for numeric override functionality (bug fix for LLM revising user targets).

This module tests the fix for the bug where the LLM would "helpfully" revise
numeric targets (e.g., requested raise amount) down during the initial planning step.

The fix ensures:
1. ProgramRequest can accept explicit numeric overrides
2. Orchestrator clamps initial spec to those overrides
3. Revision policy respects the 75% global floor
4. Prompts instruct LLM not to freelance on numbers
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


class TestTargetAmountOverride:
    """Test that target_amount_override prevents LLM from reducing user intent."""

    def test_override_enforces_75_percent_floor(
        self, simple_assets: list[Asset], corporate_state: CorporateState
    ) -> None:
        """When override is provided, the final target cannot go below 75% of override."""
        # Create a mock LLM that would generate a much lower target
        mock_llm = MockLLMClient(default_target_amount=30_000_000)

        # User explicitly requests $100M
        request = ProgramRequest(
            assets=simple_assets,
            corporate_state=corporate_state,
            program_type=ProgramType.SLB,
            program_description="Raise $100M via SLB to deleverage",
            target_amount_override=100_000_000,  # Explicit override
        )

        response = run_program(request, mock_llm, DEFAULT_ENGINE_CONFIG)

        # The final spec must respect the 75% floor of the override (75M), not the LLM's 30M
        assert response.selector_spec.target_amount >= 75_000_000
        # Verify the LLM was called (but its target was overridden)
        assert mock_llm.call_counts["generate_selector_spec"] == 1

    def test_no_override_uses_llm_generated_target(
        self, simple_assets: list[Asset], corporate_state: CorporateState
    ) -> None:
        """When no override is provided, use the LLM's generated target."""
        mock_llm = MockLLMClient(default_target_amount=60_000_000)

        request = ProgramRequest(
            assets=simple_assets,
            corporate_state=corporate_state,
            program_type=ProgramType.SLB,
            program_description="Raise capital via SLB",
            # No override provided
        )

        response = run_program(request, mock_llm, DEFAULT_ENGINE_CONFIG)

        # Should use the LLM's generated target (allowing for revision)
        # The final amount will be at least 75% of LLM's generated value (45M)
        assert response.selector_spec.target_amount >= 45_000_000

    def test_override_becomes_original_target_for_revision_policy(
        self, simple_assets: list[Asset], corporate_state: CorporateState
    ) -> None:
        """The override becomes the original_target used by the revision policy floor."""
        # Create a mock that will reduce target on revision
        mock_llm = MockLLMClient(
            default_target_amount=100_000_000,  # Different from override
            revision_target_reduction=0.50,  # Try to cut target by 50%
        )

        # User explicitly requests $400M
        request = ProgramRequest(
            assets=simple_assets,
            corporate_state=corporate_state,
            program_type=ProgramType.SLB,
            program_description="Raise $400M via SLB",
            target_amount_override=400_000_000,
        )

        response = run_program(request, mock_llm, DEFAULT_ENGINE_CONFIG)

        # Even if revision happened, target cannot go below 75% of $400M = $300M
        # (The actual behavior depends on feasibility, but the floor is enforced)
        assert response.selector_spec.target_amount >= 300_000_000


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

    def test_all_overrides_together(
        self, simple_assets: list[Asset], corporate_state: CorporateState
    ) -> None:
        """Test all three overrides working together."""
        mock_llm = MockLLMClient(default_target_amount=50_000_000)

        request = ProgramRequest(
            assets=simple_assets,
            corporate_state=corporate_state,
            program_type=ProgramType.SLB,
            program_description="Raise $100M with leverage below 3.5x and coverage above 2.8x",
            target_amount_override=100_000_000,
            max_leverage_override=3.5,
            min_coverage_override=2.8,
        )

        response = run_program(request, mock_llm, DEFAULT_ENGINE_CONFIG)

        # All overrides should be applied (allowing for feasibility adjustments)
        # Target may be adjusted down slightly if needed, but should be close
        assert response.selector_spec.target_amount >= 75_000_000  # At least 75% of override
        assert response.selector_spec.hard_constraints.max_net_leverage == 3.5
        assert response.selector_spec.hard_constraints.min_fixed_charge_coverage == 2.8

    def test_partial_overrides(
        self, simple_assets: list[Asset], corporate_state: CorporateState
    ) -> None:
        """Test that some overrides can be provided while others use LLM values."""
        mock_llm = MockLLMClient(default_target_amount=50_000_000)

        request = ProgramRequest(
            assets=simple_assets,
            corporate_state=corporate_state,
            program_type=ProgramType.SLB,
            program_description="Raise capital via SLB",
            target_amount_override=100_000_000,  # Only override target
            # Leave leverage and coverage to LLM
        )

        response = run_program(request, mock_llm, DEFAULT_ENGINE_CONFIG)

        # Target should use override (allowing for feasibility)
        assert response.selector_spec.target_amount >= 75_000_000
        # Leverage should use LLM default (4.0 from MockLLMClient)
        assert response.selector_spec.hard_constraints.max_net_leverage == 4.0


# =============================================================================
# Behavioral Tests (Bug Fix Verification)
# =============================================================================


class TestBugFixBehavior:
    """Tests that verify the specific bug is fixed."""

    def test_llm_cannot_reduce_user_specified_target_in_initial_spec(
        self, simple_assets: list[Asset], corporate_state: CorporateState
    ) -> None:
        """
        Bug fix verification: LLM cannot "helpfully" reduce user's target during initial spec.

        Before fix: User says "raise $100M", LLM generates spec with $60M "to be realistic"
        After fix: Initial spec starts at exactly $100M, revision loop handles feasibility
        """
        # Create a mock that would generate a lower target
        mock_llm = MockLLMClient(default_target_amount=60_000_000)

        request = ProgramRequest(
            assets=simple_assets,
            corporate_state=corporate_state,
            program_type=ProgramType.SLB,
            program_description="Raise $100M via SLB",
            target_amount_override=100_000_000,
        )

        response = run_program(request, mock_llm, DEFAULT_ENGINE_CONFIG)

        # The initial spec starts at $100M, not the LLM's "realistic" $60M
        # The final spec may be lower due to revision, but it will be at least 75% of $100M
        # This verifies the override worked and the revision policy floor is respected
        assert response.selector_spec.target_amount >= 75_000_000

        # This is the key behavioral change: the LLM's attempt to be "helpful"
        # is overridden by the explicit user intent, and the floor ensures we
        # don't deviate too far from the user's stated goal

    def test_revision_loop_can_adjust_down_from_override_within_bounds(
        self, simple_assets: list[Asset], corporate_state: CorporateState
    ) -> None:
        """
        The revision loop can still adjust the target down within policy bounds.

        This test verifies that while the initial spec is clamped to user intent,
        the revision loop can still operate normally if the target is infeasible.
        """
        # Note: This test documents expected behavior but may not actually trigger
        # revision in this simple scenario. It's here for behavioral documentation.
        mock_llm = MockLLMClient(
            default_target_amount=100_000_000,
            revision_target_reduction=0.20,  # 20% reduction per revision
        )

        request = ProgramRequest(
            assets=simple_assets,
            corporate_state=corporate_state,
            program_type=ProgramType.SLB,
            program_description="Raise $400M via SLB",
            target_amount_override=400_000_000,
        )

        response = run_program(request, mock_llm, DEFAULT_ENGINE_CONFIG)

        # The initial spec starts at $400M (override)
        # If revision happens, it can drop by up to 20% per iteration
        # But it cannot go below 75% of $400M = $300M (global floor)
        assert response.selector_spec.target_amount >= 300_000_000

    def test_prompt_instructs_llm_not_to_freelance(self) -> None:
        """
        Verify that the prompt explicitly tells the LLM not to adjust user numbers.

        This is part of the defense-in-depth approach: both the prompt AND
        the orchestrator clamping ensure the LLM doesn't freelance.
        """
        from app.llm.prompts import format_generate_spec_system

        prompt = format_generate_spec_system("slb")

        # Check for the critical instruction
        assert "DO NOT ADJUST USER-PROVIDED NUMBERS" in prompt
        assert "EXACT EXTRACTION" in prompt
        assert "feasibility is handled by a later revision process" in prompt.lower()
