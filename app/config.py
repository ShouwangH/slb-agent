"""
Engine configuration for SLB Agent.

This module contains EngineConfig and default values as defined in DESIGN.md Section 4.8 and 5.1.
All configurable parameters live here - no magic numbers in engine code.
"""

from typing import Optional

from pydantic import BaseModel, Field

from app.models import AssetType, MarketTier


# =============================================================================
# Default Cap Rate Curve (Section 5.1)
# =============================================================================

DEFAULT_CAP_RATE_CURVE: dict[AssetType, dict[MarketTier, float]] = {
    AssetType.STORE: {
        MarketTier.TIER_1: 0.055,
        MarketTier.TIER_2: 0.065,
        MarketTier.TIER_3: 0.075,
    },
    AssetType.DISTRIBUTION_CENTER: {
        MarketTier.TIER_1: 0.045,
        MarketTier.TIER_2: 0.055,
        MarketTier.TIER_3: 0.065,
    },
    AssetType.OFFICE: {
        MarketTier.TIER_1: 0.060,
        MarketTier.TIER_2: 0.070,
        MarketTier.TIER_3: 0.080,
    },
    AssetType.MIXED_USE: {
        MarketTier.TIER_1: 0.055,
        MarketTier.TIER_2: 0.065,
        MarketTier.TIER_3: 0.075,
    },
    AssetType.OTHER: {
        MarketTier.TIER_1: 0.070,
        MarketTier.TIER_2: 0.080,
        MarketTier.TIER_3: 0.090,
    },
}


# =============================================================================
# Engine Configuration (Section 4.8)
# =============================================================================


class EngineConfig(BaseModel):
    """
    All configurable parameters for the economics engine.

    Passed explicitly to engine functions rather than hardcoded.
    Enables scenario analysis, unit testing with controlled parameters,
    and client-specific covenant configurations.
    """

    # Cap rate curve by asset type and market tier
    cap_rate_curve: dict[AssetType, dict[MarketTier, float]] = Field(
        default_factory=lambda: DEFAULT_CAP_RATE_CURVE.copy()
    )

    # Default market tier when not specified on asset
    default_market_tier: MarketTier = Field(
        default=MarketTier.TIER_2, description="Fallback tier for assets without market_tier"
    )

    # Transaction costs
    transaction_haircut: float = Field(
        default=0.025, ge=0, lt=1, description="Transaction cost as fraction of market value"
    )

    # Debt assumptions
    avg_cost_of_debt: float = Field(
        default=0.06, gt=0, lt=1, description="Blended cost of debt for interest calculations"
    )

    # SLB rent modeling
    slb_rent_multiplier: float = Field(
        default=1.0,
        gt=0,
        description="slb_rent = noi * multiplier (0.9-1.2 typical)",
    )

    # Selection tolerance
    target_tolerance: float = Field(
        default=0.02, ge=0, lt=1, description="Fraction under-target that is acceptable"
    )

    # Criticality threshold for "critical asset" classification
    criticality_threshold: float = Field(
        default=0.7, ge=0, le=1, description="Assets with criticality > threshold are critical"
    )

    # Numeric guardrails
    epsilon: float = Field(
        default=1e-9, gt=0, description="Denominators below this are treated as zero"
    )

    # Default hard constraints (used when not specified in SelectorSpec)
    default_max_net_leverage: float = Field(
        default=4.0, gt=0, description="Default max net_debt / ebitda"
    )
    default_min_fixed_charge_coverage: float = Field(
        default=3.0, gt=0, description="Default min ebitda / (interest + leases)"
    )
    default_min_interest_coverage: Optional[float] = Field(
        default=None, description="Default min ebitda / interest (None = not enforced)"
    )

    # Iteration defaults
    default_max_iterations: int = Field(
        default=3, ge=1, le=10, description="Default max agentic loop iterations"
    )


# =============================================================================
# Default Instance (Section 5.1)
# =============================================================================

DEFAULT_ENGINE_CONFIG = EngineConfig()


# =============================================================================
# Revision Policy Configuration
# =============================================================================


class RevisionPolicyConfig(BaseModel):
    """
    Configuration for the revision policy that constrains LLM spec changes.

    These constants control how much the LLM can adjust spec parameters during
    the agentic revision loop, ensuring bounded exploration.
    """

    # Target amount bounds
    global_target_floor_fraction: float = Field(
        default=0.75,
        gt=0,
        le=1,
        description="Minimum target as fraction of original (prevents deviation from user intent)",
    )
    max_per_iteration_target_drop_fraction: float = Field(
        default=0.20,
        gt=0,
        lt=1,
        description="Maximum target reduction per iteration (ensures gradual exploration)",
    )

    # Asset filter relaxation bounds
    max_criticality_step: float = Field(
        default=0.1, gt=0, description="Max increase in max_criticality per iteration"
    )
    max_criticality_ceiling: float = Field(
        default=0.8, gt=0, le=1, description="Absolute ceiling for max_criticality"
    )
    max_criticality_default_baseline: float = Field(
        default=0.5,
        gt=0,
        le=1,
        description="Baseline for max_criticality when previous value was None",
    )

    min_leaseability_step: float = Field(
        default=0.1, gt=0, description="Max decrease in min_leaseability_score per iteration"
    )
    min_leaseability_floor: float = Field(
        default=0.2, ge=0, lt=1, description="Absolute floor for min_leaseability_score"
    )
    min_leaseability_default_baseline: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Baseline for min_leaseability_score when previous value was None",
    )


DEFAULT_REVISION_POLICY_CONFIG = RevisionPolicyConfig()


# =============================================================================
# Prompt / LLM Spec Generation Configuration
# =============================================================================


class PromptConfig(BaseModel):
    """
    Configuration for prompt-based spec generation defaults.

    Controls fallback values when the LLM must infer parameters not
    explicitly stated in the program description.
    """

    # Default target amount estimation (when user doesn't specify)
    default_target_fraction_of_portfolio: float = Field(
        default=0.175,
        gt=0,
        lt=1,
        description="Estimated target as fraction of portfolio value (15-20% typical)",
    )
    default_portfolio_cap_rate: float = Field(
        default=0.065,
        gt=0,
        lt=1,
        description="Cap rate for estimating portfolio value from NOI when needed",
    )


DEFAULT_PROMPT_CONFIG = PromptConfig()


# =============================================================================
# LLM Configuration (Section 9.1)
# =============================================================================


class LLMConfig(BaseModel):
    """
    Configuration for LLM client.

    Controls model selection, temperature, and token limits.
    """

    model: str = Field(
        default="gpt-5.2",
        description="OpenAI model to use (gpt-4o is the latest and most capable)",
    )
    temperature: float = Field(
        default=0.2,
        ge=0,
        le=2,
        description="Sampling temperature (low for structured output)",
    )
    max_tokens: int = Field(
        default=2000,
        gt=0,
        description="Maximum tokens in response",
    )


DEFAULT_LLM_CONFIG = LLMConfig()
