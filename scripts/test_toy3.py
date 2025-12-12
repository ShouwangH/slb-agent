"""
Quick test of Toy 3 scenario with OpenAI LLM.

Tests whether the newer model correctly extracts $60M target.
"""

import os
import sys

from app.config import EngineConfig
from app.llm.openai_client import OpenAILLMClient
from app.models import (
    Asset,
    AssetType,
    CorporateState,
    MarketTier,
    ProgramRequest,
    ProgramType,
)
from app.orchestrator import run_program


# Toy scenario assets
TOY_CONFIG = EngineConfig(
    cap_rate_curve={
        AssetType.STORE: {
            MarketTier.TIER_1: 0.06,
            MarketTier.TIER_2: 0.065,
        },
        AssetType.OFFICE: {
            MarketTier.TIER_1: 0.055,
        },
        AssetType.DISTRIBUTION_CENTER: {MarketTier.TIER_1: 0.05, MarketTier.TIER_2: 0.06},
        AssetType.MIXED_USE: {MarketTier.TIER_1: 0.06, MarketTier.TIER_2: 0.07},
        AssetType.OTHER: {MarketTier.TIER_1: 0.07, MarketTier.TIER_2: 0.08},
    },
    transaction_haircut=0.025,
    slb_rent_multiplier=1.0,
    avg_cost_of_debt=0.06,
    epsilon=1e-6,
    criticality_threshold=0.7,
    target_tolerance=0.05,
)

TOY_CORPORATE = CorporateState(
    net_debt=100_000_000,
    ebitda=40_000_000,
    interest_expense=6_000_000,
    lease_expense=0,
)

TOY_ASSETS = [
    Asset(
        asset_id="HQ",
        asset_type=AssetType.OFFICE,
        market="New York, NY",
        market_tier=MarketTier.TIER_1,
        noi=4_000_000,
        book_value=50_000_000,
        criticality=0.95,
        leaseability_score=0.30,
    ),
    Asset(
        asset_id="STORE_NYC",
        asset_type=AssetType.STORE,
        market="New York, NY",
        market_tier=MarketTier.TIER_1,
        noi=2_000_000,
        book_value=25_000_000,
        criticality=0.40,
        leaseability_score=0.90,
    ),
    Asset(
        asset_id="STORE_AUSTIN",
        asset_type=AssetType.STORE,
        market="Austin, TX",
        market_tier=MarketTier.TIER_2,
        noi=1_500_000,
        book_value=18_000_000,
        criticality=0.30,
        leaseability_score=0.80,
    ),
]


def main():
    print("=" * 80)
    print("TESTING TOY 3 SCENARIO WITH NEWER MODEL")
    print("=" * 80)

    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n❌ ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    llm = OpenAILLMClient()
    print(f"\n✅ Using model: {llm.config.model}")

    # Toy 3: Target too high, should extract $60M exactly
    print("\n" + "-" * 80)
    print("Toy Scenario 3: $60M target (max achievable ~$55M with stores)")
    print("-" * 80)

    request = ProgramRequest(
        assets=TOY_ASSETS,
        corporate_state=TOY_CORPORATE,
        program_type=ProgramType.SLB,
        program_description="""
        Raise $60M via sale-leaseback. Focus on retail stores only.
        Target leverage below 4.0x, fixed charge coverage above 2.5x.
        Keep critical asset concentration below 70%.
        (Note: Maximum achievable with stores is ~$55M, so this will need revision)
        """,
    )

    try:
        response = run_program(request, llm, TOY_CONFIG)

        spec = response.selector_spec
        outcome = response.outcome

        print(f"\n📊 LLM-Generated Spec:")
        print(f"  Target Amount: ${spec.target_amount:,.0f}")
        print(f"  Expected: $60,000,000")

        if spec.target_amount == 60_000_000:
            print("  ✅ CORRECT - LLM extracted $60M exactly!")
        else:
            print(f"  ❌ INCORRECT - LLM extracted ${spec.target_amount:,.0f} instead of $60M")

        print(f"\n📈 Outcome:")
        print(f"  Status: {outcome.status.value.upper()}")
        print(f"  Proceeds: ${outcome.proceeds:,.0f}")
        print(f"  Assets Selected: {len(outcome.selected_assets)}")

        if outcome.selected_assets:
            print(f"\n  Selected Assets:")
            for selection in outcome.selected_assets:
                print(f"    - {selection.asset.asset_id}: ${selection.proceeds:,.0f}")

        if outcome.violations:
            print(f"\n  Violations: {[v.code for v in outcome.violations]}")

        print(f"\n💬 Summary:")
        print(f"  {response.explanation.summary}")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
