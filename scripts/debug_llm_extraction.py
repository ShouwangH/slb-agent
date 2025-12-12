"""
Debug LLM extraction behavior.

Shows exact prompts sent to LLM and responses received to understand
why the model reduces target amounts.
"""

import json
import os
import sys

from app.config import DEFAULT_ENGINE_CONFIG, DEFAULT_PROMPT_CONFIG
from app.llm.openai_client import OpenAILLMClient
from app.llm.prompts import format_generate_spec_system, format_generate_spec_user
from app.models import (
    Asset,
    AssetType,
    CorporateState,
    MarketTier,
    ProgramType,
)
from app.orchestrator import summarize_assets


# Toy scenario assets
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
    print("DEBUG: LLM EXTRACTION BEHAVIOR")
    print("=" * 80)

    if not os.environ.get("OPENAI_API_KEY"):
        print("\n❌ ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    llm = OpenAILLMClient()
    print(f"\n✅ Using model: {llm.config.model}")

    # Test case: User says "$60M"
    program_description = """
        Raise $60M via sale-leaseback. Focus on retail stores only.
        Target leverage below 4.0x, fixed charge coverage above 2.5x.
        Keep critical asset concentration below 70%.
        """

    print("\n" + "-" * 80)
    print("TEST CASE: User says 'Raise $60M'")
    print("-" * 80)

    # Generate asset summary
    asset_summary = summarize_assets(TOY_ASSETS)

    print("\n📋 ASSET SUMMARY:")
    print(asset_summary)

    # Generate prompts
    system_prompt = format_generate_spec_system(
        ProgramType.SLB.value,
        DEFAULT_ENGINE_CONFIG,
        DEFAULT_PROMPT_CONFIG,
    )
    user_prompt = format_generate_spec_user(
        ProgramType.SLB.value,
        program_description,
        asset_summary,
    )

    print("\n" + "=" * 80)
    print("SYSTEM PROMPT (sent to LLM)")
    print("=" * 80)
    print(system_prompt)

    print("\n" + "=" * 80)
    print("USER PROMPT (sent to LLM)")
    print("=" * 80)
    print(user_prompt)

    # Call LLM
    print("\n" + "=" * 80)
    print("CALLING LLM...")
    print("=" * 80)

    try:
        spec = llm.generate_selector_spec(
            ProgramType.SLB,
            program_description,
            asset_summary,
        )

        print("\n" + "=" * 80)
        print("LLM RESPONSE")
        print("=" * 80)
        print(json.dumps(spec.model_dump(), indent=2))

        print("\n" + "=" * 80)
        print("ANALYSIS")
        print("=" * 80)
        print(f"User said: 'Raise $60M'")
        print(f"LLM extracted: ${spec.target_amount:,.0f}")

        if spec.target_amount == 60_000_000:
            print("✅ CORRECT - Exact extraction!")
        else:
            diff = spec.target_amount - 60_000_000
            print(f"❌ INCORRECT - Off by ${abs(diff):,.0f} ({'under' if diff < 0 else 'over'})")
            print(f"\nPossible reasons:")
            print(f"  1. Model is reasoning about feasibility despite instructions")
            print(f"  2. Model is using asset summary to cap the target")
            print(f"  3. Structured output constraints may be influencing behavior")

        print(f"\n📊 Other extractions:")
        if spec.asset_filters.include_asset_types:
            types = [t.value for t in spec.asset_filters.include_asset_types]
            print(f"  Include types: {types}")
        if spec.hard_constraints.max_net_leverage:
            print(f"  Max leverage: {spec.hard_constraints.max_net_leverage:.1f}x")
        if spec.hard_constraints.min_fixed_charge_coverage:
            print(f"  Min FCC: {spec.hard_constraints.min_fixed_charge_coverage:.1f}x")
        if spec.hard_constraints.max_critical_fraction:
            print(f"  Max critical: {spec.hard_constraints.max_critical_fraction:.1%}")

    except Exception as e:
        print(f"\n❌ LLM call failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
