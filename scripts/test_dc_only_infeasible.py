"""
Test DC-Only Scenario with Infeasible Target.

Verifies that when user requests "only distribution centers" with an infeasible
target, the system does NOT select stores (even though they exist and could help).
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


# Portfolio B Assets
PORTFOLIO_B_ASSETS = [
    Asset(
        asset_id="hq-nyc",
        name="NYC HQ",
        asset_type=AssetType.OFFICE,
        market="New York, NY",
        market_tier=MarketTier.TIER_1,
        noi=5_000_000.0,
        book_value=90_000_000.0,
        criticality=0.98,
        leaseability_score=0.40,
    ),
    Asset(
        asset_id="hq-chicago",
        name="Chicago HQ",
        asset_type=AssetType.OFFICE,
        market="Chicago, IL",
        market_tier=MarketTier.TIER_1,
        noi=3_000_000.0,
        book_value=60_000_000.0,
        criticality=0.90,
        leaseability_score=0.50,
    ),
    Asset(
        asset_id="dc-nj",
        name="Northeast DC",
        asset_type=AssetType.DISTRIBUTION_CENTER,
        market="Newark, NJ",
        market_tier=MarketTier.TIER_1,
        noi=4_000_000.0,
        book_value=70_000_000.0,
        criticality=0.80,
        leaseability_score=0.75,
    ),
    Asset(
        asset_id="dc-tx",
        name="Texas DC",
        asset_type=AssetType.DISTRIBUTION_CENTER,
        market="Dallas, TX",
        market_tier=MarketTier.TIER_2,
        noi=3_000_000.0,
        book_value=50_000_000.0,
        criticality=0.70,
        leaseability_score=0.70,
    ),
    Asset(
        asset_id="store-nyc-1",
        name="NYC Flagship",
        asset_type=AssetType.STORE,
        market="New York, NY",
        market_tier=MarketTier.TIER_1,
        noi=2_500_000.0,
        book_value=40_000_000.0,
        criticality=0.50,
        leaseability_score=0.90,
    ),
    Asset(
        asset_id="store-nyc-2",
        name="NYC Secondary",
        asset_type=AssetType.STORE,
        market="Brooklyn, NY",
        market_tier=MarketTier.TIER_1,
        noi=1_800_000.0,
        book_value=30_000_000.0,
        criticality=0.40,
        leaseability_score=0.85,
    ),
    Asset(
        asset_id="store-la",
        name="LA Flagship",
        asset_type=AssetType.STORE,
        market="Los Angeles, CA",
        market_tier=MarketTier.TIER_1,
        noi=2_200_000.0,
        book_value=38_000_000.0,
        criticality=0.50,
        leaseability_score=0.88,
    ),
    Asset(
        asset_id="store-atl",
        name="Atlanta Store",
        asset_type=AssetType.STORE,
        market="Atlanta, GA",
        market_tier=MarketTier.TIER_2,
        noi=1_600_000.0,
        book_value=25_000_000.0,
        criticality=0.30,
        leaseability_score=0.80,
    ),
    Asset(
        asset_id="store-ia",
        name="Iowa Outlet",
        asset_type=AssetType.STORE,
        market="Des Moines, IA",
        market_tier=MarketTier.TIER_3,
        noi=900_000.0,
        book_value=12_000_000.0,
        criticality=0.20,
        leaseability_score=0.40,
    ),
    Asset(
        asset_id="spec-plant",
        name="Specialty Plant",
        asset_type=AssetType.OTHER,
        market="Topeka, KS",
        market_tier=MarketTier.TIER_3,
        noi=1_500_000.0,
        book_value=20_000_000.0,
        criticality=0.60,
        leaseability_score=0.30,
    ),
]

PORTFOLIO_B_CORPORATE = CorporateState(
    net_debt=300_000_000.0,
    ebitda=80_000_000.0,
    interest_expense=18_000_000.0,
    lease_expense=5_000_000.0,
)

PORTFOLIO_B_CONFIG = EngineConfig(
    transaction_haircut=0.025,
    slb_rent_multiplier=1.0,
    avg_cost_of_debt=0.06,
    epsilon=1e-6,
    criticality_threshold=0.7,
    target_tolerance=0.05,
    default_market_tier=MarketTier.TIER_2,
    cap_rate_curve={
        AssetType.STORE: {
            MarketTier.TIER_1: 0.06,
            MarketTier.TIER_2: 0.065,
            MarketTier.TIER_3: 0.07,
        },
        AssetType.DISTRIBUTION_CENTER: {
            MarketTier.TIER_1: 0.055,
            MarketTier.TIER_2: 0.06,
            MarketTier.TIER_3: 0.065,
        },
        AssetType.OFFICE: {
            MarketTier.TIER_1: 0.055,
            MarketTier.TIER_2: 0.06,
            MarketTier.TIER_3: 0.07,
        },
        AssetType.MIXED_USE: {
            MarketTier.TIER_1: 0.055,
            MarketTier.TIER_2: 0.06,
            MarketTier.TIER_3: 0.065,
        },
        AssetType.OTHER: {
            MarketTier.TIER_1: 0.065,
            MarketTier.TIER_2: 0.07,
            MarketTier.TIER_3: 0.075,
        },
    },
)


def main():
    print("=" * 80)
    print("DC-ONLY INFEASIBLE TARGET TEST")
    print("=" * 80)

    if not os.environ.get("OPENAI_API_KEY"):
        print("\n❌ ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    llm = OpenAILLMClient()
    print(f"\n✅ Model: {llm.config.model}, Temperature: {llm.config.temperature}")

    # Calculate max feasible with DCs only
    dc_assets = [a for a in PORTFOLIO_B_ASSETS if a.asset_type == AssetType.DISTRIBUTION_CENTER]
    print(f"\nPortfolio composition:")
    print(f"  Distribution Centers: {len(dc_assets)}")
    print(f"    - dc-nj: NOI ${dc_assets[0].noi:,.0f}")
    print(f"    - dc-tx: NOI ${dc_assets[1].noi:,.0f}")

    total_dc_noi = sum(a.noi for a in dc_assets)
    estimated_dc_value = total_dc_noi / 0.06  # Rough estimate
    print(f"  Total DC NOI: ${total_dc_noi:,.0f}")
    print(f"  Estimated max DC value: ~${estimated_dc_value:,.0f}")

    store_count = len([a for a in PORTFOLIO_B_ASSETS if a.asset_type == AssetType.STORE])
    print(f"  Stores available (should NOT be selected): {store_count}")

    print("\n" + "-" * 80)
    print("TEST: Request $150M with 'only distribution centers'")
    print("      (Max achievable with DCs is ~$117M)")
    print("-" * 80)

    request = ProgramRequest(
        assets=PORTFOLIO_B_ASSETS,
        corporate_state=PORTFOLIO_B_CORPORATE,
        program_type=ProgramType.SLB,
        program_description="""
        Raise $150M via sale-leaseback. Only consider distribution centers.
        We need to focus exclusively on our logistics assets for this transaction.

        Requirements:
        - Target: $150M
        - Only distribution centers (DCs) - do NOT include stores or offices
        - Leverage below 4.0x
        - Interest coverage above 3.0x
        - Fixed charge coverage above 3.0x
        """,
    )

    try:
        response = run_program(request, llm, PORTFOLIO_B_CONFIG)

        spec = response.selector_spec
        outcome = response.outcome

        print(f"\n📊 LLM-Generated Spec:")
        print(f"  Target: ${spec.target_amount:,.0f}")
        if spec.asset_filters.include_asset_types:
            types = [t.value for t in spec.asset_filters.include_asset_types]
            print(f"  Include Types: {types}")
        if spec.asset_filters.exclude_asset_types:
            types = [t.value for t in spec.asset_filters.exclude_asset_types]
            print(f"  Exclude Types: {types}")

        print(f"\n📈 Outcome:")
        print(f"  Status: {outcome.status.value.upper()}")
        print(f"  Proceeds: ${outcome.proceeds:,.0f}")
        print(f"  Assets Selected: {len(outcome.selected_assets)}")

        if outcome.selected_assets:
            print(f"\n  Selected Assets:")
            stores_selected = []
            dcs_selected = []
            others_selected = []

            for selection in outcome.selected_assets:
                asset_type = selection.asset.asset_type.value
                print(f"    - {selection.asset.asset_id:15} ({asset_type:20}) ${selection.proceeds:>12,.0f}")

                if selection.asset.asset_type == AssetType.STORE:
                    stores_selected.append(selection.asset.asset_id)
                elif selection.asset.asset_type == AssetType.DISTRIBUTION_CENTER:
                    dcs_selected.append(selection.asset.asset_id)
                else:
                    others_selected.append(selection.asset.asset_id)

            print(f"\n  Asset Type Breakdown:")
            print(f"    DCs selected: {len(dcs_selected)}")
            print(f"    Stores selected: {len(stores_selected)}")
            print(f"    Others selected: {len(others_selected)}")

            # CRITICAL TEST: No stores should be selected
            if stores_selected:
                print(f"\n  ❌ FAILURE: Stores were selected despite 'only DCs' requirement!")
                print(f"     Stores: {stores_selected}")
            else:
                print(f"\n  ✅ SUCCESS: No stores selected (explicit filter preserved)")

            # Check if only DCs were selected
            if len(dcs_selected) == len(outcome.selected_assets):
                print(f"  ✅ SUCCESS: Only DCs selected (100% compliance)")
            else:
                print(f"  ⚠️  WARNING: Non-DC assets selected")

        else:
            print(f"\n  No assets selected")

        if outcome.violations:
            print(f"\n  Violations: {[v.code for v in outcome.violations]}")

        print(f"\n💬 Summary:")
        print(f"  {response.explanation.summary}")

        # Final verdict
        print("\n" + "=" * 80)
        print("TEST VERDICT")
        print("=" * 80)

        stores_in_selection = any(
            s.asset.asset_type == AssetType.STORE
            for s in outcome.selected_assets
        )

        if stores_in_selection:
            print("\n❌ FAILED: System selected stores despite explicit 'only DCs' requirement")
            print("   This means the LLM is not respecting explicit asset type filters.")
        else:
            print("\n✅ PASSED: System respected 'only DCs' filter")
            print(f"   Target: ${spec.target_amount:,.0f}")
            print(f"   Status: {outcome.status.value}")
            print(f"   Proceeds: ${outcome.proceeds:,.0f} (from DCs only)")

            if spec.target_amount == 150_000_000:
                print("\n✅ BONUS: LLM preserved $150M target despite infeasibility")
            else:
                print(f"\n⚠️  NOTE: LLM reduced target to ${spec.target_amount:,.0f}")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
