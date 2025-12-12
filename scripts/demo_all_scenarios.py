"""
Comprehensive Demo: All Scenarios with OpenAI LLM

Runs both toy scenarios and Portfolio B scenarios to validate LLM behavior.

Scenarios:
- Toy 1: Two stores selected (OK)
- Toy 2: No eligible assets (INFEASIBLE)
- Toy 3: Target too high, revision needed (INFEASIBLE → OK)
- Portfolio B1: Conservative deleveraging (OK)
- Portfolio B2: Higher target with DCs (OK)
- Portfolio B3: Tight critical concentration (INFEASIBLE)

Usage:
    export OPENAI_API_KEY=sk-...
    python -m scripts.demo_all_scenarios
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


# =============================================================================
# Toy Scenarios Assets & Config
# =============================================================================

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


# =============================================================================
# Portfolio B Assets & Config
# =============================================================================

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


# =============================================================================
# Helpers
# =============================================================================


def print_separator(char="=", length=80):
    print(char * length)


def print_outcome(scenario_name: str, response, expected_status: str):
    """Print outcome summary with comparison to expected."""
    outcome = response.outcome
    spec = response.selector_spec

    status_match = "✅" if outcome.status.value == expected_status.lower() else "❌"

    print(f"\n{scenario_name}")
    print(f"  Status: {outcome.status.value.upper()} {status_match} (expected: {expected_status})")
    print(f"  Target: ${spec.target_amount:,.0f}")
    print(f"  Proceeds: ${outcome.proceeds:,.0f}")
    print(f"  Assets: {len(outcome.selected_assets)} selected")
    if outcome.selected_assets:
        asset_ids = [s.asset.asset_id for s in outcome.selected_assets]
        print(f"    {', '.join(asset_ids)}")
    print(
        f"  Leverage: {outcome.leverage_before:.2f}x → {outcome.leverage_after:.2f}x"
    )
    if outcome.violations:
        print(f"  Violations: {[v.code for v in outcome.violations]}")


# =============================================================================
# Main
# =============================================================================


def main():
    print_separator("=")
    print("COMPREHENSIVE SCENARIO DEMO - OpenAI LLM")
    print_separator("=")

    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n❌ ERROR: OPENAI_API_KEY not set")
        print("  export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    print("\n✅ OpenAI API key found")

    llm = OpenAILLMClient()

    # =========================================================================
    # TOY SCENARIOS
    # =========================================================================

    print("\n" + "=" * 80)
    print("TOY SCENARIOS")
    print("=" * 80)

    # Toy 1: OK
    print("\n[1/6] Toy Scenario 1: Two stores selected (expected: OK)")
    request_toy1 = ProgramRequest(
        assets=TOY_ASSETS,
        corporate_state=TOY_CORPORATE,
        program_type=ProgramType.SLB,
        program_description="""
        Raise $50M via sale-leaseback. Focus on retail stores only, avoid HQ offices.
        Target leverage below 4.0x, fixed charge coverage above 2.5x.
        Prefer low-criticality assets with high leaseability scores.
        Keep critical asset concentration below 70%.
        """,
    )
    try:
        response_toy1 = run_program(request_toy1, llm, TOY_CONFIG)
        print_outcome("Toy 1", response_toy1, "OK")
    except Exception as e:
        print(f"❌ Failed: {e}")

    # Toy 2: No eligible assets
    print("\n[2/6] Toy Scenario 2: No eligible assets (expected: INFEASIBLE)")
    request_toy2 = ProgramRequest(
        assets=TOY_ASSETS,
        corporate_state=TOY_CORPORATE,
        program_type=ProgramType.SLB,
        program_description="""
        Raise $20M via sale-leaseback. Only consider mixed-use properties.
        """,
    )
    try:
        response_toy2 = run_program(request_toy2, llm, TOY_CONFIG)
        print_outcome("Toy 2", response_toy2, "INFEASIBLE")
    except Exception as e:
        print(f"❌ Failed: {e}")

    # Toy 3: Target too high, revision needed
    print("\n[3/6] Toy Scenario 3: Target too high, revision (expected: INFEASIBLE or OK after revision)")
    request_toy3 = ProgramRequest(
        assets=TOY_ASSETS,
        corporate_state=TOY_CORPORATE,
        program_type=ProgramType.SLB,
        program_description="""
        Raise $60M via sale-leaseback. Focus on retail stores only.
        Target leverage below 4.0x, fixed charge coverage above 2.5x.
        Keep critical asset concentration below 70%.
        """,
    )
    try:
        response_toy3 = run_program(request_toy3, llm, TOY_CONFIG)
        print_outcome("Toy 3", response_toy3, "OK or INFEASIBLE")
    except Exception as e:
        print(f"❌ Failed: {e}")

    # =========================================================================
    # PORTFOLIO B SCENARIOS
    # =========================================================================

    print("\n" + "=" * 80)
    print("PORTFOLIO B SCENARIOS")
    print("=" * 80)

    # B1: Conservative deleveraging
    print("\n[4/6] Portfolio B1: Conservative deleveraging (expected: OK)")
    request_b1 = ProgramRequest(
        assets=PORTFOLIO_B_ASSETS,
        corporate_state=PORTFOLIO_B_CORPORATE,
        program_type=ProgramType.SLB,
        program_description="""
        Conservative deleveraging via SLB. Target $120M in proceeds.
        Focus on non-critical retail stores with good re-leasing potential.
        Avoid selling HQ offices (too critical to operations).

        Requirements:
        - Leverage below 4.0x
        - Interest coverage above 3.5x
        - Fixed charge coverage above 3.0x
        - Keep critical asset concentration below 60%
        - Only select stores with leaseability above 0.6
        - Avoid critical assets (criticality above 0.6)
        """,
    )
    try:
        response_b1 = run_program(request_b1, llm, PORTFOLIO_B_CONFIG)
        print_outcome("Portfolio B1", response_b1, "OK")
    except Exception as e:
        print(f"❌ Failed: {e}")

    # B2: Higher target with DCs
    print("\n[5/6] Portfolio B2: Higher target with DCs (expected: OK)")
    request_b2 = ProgramRequest(
        assets=PORTFOLIO_B_ASSETS,
        corporate_state=PORTFOLIO_B_CORPORATE,
        program_type=ProgramType.SLB,
        program_description="""
        Aggressive deleveraging via SLB. Target $180M in proceeds.
        Include both retail stores and distribution centers.
        Avoid selling HQ offices.

        Requirements:
        - Leverage below 4.0x
        - Interest coverage above 3.0x
        - Fixed charge coverage above 3.0x
        - Keep critical asset concentration below 80%
        """,
    )
    try:
        response_b2 = run_program(request_b2, llm, PORTFOLIO_B_CONFIG)
        print_outcome("Portfolio B2", response_b2, "OK")
    except Exception as e:
        print(f"❌ Failed: {e}")

    # B3: Tight critical concentration
    print("\n[6/6] Portfolio B3: Tight critical concentration (expected: INFEASIBLE)")
    request_b3 = ProgramRequest(
        assets=PORTFOLIO_B_ASSETS,
        corporate_state=PORTFOLIO_B_CORPORATE,
        program_type=ProgramType.SLB,
        program_description="""
        SLB focusing on HQ offices and distribution centers. Target $80M.
        Very strict constraint: critical asset concentration must stay below 49%.

        Requirements:
        - Only consider offices and distribution centers
        - Leverage below 4.0x
        - Interest coverage above 3.0x
        - Fixed charge coverage above 3.0x
        - Critical asset concentration MUST be below 49%
        """,
    )
    try:
        response_b3 = run_program(request_b3, llm, PORTFOLIO_B_CONFIG)
        print_outcome("Portfolio B3", response_b3, "INFEASIBLE")
    except Exception as e:
        print(f"❌ Failed: {e}")

    print("\n" + "=" * 80)
    print("ALL SCENARIOS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
