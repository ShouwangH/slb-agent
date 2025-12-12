"""
OpenAI LLM Demo Script

Tests the real OpenAI LLM client with Portfolio B.
Requires OPENAI_API_KEY environment variable to be set.

Usage:
    # Set your API key first
    export OPENAI_API_KEY=sk-...

    # Or create .env file:
    cp .env.example .env
    # Edit .env and add OPENAI_API_KEY=sk-...

    # Run the demo
    python -m scripts.demo_with_openai
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
# Portfolio B Assets (same as demo_portfolio_b.py)
# =============================================================================

PORTFOLIO_B_ASSETS: list[Asset] = [
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
# Demo Function
# =============================================================================


def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def main():
    """Run OpenAI LLM demo."""
    print_separator("=")
    print("OPENAI LLM DEMO - SLB AGENT")
    print_separator("=")

    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n❌ ERROR: OPENAI_API_KEY not found in environment variables.")
        print("\nPlease set your API key:")
        print("  export OPENAI_API_KEY=sk-...")
        print("\nOr create a .env file:")
        print("  cp .env.example .env")
        print("  # Edit .env and add OPENAI_API_KEY=sk-...")
        print()
        sys.exit(1)

    print(f"\n✅ Found OPENAI_API_KEY: {api_key[:20]}...")

    # Portfolio summary
    print("\nPortfolio Summary:")
    print(f"  Total Assets: {len(PORTFOLIO_B_ASSETS)}")
    print(f"  Total NOI: ${sum(a.noi for a in PORTFOLIO_B_ASSETS):,.0f}")
    print(f"  Total Book Value: ${sum(a.book_value for a in PORTFOLIO_B_ASSETS):,.0f}")

    print("\nCorporate State:")
    print(f"  Net Debt: ${PORTFOLIO_B_CORPORATE.net_debt:,.0f}")
    print(f"  EBITDA: ${PORTFOLIO_B_CORPORATE.ebitda:,.0f}")
    print(f"  Interest Expense: ${PORTFOLIO_B_CORPORATE.interest_expense:,.0f}")
    print(f"  Lease Expense: ${PORTFOLIO_B_CORPORATE.lease_expense:,.0f}")

    baseline_leverage = PORTFOLIO_B_CORPORATE.net_debt / PORTFOLIO_B_CORPORATE.ebitda
    baseline_ic = PORTFOLIO_B_CORPORATE.ebitda / PORTFOLIO_B_CORPORATE.interest_expense
    baseline_fcc = PORTFOLIO_B_CORPORATE.ebitda / (
        PORTFOLIO_B_CORPORATE.interest_expense + PORTFOLIO_B_CORPORATE.lease_expense
    )

    print("\nBaseline Metrics:")
    print(f"  Leverage: {baseline_leverage:.2f}x")
    print(f"  Interest Coverage: {baseline_ic:.2f}x")
    print(f"  Fixed Charge Coverage: {baseline_fcc:.2f}x")

    # Create LLM client
    print("\n" + "=" * 80)
    print("Initializing OpenAI LLM Client...")
    print("=" * 80)

    try:
        llm = OpenAILLMClient()
        print("✅ OpenAI client initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize OpenAI client: {e}")
        sys.exit(1)

    # Create request with natural language description
    print("\n" + "=" * 80)
    print("Running SLB Program with OpenAI LLM")
    print("=" * 80)

    program_description = """
    We need to raise approximately $120M through a sale-leaseback program to reduce
    our leverage. We want to focus on non-critical retail stores with good
    re-leasing potential. Our target is to bring leverage down from 3.75x to around
    2.0x while maintaining strong fixed charge coverage above 3.0x.

    Key requirements:
    - Target proceeds: ~$120M
    - Avoid selling HQ offices (too critical to operations)
    - Prefer stores with high leaseability scores
    - Keep critical asset exposure low
    - Maintain leverage below 4.0x and fixed charge coverage above 3.0x
    """.strip()

    print(f"\nProgram Description:")
    print(f"  {program_description}")
    print()

    request = ProgramRequest(
        assets=PORTFOLIO_B_ASSETS,
        corporate_state=PORTFOLIO_B_CORPORATE,
        program_type=ProgramType.SLB,
        program_description=program_description,
    )

    # Run orchestrator with OpenAI LLM
    print("Calling orchestrator with OpenAI LLM...")
    print("(This may take 10-30 seconds depending on API response time)")
    print()

    try:
        response = run_program(request, llm, PORTFOLIO_B_CONFIG)
        print("✅ Program completed successfully!")
    except Exception as e:
        print(f"❌ Program failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    outcome = response.outcome
    spec = response.selector_spec

    print(f"\n📊 LLM-Generated Spec:")
    print(f"  Program Type: {spec.program_type.value}")
    print(f"  Objective: {spec.objective.value}")
    print(f"  Target Amount: ${spec.target_amount:,.0f}")
    print(f"  Max Iterations: {spec.max_iterations}")

    print(f"\n  Hard Constraints:")
    if spec.hard_constraints.max_net_leverage:
        print(f"    - Max Net Leverage: {spec.hard_constraints.max_net_leverage:.2f}x")
    if spec.hard_constraints.min_interest_coverage:
        print(
            f"    - Min Interest Coverage: {spec.hard_constraints.min_interest_coverage:.2f}x"
        )
    if spec.hard_constraints.min_fixed_charge_coverage:
        print(
            f"    - Min Fixed Charge Coverage: {spec.hard_constraints.min_fixed_charge_coverage:.2f}x"
        )
    if spec.hard_constraints.max_critical_fraction:
        print(
            f"    - Max Critical Fraction: {spec.hard_constraints.max_critical_fraction:.1%}"
        )

    print(f"\n  Asset Filters:")
    if spec.asset_filters.include_asset_types:
        types = [t.value for t in spec.asset_filters.include_asset_types]
        print(f"    - Include Types: {types}")
    if spec.asset_filters.exclude_asset_types:
        types = [t.value for t in spec.asset_filters.exclude_asset_types]
        print(f"    - Exclude Types: {types}")
    if spec.asset_filters.min_leaseability_score:
        print(
            f"    - Min Leaseability: {spec.asset_filters.min_leaseability_score:.2f}"
        )
    if spec.asset_filters.max_criticality:
        print(f"    - Max Criticality: {spec.asset_filters.max_criticality:.2f}")

    print(f"\n📈 Outcome:")
    print(f"  Status: {outcome.status.value.upper()}")
    print(f"  Proceeds: ${outcome.proceeds:,.0f}")
    print(f"  Assets Selected: {len(outcome.selected_assets)}")

    print(f"\n  Selected Assets:")
    for selection in outcome.selected_assets:
        print(
            f"    - {selection.asset.asset_id:15} | "
            f"{selection.asset.name:20} | "
            f"Proceeds: ${selection.proceeds:>12,.0f} | "
            f"Rent: ${selection.slb_rent:>10,.0f}"
        )

    print(f"\n  Metrics:")
    print(
        f"    Leverage:           {outcome.leverage_before:.2f}x → {outcome.leverage_after:.2f}x"
    )
    if outcome.interest_coverage_before and outcome.interest_coverage_after:
        print(
            f"    Interest Coverage:  {outcome.interest_coverage_before:.2f}x → {outcome.interest_coverage_after:.2f}x"
        )
    if outcome.fixed_charge_coverage_before and outcome.fixed_charge_coverage_after:
        print(
            f"    Fixed Charge Cov:   {outcome.fixed_charge_coverage_before:.2f}x → {outcome.fixed_charge_coverage_after:.2f}x"
        )
    print(f"    Critical Fraction:  {outcome.critical_fraction:.1%}")

    if outcome.violations:
        print(f"\n  ⚠️  Violations ({len(outcome.violations)}):")
        for v in outcome.violations:
            print(f"    - {v.code}: {v.detail}")

    if outcome.warnings:
        print(f"\n  ⚠️  Warnings ({len(outcome.warnings)}):")
        for w in outcome.warnings:
            print(f"    - {w}")

    print(f"\n💬 LLM-Generated Explanation:")
    print(f"  {response.explanation.summary}")

    print(f"\n  Explanation Details ({len(response.explanation.nodes)} nodes):")
    for node in response.explanation.nodes:
        severity_icon = {"info": "ℹ️", "warning": "⚠️", "error": "❌"}.get(
            node.severity, "•"
        )
        print(f"    {severity_icon} [{node.category:10}] {node.label}")
        if node.detail:
            print(f"       {node.detail}")

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
