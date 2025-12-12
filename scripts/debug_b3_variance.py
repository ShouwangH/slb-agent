"""
Debug Portfolio B3 variance.

Investigates why B3 shows different target amounts across runs.
"""

import os
import sys

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
from app.config import EngineConfig


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
    print("DEBUG: PORTFOLIO B3 VARIANCE INVESTIGATION")
    print("=" * 80)

    if not os.environ.get("OPENAI_API_KEY"):
        print("\n❌ ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    llm = OpenAILLMClient()
    print(f"\n✅ Using model: {llm.config.model}")
    print(f"   Temperature: {llm.config.temperature}")
    print(f"   Max tokens: {llm.config.max_tokens}")

    program_description = """
        SLB focusing on HQ offices and distribution centers. Target $80M.
        Very strict constraint: critical asset concentration must stay below 49%.

        Requirements:
        - Only consider offices and distribution centers
        - Leverage below 4.0x
        - Interest coverage above 3.0x
        - Fixed charge coverage above 3.0x
        - Critical asset concentration MUST be below 49%
        """

    print("\n" + "-" * 80)
    print("RUNNING B3 TEN TIMES TO MEASURE VARIANCE")
    print("-" * 80)

    results = []

    for run in range(10):
        # Trace LLM calls for detailed output
        call_log = []

        original_gen = llm.generate_selector_spec
        original_rev = llm.revise_selector_spec

        def trace_gen(program_type, program_description, asset_summary):
            r = original_gen(program_type, program_description, asset_summary)
            call_log.append(('generate', r.target_amount))
            return r

        def trace_rev(original_description, previous_spec, outcome):
            r = original_rev(original_description, previous_spec, outcome)
            call_log.append(('revise', r.target_amount))
            return r

        llm.generate_selector_spec = trace_gen
        llm.revise_selector_spec = trace_rev

        request = ProgramRequest(
            assets=PORTFOLIO_B_ASSETS,
            corporate_state=PORTFOLIO_B_CORPORATE,
            program_type=ProgramType.SLB,
            program_description=program_description,
        )

        try:
            response = run_program(request, llm, PORTFOLIO_B_CONFIG)

            final_target = response.selector_spec.target_amount
            final_status = response.outcome.status.value

            results.append({
                'run': run + 1,
                'target': final_target,
                'status': final_status,
                'calls': call_log.copy(),
            })

            print(f"  Run {run + 1:2d}: ${final_target:>11,.0f} ({final_status:12s}) - {len(call_log)} LLM calls")

        except Exception as e:
            print(f"  Run {run + 1:2d}: ❌ FAILED - {e}")
            results.append({
                'run': run + 1,
                'target': None,
                'status': 'ERROR',
                'calls': [],
            })

        # Restore original methods
        llm.generate_selector_spec = original_gen
        llm.revise_selector_spec = original_rev

    print("\n" + "=" * 80)
    print("VARIANCE ANALYSIS")
    print("=" * 80)

    targets = [r['target'] for r in results if r['target'] is not None]

    if targets:
        unique_targets = sorted(set(targets))
        print(f"\nUnique targets: {len(unique_targets)}")
        for t in unique_targets:
            count = targets.count(t)
            pct = (count / len(targets)) * 100
            print(f"  ${t:,.0f}: {count}/10 runs ({pct:.0f}%)")

        min_target = min(targets)
        max_target = max(targets)
        avg_target = sum(targets) / len(targets)

        print(f"\nRange:")
        print(f"  Min: ${min_target:,.0f}")
        print(f"  Max: ${max_target:,.0f}")
        print(f"  Avg: ${avg_target:,.0f}")
        print(f"  Variance: ${max_target - min_target:,.0f} ({((max_target - min_target) / 80_000_000 * 100):.1f}% of $80M)")

        print(f"\nDeviation from $80M target:")
        for t in unique_targets:
            diff = t - 80_000_000
            pct = (diff / 80_000_000) * 100
            print(f"  ${t:,.0f}: {diff:+,.0f} ({pct:+.1f}%)")

    print("\n" + "=" * 80)
    print("DETAILED CALL TRACES")
    print("=" * 80)

    # Group by final target
    from collections import defaultdict
    by_target = defaultdict(list)
    for r in results:
        if r['target']:
            by_target[r['target']].append(r)

    for target in sorted(by_target.keys()):
        runs = by_target[target]
        print(f"\n${target:,.0f} ({len(runs)} runs):")

        # Show one example trace
        example = runs[0]
        print(f"  Example (Run {example['run']}):")
        for i, (call_type, amount) in enumerate(example['calls']):
            print(f"    {i+1}. {call_type:8s} → ${amount:,.0f}")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    if len(unique_targets) == 1:
        print(f"\n✅ DETERMINISTIC: All runs produced ${unique_targets[0]:,.0f}")
    else:
        print(f"\n⚠️  NON-DETERMINISTIC: {len(unique_targets)} different targets across 10 runs")
        print(f"\nLikely causes:")
        print(f"  - Temperature = {llm.config.temperature} (non-zero introduces sampling variance)")
        print(f"  - Structured output with constraints may have multiple valid solutions")
        print(f"  - LLM reasoning about 'explicit' vs 'soft' requirements varies")


if __name__ == "__main__":
    main()
