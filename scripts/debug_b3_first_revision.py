"""
Debug B3 First Revision Variance.

Tests to pinpoint exactly where non-determinism occurs in the B3 scenario.
"""

import os
import sys

from app.config import EngineConfig
from app.engine.selector import select_assets
from app.llm.openai_client import OpenAILLMClient
from app.models import (
    Asset,
    AssetFilters,
    AssetType,
    CorporateState,
    HardConstraints,
    MarketTier,
    ProgramType,
    SelectorSpec,
    SoftPreferences,
)
from scripts.debug_b3_variance import PORTFOLIO_B_ASSETS, PORTFOLIO_B_CORPORATE, PORTFOLIO_B_CONFIG


def main():
    print("=" * 80)
    print("B3 FIRST REVISION VARIANCE TEST")
    print("=" * 80)

    if not os.environ.get("OPENAI_API_KEY"):
        print("\n❌ ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    llm = OpenAILLMClient()
    print(f"\n✅ Model: {llm.config.model}, Temperature: {llm.config.temperature}")

    # Step 1: Test initial spec generation (10 times)
    print("\n" + "-" * 80)
    print("STEP 1: Initial Spec Generation (10 runs)")
    print("-" * 80)

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

    asset_summary = "10 assets across 7 markets. Types: 2 office, 2 distribution_center, 5 store, 1 other. Total book value: $435,000,000. Total NOI: $24,500,000. Avg criticality: 0.58. Avg leaseability: 0.69."

    initial_specs = []
    for i in range(10):
        spec = llm.generate_selector_spec(
            ProgramType.SLB,
            program_description,
            asset_summary,
        )
        initial_specs.append(spec.target_amount)
        print(f"  Run {i+1}: ${spec.target_amount:,.0f}")

    unique_initial = set(initial_specs)
    print(f"\nResult: {len(unique_initial)} unique value(s)")
    if len(unique_initial) == 1:
        print(f"✅ DETERMINISTIC: All runs produced ${list(unique_initial)[0]:,.0f}")
    else:
        print(f"❌ NON-DETERMINISTIC")

    # Step 2: Run engine with the initial spec
    print("\n" + "-" * 80)
    print("STEP 2: Engine Run with Initial Spec (10 runs)")
    print("-" * 80)

    initial_spec = SelectorSpec(
        program_type=ProgramType.SLB,
        objective="maximize_proceeds",
        target_amount=80_000_000,
        hard_constraints=HardConstraints(
            max_net_leverage=4.0,
            min_interest_coverage=3.0,
            min_fixed_charge_coverage=3.0,
            max_critical_fraction=0.49,
        ),
        asset_filters=AssetFilters(
            include_asset_types=[AssetType.OFFICE, AssetType.DISTRIBUTION_CENTER],
        ),
        soft_preferences=SoftPreferences(),
        max_iterations=3,
    )

    engine_outcomes = []
    for i in range(10):
        outcome = select_assets(
            PORTFOLIO_B_ASSETS,
            PORTFOLIO_B_CORPORATE,
            initial_spec,
            PORTFOLIO_B_CONFIG,
        )
        engine_outcomes.append((outcome.status.value, outcome.proceeds, len(outcome.violations)))
        print(f"  Run {i+1}: {outcome.status.value:12s}, proceeds: ${outcome.proceeds:,.0f}, violations: {len(outcome.violations)}")

    unique_outcomes = set(engine_outcomes)
    print(f"\nResult: {len(unique_outcomes)} unique outcome(s)")
    if len(unique_outcomes) == 1:
        print(f"✅ DETERMINISTIC: Engine always produces same outcome")
    else:
        print(f"❌ NON-DETERMINISTIC")

    # Use the deterministic outcome for next step
    first_outcome = select_assets(
        PORTFOLIO_B_ASSETS,
        PORTFOLIO_B_CORPORATE,
        initial_spec,
        PORTFOLIO_B_CONFIG,
    )

    # Step 3: Test FIRST revision with the outcome from engine
    print("\n" + "-" * 80)
    print("STEP 3: First Revision after Initial Engine Run (10 runs)")
    print("-" * 80)
    print(f"Input: Previous spec had ${initial_spec.target_amount:,.0f}")
    print(f"       Engine returned: {first_outcome.status.value}, ${first_outcome.proceeds:,.0f}")
    print(f"       Violations: {[v.code for v in first_outcome.violations]}")
    print()

    first_revisions = []
    for i in range(10):
        revised_spec = llm.revise_selector_spec(
            program_description,
            initial_spec,
            first_outcome,
        )
        first_revisions.append(revised_spec.target_amount)
        print(f"  Run {i+1}: ${revised_spec.target_amount:,.0f}")

    unique_revisions = set(first_revisions)
    print(f"\nResult: {len(unique_revisions)} unique value(s)")
    if len(unique_revisions) == 1:
        print(f"✅ DETERMINISTIC: All revisions produced ${list(unique_revisions)[0]:,.0f}")
    else:
        print(f"❌ NON-DETERMINISTIC at FIRST REVISION step")
        for val in sorted(unique_revisions):
            count = first_revisions.count(val)
            pct = (count / len(first_revisions)) * 100
            print(f"   ${val:,.0f}: {count}/10 runs ({pct:.0f}%)")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nStep 1 - Initial Generation: {'✅ Deterministic' if len(unique_initial) == 1 else '❌ Non-deterministic'}")
    print(f"Step 2 - Engine Execution:   {'✅ Deterministic' if len(unique_outcomes) == 1 else '❌ Non-deterministic'}")
    print(f"Step 3 - First Revision:     {'✅ Deterministic' if len(unique_revisions) == 1 else '❌ Non-deterministic'}")

    if len(unique_revisions) > 1:
        print(f"\n⚠️  VARIANCE DETECTED AT FIRST REVISION STEP")
        print(f"\nThis means:")
        print(f"  - Initial spec generation is deterministic ($80M)")
        print(f"  - Engine execution is deterministic (infeasible, ~$49M proceeds)")
        print(f"  - BUT the LLM's revision decision varies")
        print(f"\nPossible causes:")
        print(f"  1. OpenAI API has some non-determinism even at temperature=0")
        print(f"  2. Structured output parsing may have multiple valid solutions")
        print(f"  3. The revision prompt allows subjective interpretation of 'explicit'")
        print(f"\nRange: ${min(first_revisions):,.0f} to ${max(first_revisions):,.0f}")
        print(f"All values within policy bounds: {min(first_revisions) >= 80_000_000 * 0.75}")


if __name__ == "__main__":
    main()
