"""
Portfolio B Demo Script

Runs three scenarios through the orchestrator to validate engine behavior:
- B1: Conservative deleveraging (stores only, OK)
- B2: Higher target with DCs (OK)
- B3: Tight critical concentration (INFEASIBLE)

Usage:
    python scripts/demo_portfolio_b.py
"""

from app.config import EngineConfig
from app.llm.mock import MockLLMClient
from app.models import (
    Asset,
    AssetFilters,
    AssetType,
    CorporateState,
    HardConstraints,
    MarketTier,
    Objective,
    ProgramRequest,
    ProgramType,
    SelectorSpec,
    SoftPreferences,
)
from app.orchestrator import run_program


# =============================================================================
# Portfolio B Assets
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


# =============================================================================
# Corporate State
# =============================================================================

PORTFOLIO_B_CORPORATE = CorporateState(
    net_debt=300_000_000.0,
    ebitda=80_000_000.0,
    interest_expense=18_000_000.0,
    lease_expense=5_000_000.0,
)


# =============================================================================
# Engine Config
# =============================================================================

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
# Scenario Specs
# =============================================================================

# B1: Conservative deleveraging (stores only, OK)
B1_SPEC = SelectorSpec(
    program_type=ProgramType.SLB,
    objective=Objective.MAXIMIZE_PROCEEDS,
    target_amount=120_000_000.0,
    hard_constraints=HardConstraints(
        max_net_leverage=4.0,
        min_interest_coverage=3.5,
        min_fixed_charge_coverage=3.0,
        max_critical_fraction=0.6,
    ),
    soft_preferences=SoftPreferences(
        prefer_low_criticality=True,
        prefer_high_leaseability=True,
        weight_criticality=1.0,
        weight_leaseability=1.0,
    ),
    asset_filters=AssetFilters(
        include_asset_types=None,
        exclude_asset_types=None,
        exclude_markets=None,
        min_leaseability_score=0.6,
        max_criticality=0.6,
    ),
    max_iterations=3,
)

# B2: Higher target with DCs (OK)
B2_SPEC = SelectorSpec(
    program_type=ProgramType.SLB,
    objective=Objective.MAXIMIZE_PROCEEDS,
    target_amount=180_000_000.0,
    hard_constraints=HardConstraints(
        max_net_leverage=4.0,
        min_interest_coverage=3.0,
        min_fixed_charge_coverage=3.0,
        max_critical_fraction=0.8,
    ),
    soft_preferences=SoftPreferences(
        prefer_low_criticality=True,
        prefer_high_leaseability=True,
        weight_criticality=1.0,
        weight_leaseability=1.0,
    ),
    asset_filters=AssetFilters(
        include_asset_types=[AssetType.STORE, AssetType.DISTRIBUTION_CENTER],
        exclude_asset_types=None,
        exclude_markets=None,
        min_leaseability_score=None,
        max_criticality=None,
    ),
    max_iterations=3,
)

# B3: Tight critical concentration (INFEASIBLE)
B3_SPEC = SelectorSpec(
    program_type=ProgramType.SLB,
    objective=Objective.MAXIMIZE_PROCEEDS,
    target_amount=80_000_000.0,
    hard_constraints=HardConstraints(
        max_net_leverage=4.0,
        min_interest_coverage=3.0,
        min_fixed_charge_coverage=3.0,
        max_critical_fraction=0.49,
    ),
    soft_preferences=SoftPreferences(
        prefer_low_criticality=True,
        prefer_high_leaseability=True,
        weight_criticality=1.0,
        weight_leaseability=1.0,
    ),
    asset_filters=AssetFilters(
        include_asset_types=[AssetType.OFFICE, AssetType.DISTRIBUTION_CENTER],
        exclude_asset_types=None,
        exclude_markets=None,
        min_leaseability_score=None,
        max_criticality=None,
    ),
    max_iterations=3,
)


# =============================================================================
# Helper Functions
# =============================================================================


def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def print_outcome_summary(scenario_name: str, response):
    """Print a summary of the program response."""
    print_separator()
    print(f"SCENARIO: {scenario_name}")
    print_separator()

    outcome = response.outcome
    spec = response.selector_spec

    print(f"\nStatus: {outcome.status.value.upper()}")
    print(f"Target: ${spec.target_amount:,.0f}")
    print(f"Proceeds: ${outcome.proceeds:,.0f}")
    print(f"\nSelected Assets ({len(outcome.selected_assets)}):")
    for selection in outcome.selected_assets:
        print(
            f"  - {selection.asset.asset_id:15} | "
            f"{selection.asset.name:20} | "
            f"Proceeds: ${selection.proceeds:>12,.0f} | "
            f"Rent: ${selection.slb_rent:>10,.0f}"
        )

    print(f"\nMetrics:")
    print(
        f"  Leverage:           {outcome.leverage_before:.2f}x → {outcome.leverage_after:.2f}x"
    )
    if outcome.interest_coverage_before and outcome.interest_coverage_after:
        print(
            f"  Interest Coverage:  {outcome.interest_coverage_before:.2f}x → {outcome.interest_coverage_after:.2f}x"
        )
    if outcome.fixed_charge_coverage_before and outcome.fixed_charge_coverage_after:
        print(
            f"  Fixed Charge Cov:   {outcome.fixed_charge_coverage_before:.2f}x → {outcome.fixed_charge_coverage_after:.2f}x"
        )
    print(f"  Critical Fraction:  {outcome.critical_fraction:.1%}")

    if outcome.violations:
        print(f"\nViolations ({len(outcome.violations)}):")
        for v in outcome.violations:
            print(f"  - {v.code}: {v.detail}")

    if outcome.warnings:
        print(f"\nWarnings ({len(outcome.warnings)}):")
        for w in outcome.warnings:
            print(f"  - {w}")

    print(f"\nExplanation Summary:")
    print(f"  {response.explanation.summary}")

    print(f"\nExplanation Nodes ({len(response.explanation.nodes)}):")
    for node in response.explanation.nodes:
        print(f"  - [{node.severity.upper():7}] {node.category:10} | {node.label}")

    print()


# =============================================================================
# Main Demo
# =============================================================================


def main():
    """Run all three Portfolio B scenarios."""
    print_separator("=")
    print("PORTFOLIO B DEMO - SLB AGENT")
    print_separator("=")

    # Print baseline metrics
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

    print()

    # Scenario B1: Conservative deleveraging
    print_separator("-")
    print("Running Scenario B1: Conservative Deleveraging (stores only)")
    print_separator("-")

    llm_b1 = MockLLMClient(custom_spec=B1_SPEC)
    request_b1 = ProgramRequest(
        assets=PORTFOLIO_B_ASSETS,
        corporate_state=PORTFOLIO_B_CORPORATE,
        program_type=ProgramType.SLB,
        program_description="Conservative deleveraging via SLB of high-quality stores. Target $120M.",
    )
    response_b1 = run_program(request_b1, llm_b1, PORTFOLIO_B_CONFIG)
    print_outcome_summary("B1: Conservative Deleveraging", response_b1)

    # Scenario B2: Higher target with DCs
    print_separator("-")
    print("Running Scenario B2: Higher Target with DCs")
    print_separator("-")

    llm_b2 = MockLLMClient(custom_spec=B2_SPEC)
    request_b2 = ProgramRequest(
        assets=PORTFOLIO_B_ASSETS,
        corporate_state=PORTFOLIO_B_CORPORATE,
        program_type=ProgramType.SLB,
        program_description="Aggressive deleveraging via SLB of stores and DCs. Target $180M.",
    )
    response_b2 = run_program(request_b2, llm_b2, PORTFOLIO_B_CONFIG)
    print_outcome_summary("B2: Higher Target with DCs", response_b2)

    # Scenario B3: Tight critical concentration
    print_separator("-")
    print("Running Scenario B3: Tight Critical Concentration")
    print_separator("-")

    llm_b3 = MockLLMClient(custom_spec=B3_SPEC)
    request_b3 = ProgramRequest(
        assets=PORTFOLIO_B_ASSETS,
        corporate_state=PORTFOLIO_B_CORPORATE,
        program_type=ProgramType.SLB,
        program_description="SLB of HQ/DC assets with strict critical concentration limit. Target $80M.",
    )
    response_b3 = run_program(request_b3, llm_b3, PORTFOLIO_B_CONFIG)
    print_outcome_summary("B3: Tight Critical Concentration", response_b3)

    print_separator("=")
    print("DEMO COMPLETE")
    print_separator("=")


if __name__ == "__main__":
    main()
