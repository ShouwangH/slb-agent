/**
 * Portfolio B assets for demo purposes.
 *
 * This is the full 10-asset portfolio from scripts/demo_portfolio_b.py.
 * Total book value: ~$435M
 */

import type { Asset, CorporateState } from "../types";

// Corporate state matching Portfolio B
export const DEFAULT_CORPORATE_STATE: CorporateState = {
  net_debt: 300_000_000,
  ebitda: 80_000_000,
  interest_expense: 18_000_000,
  lease_expense: 5_000_000,
};

// Portfolio B: 10 assets (~$435M book value)
export const SAMPLE_ASSETS: Asset[] = [
  {
    asset_id: "hq-nyc",
    name: "NYC HQ",
    asset_type: "office",
    market: "New York, NY",
    market_tier: 1,
    noi: 5_000_000,
    book_value: 90_000_000,
    criticality: 0.98,
    leaseability_score: 0.40,
  },
  {
    asset_id: "hq-chicago",
    name: "Chicago HQ",
    asset_type: "office",
    market: "Chicago, IL",
    market_tier: 1,
    noi: 3_000_000,
    book_value: 60_000_000,
    criticality: 0.90,
    leaseability_score: 0.50,
  },
  {
    asset_id: "dc-nj",
    name: "Northeast DC",
    asset_type: "distribution_center",
    market: "Newark, NJ",
    market_tier: 1,
    noi: 4_000_000,
    book_value: 70_000_000,
    criticality: 0.80,
    leaseability_score: 0.75,
  },
  {
    asset_id: "dc-tx",
    name: "Texas DC",
    asset_type: "distribution_center",
    market: "Dallas, TX",
    market_tier: 2,
    noi: 3_000_000,
    book_value: 50_000_000,
    criticality: 0.70,
    leaseability_score: 0.70,
  },
  {
    asset_id: "store-nyc-1",
    name: "NYC Flagship",
    asset_type: "store",
    market: "New York, NY",
    market_tier: 1,
    noi: 2_500_000,
    book_value: 40_000_000,
    criticality: 0.50,
    leaseability_score: 0.90,
  },
  {
    asset_id: "store-nyc-2",
    name: "NYC Secondary",
    asset_type: "store",
    market: "Brooklyn, NY",
    market_tier: 1,
    noi: 1_800_000,
    book_value: 30_000_000,
    criticality: 0.40,
    leaseability_score: 0.85,
  },
  {
    asset_id: "store-la",
    name: "LA Flagship",
    asset_type: "store",
    market: "Los Angeles, CA",
    market_tier: 1,
    noi: 2_200_000,
    book_value: 38_000_000,
    criticality: 0.50,
    leaseability_score: 0.88,
  },
  {
    asset_id: "store-atl",
    name: "Atlanta Store",
    asset_type: "store",
    market: "Atlanta, GA",
    market_tier: 2,
    noi: 1_600_000,
    book_value: 25_000_000,
    criticality: 0.30,
    leaseability_score: 0.80,
  },
  {
    asset_id: "store-ia",
    name: "Iowa Outlet",
    asset_type: "store",
    market: "Des Moines, IA",
    market_tier: 3,
    noi: 900_000,
    book_value: 12_000_000,
    criticality: 0.20,
    leaseability_score: 0.40,
  },
  {
    asset_id: "spec-plant",
    name: "Specialty Plant",
    asset_type: "other",
    market: "Topeka, KS",
    market_tier: 3,
    noi: 1_500_000,
    book_value: 20_000_000,
    criticality: 0.60,
    leaseability_score: 0.30,
  },
];
