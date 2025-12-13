/**
 * Form component for creating new scenario runs.
 *
 * Allows users to input program description and optional overrides.
 */

import { useState } from "react";
import type { ProgramRequest, Asset, CorporateState } from "../types";

interface ScenarioFormProps {
  onSubmit: (request: ProgramRequest) => Promise<void>;
  isSubmitting: boolean;
}

// Default corporate state for demo purposes
const DEFAULT_CORPORATE_STATE: CorporateState = {
  net_debt: 500_000_000,
  ebitda: 200_000_000,
  interest_expense: 25_000_000,
  lease_expense: 10_000_000,
};

// Sample assets for demo purposes
const SAMPLE_ASSETS: Asset[] = [
  {
    asset_id: "A001",
    asset_type: "store",
    market: "Dallas, TX",
    noi: 500_000,
    book_value: 5_000_000,
    criticality: 0.3,
    leaseability_score: 0.8,
    name: "Dallas Store #1",
  },
  {
    asset_id: "A002",
    asset_type: "distribution_center",
    market: "Chicago, IL",
    noi: 1_000_000,
    book_value: 12_000_000,
    criticality: 0.5,
    leaseability_score: 0.7,
    name: "Chicago DC",
  },
  {
    asset_id: "A003",
    asset_type: "store",
    market: "Austin, TX",
    noi: 450_000,
    book_value: 4_500_000,
    criticality: 0.2,
    leaseability_score: 0.85,
    name: "Austin Store #1",
  },
  {
    asset_id: "A004",
    asset_type: "office",
    market: "New York, NY",
    noi: 2_000_000,
    book_value: 25_000_000,
    criticality: 0.7,
    leaseability_score: 0.6,
    name: "NYC Office HQ",
  },
  {
    asset_id: "A005",
    asset_type: "store",
    market: "Phoenix, AZ",
    noi: 380_000,
    book_value: 3_800_000,
    criticality: 0.25,
    leaseability_score: 0.75,
    name: "Phoenix Store #1",
  },
];

export function ScenarioForm({ onSubmit, isSubmitting }: ScenarioFormProps) {
  const [description, setDescription] = useState(
    "Raise $10M via SLB with conservative leverage constraints"
  );
  const [targetOverride, setTargetOverride] = useState("");
  const [maxLeverageOverride, setMaxLeverageOverride] = useState("");
  const [minCoverageOverride, setMinCoverageOverride] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    const request: ProgramRequest = {
      assets: SAMPLE_ASSETS,
      corporate_state: DEFAULT_CORPORATE_STATE,
      program_type: "slb",
      program_description: description,
      target_amount_override: targetOverride
        ? parseFloat(targetOverride)
        : null,
      max_leverage_override: maxLeverageOverride
        ? parseFloat(maxLeverageOverride)
        : null,
      min_coverage_override: minCoverageOverride
        ? parseFloat(minCoverageOverride)
        : null,
    };

    await onSubmit(request);
  };

  return (
    <form className="scenario-form" onSubmit={handleSubmit}>
      <div className="form-section">
        <label htmlFor="description">Program Description</label>
        <textarea
          id="description"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Describe the funding program..."
          rows={3}
          required
        />
        <p className="form-hint">
          Describe the funding goal. The LLM will extract target amounts and
          constraints.
        </p>
      </div>

      <div className="form-section">
        <h4>Optional Overrides</h4>
        <p className="form-hint">
          Override values are "sacred" and cannot be reduced by the optimizer.
        </p>

        <div className="form-row">
          <label htmlFor="targetOverride">Target Amount ($)</label>
          <input
            id="targetOverride"
            type="number"
            value={targetOverride}
            onChange={(e) => setTargetOverride(e.target.value)}
            placeholder="e.g., 10000000"
            min="0"
            step="1000000"
          />
        </div>

        <div className="form-row">
          <label htmlFor="maxLeverageOverride">Max Leverage (x)</label>
          <input
            id="maxLeverageOverride"
            type="number"
            value={maxLeverageOverride}
            onChange={(e) => setMaxLeverageOverride(e.target.value)}
            placeholder="e.g., 4.0"
            min="0"
            step="0.1"
          />
        </div>

        <div className="form-row">
          <label htmlFor="minCoverageOverride">Min Coverage (x)</label>
          <input
            id="minCoverageOverride"
            type="number"
            value={minCoverageOverride}
            onChange={(e) => setMinCoverageOverride(e.target.value)}
            placeholder="e.g., 2.0"
            min="0"
            step="0.1"
          />
        </div>
      </div>

      <button type="submit" className="primary" disabled={isSubmitting}>
        {isSubmitting ? "Creating Run..." : "Create Run"}
      </button>
    </form>
  );
}
