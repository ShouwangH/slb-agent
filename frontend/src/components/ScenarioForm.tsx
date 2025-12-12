/**
 * Form component for creating new scenario runs.
 *
 * Allows users to input program description and optional overrides.
 */

import { useState } from "react";
import type { ProgramRequest } from "../types";
import { AdvancedConfigSection } from "./AdvancedConfigSection";
import { SAMPLE_ASSETS, DEFAULT_CORPORATE_STATE } from "../data/sampleAssets";

interface ScenarioFormProps {
  onSubmit: (request: ProgramRequest) => Promise<void>;
  isSubmitting: boolean;
}

export function ScenarioForm({ onSubmit, isSubmitting }: ScenarioFormProps) {
  const [description, setDescription] = useState(
    "Raise $60M via SLB with conservative leverage constraints"
  );
  const [floorOverride, setFloorOverride] = useState("");
  const [maxLeverageOverride, setMaxLeverageOverride] = useState("");
  const [minCoverageOverride, setMinCoverageOverride] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    const request: ProgramRequest = {
      assets: SAMPLE_ASSETS,
      corporate_state: DEFAULT_CORPORATE_STATE,
      program_type: "slb",
      program_description: description,
      floor_override: floorOverride ? parseFloat(floorOverride) : null,
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

      <AdvancedConfigSection
        isExpanded={showAdvanced}
        onToggle={() => setShowAdvanced(!showAdvanced)}
      >
        <p className="form-hint">
          Set a minimum acceptable amount if you're flexible on the target.
          Leave empty to require the exact amount from your description.
        </p>

        <div className="form-row">
          <label htmlFor="floorOverride">Minimum Acceptable Amount ($)</label>
          <input
            id="floorOverride"
            type="number"
            value={floorOverride}
            onChange={(e) => setFloorOverride(e.target.value)}
            placeholder="e.g., 8000000 (or leave empty for exact target)"
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
      </AdvancedConfigSection>

      <button type="submit" className="primary" disabled={isSubmitting}>
        {isSubmitting ? "Creating Run..." : "Create Run"}
      </button>
    </form>
  );
}
