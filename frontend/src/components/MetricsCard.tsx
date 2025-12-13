/**
 * Metrics card for displaying leverage and coverage ratios.
 *
 * Shows before/after comparison of financial metrics.
 */

import type { ProgramOutcome } from "../types";

interface MetricsCardProps {
  outcome: ProgramOutcome;
}

export function MetricsCard({ outcome }: MetricsCardProps) {
  return (
    <div className="card metrics-card">
      <div className="card-header">Financial Metrics</div>

      <div className="metrics-section">
        <h4>Leverage</h4>
        <div className="metric-comparison">
          <div className="metric-before">
            <span className="label">Before</span>
            <span className="value">
              {formatRatio(outcome.leverage_before)}
            </span>
          </div>
          <span className="arrow">→</span>
          <div className="metric-after">
            <span className="label">After</span>
            <span className="value">
              {formatRatio(outcome.leverage_after)}
            </span>
          </div>
          {outcome.leverage_before !== null &&
            outcome.leverage_after !== null && (
              <span
                className={`change ${outcome.leverage_after < outcome.leverage_before ? "positive" : "negative"}`}
              >
                {outcome.leverage_after < outcome.leverage_before ? "↓" : "↑"}
              </span>
            )}
        </div>
      </div>

      <div className="metrics-section">
        <h4>Interest Coverage</h4>
        <div className="metric-comparison">
          <div className="metric-before">
            <span className="label">Before</span>
            <span className="value">
              {formatRatio(outcome.interest_coverage_before)}
            </span>
          </div>
          <span className="arrow">→</span>
          <div className="metric-after">
            <span className="label">After</span>
            <span className="value">
              {formatRatio(outcome.interest_coverage_after)}
            </span>
          </div>
          {outcome.interest_coverage_before !== null &&
            outcome.interest_coverage_after !== null && (
              <span
                className={`change ${outcome.interest_coverage_after > outcome.interest_coverage_before ? "positive" : "negative"}`}
              >
                {outcome.interest_coverage_after >
                outcome.interest_coverage_before
                  ? "↑"
                  : "↓"}
              </span>
            )}
        </div>
      </div>

      <div className="metrics-section">
        <h4>Fixed Charge Coverage</h4>
        <div className="metric-comparison">
          <div className="metric-before">
            <span className="label">Before</span>
            <span className="value">
              {formatRatio(outcome.fixed_charge_coverage_before)}
            </span>
          </div>
          <span className="arrow">→</span>
          <div className="metric-after">
            <span className="label">After</span>
            <span className="value">
              {formatRatio(outcome.fixed_charge_coverage_after)}
            </span>
          </div>
          {outcome.fixed_charge_coverage_before !== null &&
            outcome.fixed_charge_coverage_after !== null && (
              <span
                className={`change ${outcome.fixed_charge_coverage_after > outcome.fixed_charge_coverage_before ? "positive" : "negative"}`}
              >
                {outcome.fixed_charge_coverage_after >
                outcome.fixed_charge_coverage_before
                  ? "↑"
                  : "↓"}
              </span>
            )}
        </div>
      </div>

      <div className="metric-row">
        <span className="metric-label">Critical Fraction</span>
        <span className="metric-value">
          {(outcome.critical_fraction * 100).toFixed(1)}%
        </span>
      </div>

      {outcome.warnings.length > 0 && (
        <div className="warnings-section">
          <h4>Warnings</h4>
          <ul>
            {outcome.warnings.map((warning, i) => (
              <li key={i}>{warning}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function formatRatio(value: number | null): string {
  if (value === null) return "N/A";
  return `${value.toFixed(2)}x`;
}
