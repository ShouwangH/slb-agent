/**
 * Numeric invariants card showing target/floor/source information.
 *
 * Displays the deterministic bounds that govern the optimization loop.
 */

import type { AuditTrace } from "../types";

interface NumericInvariantsCardProps {
  auditTrace: AuditTrace;
  finalProceeds: number;
}

export function NumericInvariantsCard({
  auditTrace,
  finalProceeds,
}: NumericInvariantsCardProps) {
  const { original_target, floor_target, floor_fraction, target_source } =
    auditTrace;
  const lastEntry = auditTrace.entries[auditTrace.entries.length - 1];
  const currentTarget = lastEntry?.target_after ?? original_target;

  const isSacred = target_source === "user_override";
  const targetDelta =
    currentTarget !== original_target
      ? ((currentTarget - original_target) / original_target) * 100
      : null;

  return (
    <div className="card numeric-invariants-card">
      <div className="card-header">Numeric Invariants</div>

      <div className="metric-row">
        <span className="metric-label">
          Requested Target
          <span className={`source-badge ${target_source}`}>
            {isSacred ? "User Override" : "LLM Extracted"}
          </span>
        </span>
        <span className="metric-value">{formatCurrency(original_target)}</span>
      </div>

      <div className="metric-row">
        <span className="metric-label">Current Target</span>
        <span className="metric-value">
          {formatCurrency(currentTarget)}
          {targetDelta !== null && (
            <span className={`metric-delta ${targetDelta < 0 ? "negative" : "positive"}`}>
              {targetDelta > 0 ? "+" : ""}
              {targetDelta.toFixed(1)}%
            </span>
          )}
        </span>
      </div>

      <div className="metric-row">
        <span className="metric-label">
          Floor ({(floor_fraction * 100).toFixed(0)}% of original)
          {isSacred && <span className="sacred-badge">Sacred</span>}
        </span>
        <span className="metric-value">{formatCurrency(floor_target)}</span>
      </div>

      <div className="metric-row">
        <span className="metric-label">Actual Proceeds</span>
        <span className="metric-value">{formatCurrency(finalProceeds)}</span>
      </div>

      {currentTarget === floor_target && !isSacred && (
        <div className="invariant-warning">
          Floor reached - cannot reduce target further
        </div>
      )}

      {isSacred && (
        <div className="invariant-note">
          User override is sacred - target cannot be reduced
        </div>
      )}
    </div>
  );
}

function formatCurrency(value: number): string {
  if (value >= 1_000_000) {
    return `$${(value / 1_000_000).toFixed(1)}M`;
  }
  if (value >= 1_000) {
    return `$${(value / 1_000).toFixed(0)}K`;
  }
  return `$${value.toFixed(0)}`;
}
