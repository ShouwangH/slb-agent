/**
 * Audit trace timeline component.
 *
 * Displays the iteration history of the optimization loop with spec changes,
 * outcomes, and policy violations at each step.
 */

import type { AuditTrace, AuditTraceEntry } from "../types";

interface AuditTraceTimelineProps {
  auditTrace: AuditTrace;
}

export function AuditTraceTimeline({ auditTrace }: AuditTraceTimelineProps) {
  const { entries, original_target, floor_target, floor_fraction, target_source } =
    auditTrace;

  if (entries.length === 0) {
    return (
      <div className="audit-timeline">
        <div className="card-header">Optimization Timeline</div>
        <div className="empty-state">No iterations recorded.</div>
      </div>
    );
  }

  return (
    <div className="audit-timeline">
      <div className="timeline-header">
        <h3>Optimization Timeline</h3>
        <div className="invariants-summary">
          <div className="target-info">
            <span>Original: {formatCurrency(original_target)}</span>
            <span className={`source-badge ${target_source}`}>
              {target_source === "user_override" ? "User Override" : "LLM Extracted"}
            </span>
          </div>
          <div className="floor-info">
            Floor ({(floor_fraction * 100).toFixed(0)}%):{" "}
            {formatCurrency(floor_target)}
            {target_source === "user_override" && (
              <span className="sacred-badge">Sacred</span>
            )}
          </div>
        </div>
      </div>

      <div className="timeline-entries">
        {entries.map((entry, index) => (
          <TimelineEntry
            key={entry.iteration}
            entry={entry}
            isLast={index === entries.length - 1}
          />
        ))}
      </div>

      <div className="timeline-footer">
        <span>Completed: {formatTimestamp(auditTrace.completed_at)}</span>
        <span>Duration: {calculateDuration(auditTrace.started_at, auditTrace.completed_at)}</span>
      </div>
    </div>
  );
}

interface TimelineEntryProps {
  entry: AuditTraceEntry;
  isLast: boolean;
}

function TimelineEntry({ entry, isLast }: TimelineEntryProps) {
  const { outcome_snapshot, policy_violations } = entry;

  return (
    <div className={`timeline-entry ${entry.phase} ${isLast ? "final" : ""}`}>
      <div className="entry-marker">
        <div className={`marker-dot ${outcome_snapshot.status}`} />
        {!isLast && <div className="marker-line" />}
      </div>

      <div className="entry-content">
        <div className="entry-header">
          <span className="iteration-label">
            {entry.phase === "initial" ? "Initial" : `Revision ${entry.iteration}`}
          </span>
          <span className={`status-badge ${outcome_snapshot.status}`}>
            {outcome_snapshot.status}
          </span>
          <span className="timestamp">{formatTimestamp(entry.timestamp)}</span>
        </div>

        <div className="entry-target">
          {entry.target_before !== null && (
            <>
              <span className="before">{formatCurrency(entry.target_before)}</span>
              <span className="arrow">→</span>
            </>
          )}
          <span className="after">{formatCurrency(entry.target_after)}</span>
        </div>

        <div className="entry-metrics">
          <div className="metric">
            <span className="label">Proceeds</span>
            <span className="value">{formatCurrency(outcome_snapshot.proceeds)}</span>
          </div>
          {outcome_snapshot.leverage_after !== null && (
            <div className="metric">
              <span className="label">Leverage</span>
              <span className="value">{outcome_snapshot.leverage_after.toFixed(2)}x</span>
            </div>
          )}
          {outcome_snapshot.interest_coverage_after !== null && (
            <div className="metric">
              <span className="label">Int Cov</span>
              <span className="value">
                {outcome_snapshot.interest_coverage_after.toFixed(2)}x
              </span>
            </div>
          )}
          {outcome_snapshot.fixed_charge_coverage_after !== null && (
            <div className="metric">
              <span className="label">FCC</span>
              <span className="value">
                {outcome_snapshot.fixed_charge_coverage_after.toFixed(2)}x
              </span>
            </div>
          )}
        </div>

        {/* Engine constraint violations */}
        {outcome_snapshot.violations.length > 0 && (
          <div className="violations engine-violations">
            <div className="violations-label">Constraint Violations:</div>
            {outcome_snapshot.violations.map((v, i) => (
              <div key={i} className="violation">
                <code>{v.code}</code>: {v.detail}
                <span className="violation-values">
                  (actual: {v.actual.toFixed(2)}, limit: {v.limit.toFixed(2)})
                </span>
              </div>
            ))}
          </div>
        )}

        {/* Policy violations (deterministic bounds) */}
        {policy_violations.length > 0 && (
          <div className="violations policy-violations">
            <div className="violations-label">Policy Adjustments:</div>
            {policy_violations.map((v, i) => (
              <div
                key={i}
                className={`violation ${v.adjusted_to !== null ? "clamped" : "rejected"}`}
              >
                <code>{v.code}</code>: {v.field}
                {v.attempted !== null && (
                  <span> tried {formatValue(v.attempted)}</span>
                )}
                {v.limit !== null && (
                  <span> (limit: {formatValue(v.limit)})</span>
                )}
                {v.adjusted_to !== null && (
                  <span className="adjustment">
                    → clamped to {formatValue(v.adjusted_to)}
                  </span>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
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

function formatValue(value: number): string {
  if (value >= 1_000_000) {
    return `$${(value / 1_000_000).toFixed(1)}M`;
  }
  if (value >= 1_000) {
    return `$${(value / 1_000).toFixed(0)}K`;
  }
  // Could be a ratio or percentage
  if (value < 10) {
    return value.toFixed(2);
  }
  return value.toFixed(0);
}

function formatTimestamp(isoString: string | null): string {
  if (!isoString) return "—";
  const date = new Date(isoString);
  return date.toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function calculateDuration(start: string, end: string | null): string {
  if (!end) return "—";
  const startDate = new Date(start);
  const endDate = new Date(end);
  const diffMs = endDate.getTime() - startDate.getTime();
  if (diffMs < 1000) {
    return `${diffMs}ms`;
  }
  return `${(diffMs / 1000).toFixed(1)}s`;
}
