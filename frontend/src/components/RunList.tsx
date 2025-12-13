/**
 * Run list component for displaying run history.
 *
 * Shows a list of previous runs with status and allows selection.
 */

import type { RunListItem } from "../types";

interface RunListProps {
  runs: RunListItem[];
  selectedRunId: string | null;
  onSelectRun: (runId: string) => void;
  isLoading: boolean;
}

export function RunList({
  runs,
  selectedRunId,
  onSelectRun,
  isLoading,
}: RunListProps) {
  if (isLoading) {
    return (
      <div className="run-list">
        <h3>Run History</h3>
        <div className="loading-spinner">Loading runs...</div>
      </div>
    );
  }

  if (runs.length === 0) {
    return (
      <div className="run-list">
        <h3>Run History</h3>
        <div className="empty-state">
          No runs yet. Create one using the form.
        </div>
      </div>
    );
  }

  return (
    <div className="run-list">
      <h3>Run History</h3>
      <ul className="run-items">
        {runs.map((run) => (
          <li
            key={run.run_id}
            className={`run-item ${selectedRunId === run.run_id ? "selected" : ""}`}
            onClick={() => onSelectRun(run.run_id)}
          >
            <div className="run-item-header">
              <span
                className={`status-badge ${run.status === "completed" ? "completed" : "failed"}`}
              >
                {run.status}
              </span>
              <span className="run-time">
                {formatTime(run.created_at)}
              </span>
            </div>
            <div className="run-description">
              {truncate(run.program_description, 60)}
            </div>
            {run.fund_id && (
              <div className="run-fund">Fund: {run.fund_id}</div>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

function formatTime(isoString: string): string {
  const date = new Date(isoString);
  return date.toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function truncate(str: string, maxLength: number): string {
  if (str.length <= maxLength) return str;
  return str.slice(0, maxLength - 3) + "...";
}
