/**
 * Run list panel for the left sidebar.
 *
 * Displays runs with Auquan-style status chips and timestamps.
 */

import type { RunListItem } from "../types";

interface RunsPanelProps {
  runs: RunListItem[];
  selectedRunId: string | null;
  onSelectRun: (runId: string) => void;
  isLoading: boolean;
}

function formatTime(isoString: string): string {
  const date = new Date(isoString);
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function formatDate(isoString: string): string {
  const date = new Date(isoString);
  const now = new Date();
  const isToday = date.toDateString() === now.toDateString();

  if (isToday) {
    return formatTime(isoString);
  }

  return date.toLocaleDateString([], { month: "short", day: "numeric" });
}

function truncate(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength) + "...";
}

export function RunsPanel({
  runs,
  selectedRunId,
  onSelectRun,
  isLoading,
}: RunsPanelProps) {
  if (isLoading) {
    return (
      <div className="runs-panel">
        <div className="runs-panel-loading">
          <span className="spinner-small" />
          <span>Loading runs...</span>
        </div>
      </div>
    );
  }

  if (runs.length === 0) {
    return (
      <div className="runs-panel">
        <div className="runs-panel-empty">
          No runs yet. Create one using the form.
        </div>
      </div>
    );
  }

  return (
    <div className="runs-panel">
      <div className="runs-panel-list">
        {runs.map((run) => (
          <button
            key={run.run_id}
            type="button"
            className={`run-row ${selectedRunId === run.run_id ? "selected" : ""}`}
            onClick={() => onSelectRun(run.run_id)}
          >
            <div className="run-row-header">
              <span className={`run-status-chip ${run.status}`}>
                {run.status === "completed" ? "Completed" : "Failed"}
              </span>
              <span className="run-row-time">{formatDate(run.created_at)}</span>
            </div>
            <div className="run-row-description">
              {truncate(run.program_description, 50)}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
