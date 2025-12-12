/**
 * Main scenario planner page with three-column layout.
 *
 * Left: Run list sidebar
 * Center: Main content (form, audit timeline, assets)
 * Right: Metrics and invariants
 */

import { useState, useEffect, useCallback } from "react";
import {
  ScenarioForm,
  RunList,
  LoadingSpinner,
  AuditTraceTimeline,
  NumericInvariantsCard,
  MetricsCard,
  AssetTable,
} from "../components";
import { createRun, getRun, listRuns, ApiError } from "../api/runs";
import type {
  ProgramRequest,
  RunListItem,
  RunRecord,
  ProgramResponse,
} from "../types";

export function ScenarioPlannerPage() {
  // Run list state
  const [runs, setRuns] = useState<RunListItem[]>([]);
  const [isLoadingRuns, setIsLoadingRuns] = useState(true);

  // Selected run state
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [selectedRun, setSelectedRun] = useState<RunRecord | null>(null);
  const [isLoadingRun, setIsLoadingRun] = useState(false);

  // Form state
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Error state
  const [error, setError] = useState<string | null>(null);

  // Load runs on mount
  useEffect(() => {
    refreshRunList();
  }, []);

  // Load selected run details
  useEffect(() => {
    if (selectedRunId) {
      loadRunDetails(selectedRunId);
    } else {
      setSelectedRun(null);
    }
  }, [selectedRunId]);

  const refreshRunList = async () => {
    setIsLoadingRuns(true);
    try {
      const runList = await listRuns({ limit: 20 });
      setRuns(runList);
    } catch (e) {
      console.error("Failed to load runs:", e);
      setError("Failed to load run history");
    } finally {
      setIsLoadingRuns(false);
    }
  };

  const loadRunDetails = async (runId: string) => {
    setIsLoadingRun(true);
    setError(null);
    try {
      const run = await getRun(runId);
      setSelectedRun(run);
    } catch (e) {
      if (e instanceof ApiError && e.status === 404) {
        setError("Run not found");
      } else {
        setError("Failed to load run details");
      }
      setSelectedRun(null);
    } finally {
      setIsLoadingRun(false);
    }
  };

  const handleSubmit = useCallback(async (request: ProgramRequest) => {
    setIsSubmitting(true);
    setError(null);
    try {
      const result = await createRun(request);

      if (result.status === "failed") {
        setError(`Run failed: ${result.error}`);
      } else {
        // Refresh run list and select the new run
        await refreshRunList();
        setSelectedRunId(result.run_id);
      }
    } catch (e) {
      if (e instanceof ApiError) {
        if (e.status === 503) {
          setError("Service temporarily unavailable. Please try again.");
        } else {
          setError(`Error: ${e.detail || e.message}`);
        }
      } else {
        setError("An unexpected error occurred");
      }
    } finally {
      setIsSubmitting(false);
    }
  }, []);

  const handleSelectRun = useCallback((runId: string) => {
    setSelectedRunId(runId);
    setError(null);
  }, []);

  const response: ProgramResponse | null = selectedRun?.response ?? null;

  return (
    <div className="scenario-planner">
      {/* Left Sidebar: Run History */}
      <aside className="sidebar-left">
        <RunList
          runs={runs}
          selectedRunId={selectedRunId}
          onSelectRun={handleSelectRun}
          isLoading={isLoadingRuns}
        />
      </aside>

      {/* Main Content */}
      <main className="main-content">
        {error && (
          <div className="error-banner">
            {error}
            <button
              className="error-dismiss"
              onClick={() => setError(null)}
              aria-label="Dismiss error"
            >
              ×
            </button>
          </div>
        )}

        <ScenarioForm onSubmit={handleSubmit} isSubmitting={isSubmitting} />

        {isLoadingRun && <LoadingSpinner message="Loading run details..." />}

        {!isLoadingRun && selectedRun && (
          <div className="run-details">
            <div className="run-details-header">
              <h2>Run Details</h2>
              <span
                className={`status-badge ${selectedRun.status === "completed" ? "completed" : "failed"}`}
              >
                {selectedRun.status}
              </span>
            </div>

            {selectedRun.error && (
              <div className="error-banner">{selectedRun.error}</div>
            )}

            {response && (
              <>
                {/* Explanation Summary */}
                <div className="card explanation-card">
                  <div className="card-header">Summary</div>
                  <p>{response.explanation.summary}</p>
                </div>

                {/* Audit Timeline */}
                {response.audit_trace && (
                  <AuditTraceTimeline auditTrace={response.audit_trace} />
                )}

                {/* Selected Assets */}
                <AssetTable
                  selections={response.outcome.selected_assets}
                  totalProceeds={response.outcome.proceeds}
                />
              </>
            )}
          </div>
        )}

        {!isLoadingRun && !selectedRun && !isSubmitting && (
          <div className="empty-state">
            <p>Select a run from the history or create a new one.</p>
          </div>
        )}
      </main>

      {/* Right Sidebar: Metrics */}
      <aside className="sidebar-right">
        {response && response.audit_trace && (
          <>
            <NumericInvariantsCard
              auditTrace={response.audit_trace}
              finalProceeds={response.outcome.proceeds}
            />
            <MetricsCard outcome={response.outcome} />
          </>
        )}

        {!response && (
          <div className="empty-state">
            <p>Metrics will appear here when a run is selected.</p>
          </div>
        )}
      </aside>
    </div>
  );
}
