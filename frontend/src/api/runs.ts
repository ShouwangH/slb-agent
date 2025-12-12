/**
 * API client for the runs endpoints.
 *
 * Handles communication with the backend /api/runs endpoints.
 */

import type {
  CreateRunResponse,
  ErrorResponse,
  ProgramRequest,
  RunListItem,
  RunRecord,
} from "../types";

const API_BASE = "/api";

/**
 * API error class for handling backend errors.
 */
export class ApiError extends Error {
  status: number;
  code: string;
  detail?: string;

  constructor(status: number, code: string, detail?: string) {
    super(`API Error ${status}: ${code}`);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.detail = detail;
  }
}

/**
 * Handle API response and throw ApiError on failure.
 */
async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorData: ErrorResponse | undefined;
    try {
      errorData = await response.json();
    } catch {
      // Response body was not JSON
    }

    throw new ApiError(
      response.status,
      errorData?.code ?? "UNKNOWN_ERROR",
      errorData?.detail ?? response.statusText
    );
  }

  return response.json();
}

/**
 * Create and execute a program run.
 *
 * @param request - The program request with assets, corporate state, etc.
 * @param fundId - Optional fund identifier for grouping runs.
 * @returns The created run with status and response or error.
 *
 * @example
 * ```typescript
 * const result = await createRun({
 *   assets: [...],
 *   corporate_state: {...},
 *   program_type: "slb",
 *   program_description: "Raise $50M via SLB",
 * });
 *
 * if (result.status === "completed") {
 *   console.log("Proceeds:", result.response?.outcome.proceeds);
 * } else {
 *   console.error("Failed:", result.error);
 * }
 * ```
 */
export async function createRun(
  request: ProgramRequest,
  fundId?: string
): Promise<CreateRunResponse> {
  const url = new URL(`${API_BASE}/runs`, window.location.origin);
  if (fundId) {
    url.searchParams.set("fund_id", fundId);
  }

  const response = await fetch(url.toString(), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });

  return handleResponse<CreateRunResponse>(response);
}

/**
 * Get a single run by ID.
 *
 * @param runId - The UUID of the run to retrieve.
 * @returns The full run record including response and audit trace.
 * @throws ApiError with status 404 if run not found.
 *
 * @example
 * ```typescript
 * const run = await getRun("550e8400-e29b-41d4-a716-446655440000");
 * if (run.response?.audit_trace) {
 *   console.log("Iterations:", run.response.audit_trace.entries.length);
 * }
 * ```
 */
export async function getRun(runId: string): Promise<RunRecord> {
  const response = await fetch(`${API_BASE}/runs/${runId}`);
  return handleResponse<RunRecord>(response);
}

/**
 * List runs with optional filtering.
 *
 * @param options - Filter options.
 * @param options.fundId - Filter by fund ID.
 * @param options.limit - Maximum number of runs to return (default: 10).
 * @returns List of run summaries (without full response payload).
 *
 * @example
 * ```typescript
 * // Get all runs
 * const runs = await listRuns();
 *
 * // Filter by fund
 * const fundRuns = await listRuns({ fundId: "FUND123" });
 *
 * // Limit results
 * const recentRuns = await listRuns({ limit: 5 });
 * ```
 */
export async function listRuns(options?: {
  fundId?: string;
  limit?: number;
}): Promise<RunListItem[]> {
  const url = new URL(`${API_BASE}/runs`, window.location.origin);

  if (options?.fundId) {
    url.searchParams.set("fund_id", options.fundId);
  }
  if (options?.limit !== undefined) {
    url.searchParams.set("limit", options.limit.toString());
  }

  const response = await fetch(url.toString());
  return handleResponse<RunListItem[]>(response);
}
