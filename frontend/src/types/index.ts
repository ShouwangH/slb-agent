/**
 * TypeScript types for SLB Agent frontend.
 *
 * These types match the backend Python models exactly.
 * IMPORTANT: When backend models change, update these types to match.
 */

// =============================================================================
// Enums (match backend exactly)
// =============================================================================

export type SelectionStatus = "ok" | "infeasible" | "numeric_error";

export type AssetType =
  | "store"
  | "distribution_center"
  | "office"
  | "mixed_use"
  | "other";

export type MarketTier = 1 | 2 | 3;

export type ProgramType = "slb";

export type Objective = "maximize_proceeds" | "minimize_risk" | "balanced";

export type TargetSource = "user_override" | "llm_extraction";

// =============================================================================
// Policy Violation Codes (match PolicyViolationCode enum)
// =============================================================================

export type PolicyViolationCode =
  | "target_increased"
  | "target_drop_exceeded"
  | "target_below_floor"
  | "leverage_relaxed"
  | "interest_coverage_relaxed"
  | "fixed_charge_coverage_relaxed"
  | "critical_fraction_relaxed"
  | "constraint_deleted"
  | "criticality_step_exceeded"
  | "leaseability_step_exceeded"
  | "filter_deleted"
  | "program_type_changed";

// =============================================================================
// Input Models
// =============================================================================

export interface Asset {
  asset_id: string;
  asset_type: AssetType;
  market: string;
  noi: number;
  book_value: number;
  criticality: number;
  leaseability_score: number;
  name?: string | null;
  country?: string | null;
  market_tier?: MarketTier | null;
  tenant_name?: string | null;
  tenant_credit_score?: number | null;
  wault_years?: number | null;
  demographic_index?: number | null;
  esg_risk_score?: number | null;
  current_ltv?: number | null;
  existing_debt_amount?: number | null;
  encumbrance_type?: string | null;
}

export interface CorporateState {
  net_debt: number;
  ebitda: number;
  interest_expense: number;
  lease_expense?: number | null;
}

// =============================================================================
// Spec Models
// =============================================================================

export interface HardConstraints {
  max_net_leverage?: number | null;
  min_interest_coverage?: number | null;
  min_fixed_charge_coverage?: number | null;
  max_critical_fraction?: number | null;
}

export interface SoftPreferences {
  prefer_low_criticality: boolean;
  prefer_high_leaseability: boolean;
  weight_criticality: number;
  weight_leaseability: number;
}

export interface AssetFilters {
  include_asset_types?: AssetType[] | null;
  exclude_asset_types?: AssetType[] | null;
  exclude_markets?: string[] | null;
  min_leaseability_score?: number | null;
  max_criticality?: number | null;
}

export interface SelectorSpec {
  program_type: ProgramType;
  objective: Objective;
  target_amount: number;
  hard_constraints: HardConstraints;
  soft_preferences: SoftPreferences;
  asset_filters: AssetFilters;
  time_horizon_years?: number | null;
  max_iterations: number;
}

// =============================================================================
// Constraint Violation (from engine, matches ConstraintViolation model)
// =============================================================================

export interface ConstraintViolation {
  code: string;
  detail: string;
  actual: number;
  limit: number;
}

// =============================================================================
// Policy Violation (structured, matches PolicyViolation model)
// =============================================================================

export interface PolicyViolation {
  code: PolicyViolationCode;
  detail: string;
  field: string;
  attempted: number | null;
  limit: number | null;
  adjusted_to: number | null;
}

// =============================================================================
// Audit Trace Types (match backend exactly)
// =============================================================================

export interface SpecSnapshot {
  target_amount: number;
  max_criticality: number | null;
  min_leaseability_score: number | null;
  max_net_leverage: number | null;
  min_interest_coverage: number | null;
  min_fixed_charge_coverage: number | null;
  max_critical_fraction: number | null;
}

export interface OutcomeSnapshot {
  status: SelectionStatus;
  proceeds: number;
  leverage_after: number | null;
  interest_coverage_after: number | null;
  fixed_charge_coverage_after: number | null;
  critical_fraction: number;
  violations: ConstraintViolation[];
}

export interface AuditTraceEntry {
  iteration: number;
  phase: "initial" | "revision";
  spec_snapshot: SpecSnapshot;
  outcome_snapshot: OutcomeSnapshot;
  policy_violations: PolicyViolation[];
  target_before: number | null;
  target_after: number;
  timestamp: string;
}

export interface AuditTrace {
  entries: AuditTraceEntry[];
  original_target: number;
  floor_target: number;
  floor_fraction: number;
  target_source: TargetSource;
  started_at: string;
  completed_at: string | null;
}

// =============================================================================
// Output Models
// =============================================================================

export interface AssetSelection {
  asset: Asset;
  proceeds: number;
  slb_rent: number;
}

export interface ProgramOutcome {
  status: SelectionStatus;
  selected_assets: AssetSelection[];
  proceeds: number;

  // Pre-transaction metrics
  leverage_before: number | null;
  interest_coverage_before: number | null;
  fixed_charge_coverage_before: number | null;

  // Post-transaction metrics
  leverage_after: number | null;
  interest_coverage_after: number | null;
  fixed_charge_coverage_after: number | null;

  critical_fraction: number;
  violations: ConstraintViolation[];
  warnings: string[];
}

// =============================================================================
// Explanation Models
// =============================================================================

export type ExplanationSeverity = "info" | "warning" | "error";
export type ExplanationCategory = "constraint" | "driver" | "risk" | "alternative";

export interface ExplanationNode {
  id: string;
  label: string;
  severity: ExplanationSeverity;
  category: ExplanationCategory;
  metric?: string | null;
  baseline_value?: number | null;
  post_value?: number | null;
  threshold?: number | null;
  asset_ids?: string[] | null;
  detail?: string | null;
}

export interface Explanation {
  summary: string;
  nodes: ExplanationNode[];
}

// =============================================================================
// API Response Types
// =============================================================================

export interface ProgramResponse {
  selector_spec: SelectorSpec;
  outcome: ProgramOutcome;
  explanation: Explanation;
  audit_trace: AuditTrace | null;
}

// =============================================================================
// API Request Types
// =============================================================================

export interface ProgramRequest {
  assets: Asset[];
  corporate_state: CorporateState;
  program_type: ProgramType;
  program_description: string;
  floor_override?: number | null;
  max_leverage_override?: number | null;
  min_coverage_override?: number | null;
}

// =============================================================================
// Run Types (for /api/runs endpoints)
// =============================================================================

export interface RunRecord {
  run_id: string;
  fund_id: string | null;
  program_description: string;
  status: "completed" | "failed";
  response: ProgramResponse | null;
  error: string | null;
  created_at: string;
  // Scenario metadata (null = single-scenario run, not part of any set)
  scenario_set_id: string | null;
  scenario_kind: ScenarioKind | null;
  scenario_label: string | null;
}

export interface RunListItem {
  run_id: string;
  fund_id: string | null;
  program_description: string;
  status: "completed" | "failed";
  created_at: string;
  // Scenario metadata (null = single-scenario run, not part of any set)
  scenario_set_id: string | null;
  scenario_kind: ScenarioKind | null;
  scenario_label: string | null;
}

export interface CreateRunResponse {
  run_id: string;
  status: "completed" | "failed";
  response?: ProgramResponse;
  error?: string;
}

// =============================================================================
// Error Response
// =============================================================================

export interface ErrorResponse {
  error: string;
  detail?: string;
  code: string;
}

// =============================================================================
// Scenario Types (Multi-Scenario Orchestration)
// =============================================================================

export type ScenarioKind =
  | "base"
  | "risk_off"
  | "aggressive"
  | "geo_focus"
  | "custom";

export interface ScenarioSetSummary {
  id: string;
  brief: string;
  created_at: string;
  run_ids: string[];
}

