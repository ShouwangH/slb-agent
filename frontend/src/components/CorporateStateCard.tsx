/**
 * Corporate state card showing current financial metrics.
 *
 * Displays the starting corporate state (leverage, coverage ratios)
 * so users can understand the baseline before any transactions.
 */

import type { CorporateState } from "../types";

interface CorporateStateCardProps {
  corporateState: CorporateState;
}

function formatCurrency(value: number): string {
  if (value >= 1_000_000_000) {
    return `$${(value / 1_000_000_000).toFixed(1)}B`;
  }
  if (value >= 1_000_000) {
    return `$${(value / 1_000_000).toFixed(1)}M`;
  }
  if (value >= 1_000) {
    return `$${(value / 1_000).toFixed(0)}K`;
  }
  return `$${value.toFixed(0)}`;
}

export function CorporateStateCard({ corporateState }: CorporateStateCardProps) {
  const { net_debt, ebitda, interest_expense, lease_expense } = corporateState;

  return (
    <div className="corporate-state-card">
      <div className="corporate-state-header">Corporate State</div>

      <div className="corporate-state-metrics">
        <div className="corporate-state-row">
          <span className="metric-label">Net Debt</span>
          <span className="metric-value">{formatCurrency(net_debt)}</span>
        </div>
        <div className="corporate-state-row">
          <span className="metric-label">EBITDA</span>
          <span className="metric-value">{formatCurrency(ebitda)}</span>
        </div>
        <div className="corporate-state-row">
          <span className="metric-label">Interest Expense</span>
          <span className="metric-value">{formatCurrency(interest_expense)}</span>
        </div>
        {lease_expense != null && (
          <div className="corporate-state-row">
            <span className="metric-label">Lease Expense</span>
            <span className="metric-value">{formatCurrency(lease_expense)}</span>
          </div>
        )}
      </div>
    </div>
  );
}
