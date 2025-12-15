/**
 * Section navigation component styled like the Auquan "Sections" sidebar.
 *
 * Displays a card with expandable section rows for Assets and Runs.
 */

import { useState } from "react";
import type { CorporateState } from "../types";

export type SidebarSection = "assets" | "runs";

interface SectionNavProps {
  activeSection: SidebarSection;
  onSectionChange: (section: SidebarSection) => void;
  runCount?: number;
  assetCount?: number;
  corporateState?: CorporateState;
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

export function SectionNav({
  activeSection,
  onSectionChange,
  runCount = 0,
  assetCount = 0,
  corporateState,
}: SectionNavProps) {
  const [corporateExpanded, setCorporateExpanded] = useState(true);

  return (
    <div className="section-nav">
      <div className="section-nav-header">
        <span>Sections</span>
        <span className="section-nav-icon">
          <svg
            width="16"
            height="16"
            viewBox="0 0 16 16"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              d="M4 6L8 10L12 6"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </span>
      </div>
      <div className="section-nav-items">
        {/* Corporate State - collapsible inline */}
        {corporateState && (
          <>
            <button
              type="button"
              className="section-row"
              onClick={() => setCorporateExpanded(!corporateExpanded)}
            >
              <span className={`section-chevron ${corporateExpanded ? "expanded" : ""}`}>
                <svg
                  width="12"
                  height="12"
                  viewBox="0 0 12 12"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    d="M4.5 3L7.5 6L4.5 9"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </span>
              <span className="section-label">Corporate State</span>
            </button>
            {corporateExpanded && (
              <div className="section-inline-content">
                <div className="corporate-inline-row">
                  <span>Net Debt</span>
                  <span>{formatCurrency(corporateState.net_debt)}</span>
                </div>
                <div className="corporate-inline-row">
                  <span>EBITDA</span>
                  <span>{formatCurrency(corporateState.ebitda)}</span>
                </div>
                <div className="corporate-inline-row">
                  <span>Interest Expense</span>
                  <span>{formatCurrency(corporateState.interest_expense)}</span>
                </div>
                {corporateState.lease_expense != null && (
                  <div className="corporate-inline-row">
                    <span>Lease Expense</span>
                    <span>{formatCurrency(corporateState.lease_expense)}</span>
                  </div>
                )}
              </div>
            )}
          </>
        )}
        <button
          type="button"
          className={`section-row ${activeSection === "assets" ? "active" : ""}`}
          onClick={() => onSectionChange("assets")}
        >
          <span className={`section-chevron ${activeSection === "assets" ? "expanded" : ""}`}>
            <svg
              width="12"
              height="12"
              viewBox="0 0 12 12"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                d="M4.5 3L7.5 6L4.5 9"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </span>
          <span className="section-label">Assets</span>
          {assetCount > 0 && (
            <span className="section-count">{assetCount}</span>
          )}
        </button>
        <button
          type="button"
          className={`section-row ${activeSection === "runs" ? "active" : ""}`}
          onClick={() => onSectionChange("runs")}
        >
          <span className={`section-chevron ${activeSection === "runs" ? "expanded" : ""}`}>
            <svg
              width="12"
              height="12"
              viewBox="0 0 12 12"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                d="M4.5 3L7.5 6L4.5 9"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </span>
          <span className="section-label">Runs</span>
          {runCount > 0 && (
            <span className="section-count">{runCount}</span>
          )}
        </button>
      </div>
    </div>
  );
}
