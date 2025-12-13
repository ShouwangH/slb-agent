/**
 * Collapsible section for advanced configuration options.
 *
 * Styled to match the Auquan "Interim Reasoning" expandable pattern.
 */

import type { ReactNode } from "react";

interface AdvancedConfigSectionProps {
  isExpanded: boolean;
  onToggle: () => void;
  children: ReactNode;
  label?: string;
}

export function AdvancedConfigSection({
  isExpanded,
  onToggle,
  children,
  label = "Advanced configuration",
}: AdvancedConfigSectionProps) {
  return (
    <div className="advanced-config-section">
      <button
        type="button"
        className="advanced-config-toggle"
        onClick={onToggle}
        aria-expanded={isExpanded}
      >
        <span className="advanced-config-label">{label}</span>
        <span className={`advanced-config-icon ${isExpanded ? "expanded" : ""}`}>
          {isExpanded ? "−" : "+"}
        </span>
      </button>
      {isExpanded && (
        <div className="advanced-config-content">{children}</div>
      )}
    </div>
  );
}
