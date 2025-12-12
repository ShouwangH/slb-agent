/**
 * Left sidebar container component.
 *
 * Hosts the section navigation and conditionally displays
 * either the Assets panel or Runs panel based on active section.
 */

import { useState } from "react";
import { SectionNav, type SidebarSection } from "./SectionNav";
import { AssetsPanel } from "./AssetsPanel";
import { RunsPanel } from "./RunsPanel";
import type { Asset, RunListItem } from "../types";

interface LeftSidebarProps {
  assets: Asset[];
  runs: RunListItem[];
  selectedRunId: string | null;
  onSelectRun: (runId: string) => void;
  isLoadingRuns: boolean;
}

export function LeftSidebar({
  assets,
  runs,
  selectedRunId,
  onSelectRun,
  isLoadingRuns,
}: LeftSidebarProps) {
  const [activeSection, setActiveSection] = useState<SidebarSection>("runs");

  return (
    <div className="left-sidebar">
      <SectionNav
        activeSection={activeSection}
        onSectionChange={setActiveSection}
        runCount={runs.length}
        assetCount={assets.length}
      />

      <div className="left-sidebar-content">
        {activeSection === "assets" && <AssetsPanel assets={assets} />}
        {activeSection === "runs" && (
          <RunsPanel
            runs={runs}
            selectedRunId={selectedRunId}
            onSelectRun={onSelectRun}
            isLoading={isLoadingRuns}
          />
        )}
      </div>
    </div>
  );
}
