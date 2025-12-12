/**
 * Compact asset list panel for the left sidebar.
 *
 * Displays assets in a minimal table format styled like Auquan's "Top Risks" blocks.
 */

import type { Asset } from "../types";

interface AssetsPanelProps {
  assets: Asset[];
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

function formatAssetType(type: string): string {
  return type
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

export function AssetsPanel({ assets }: AssetsPanelProps) {
  if (assets.length === 0) {
    return (
      <div className="assets-panel">
        <div className="assets-panel-empty">No assets available</div>
      </div>
    );
  }

  return (
    <div className="assets-panel">
      <div className="assets-panel-list">
        {assets.map((asset) => (
          <div key={asset.asset_id} className="asset-row">
            <div className="asset-row-main">
              <span className="asset-row-name">
                {asset.name || asset.asset_id}
              </span>
              <span className={`asset-type-chip ${asset.asset_type}`}>
                {formatAssetType(asset.asset_type)}
              </span>
            </div>
            <div className="asset-row-details">
              <span className="asset-row-market">{asset.market}</span>
              <span className="asset-row-value">
                {formatCurrency(asset.book_value)}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
