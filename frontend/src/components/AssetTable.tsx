/**
 * Asset table component for displaying selected assets.
 *
 * Shows the assets selected for the SLB transaction with proceeds and rent.
 */

import type { AssetSelection } from "../types";

interface AssetTableProps {
  selections: AssetSelection[];
  totalProceeds: number;
}

export function AssetTable({ selections, totalProceeds }: AssetTableProps) {
  if (selections.length === 0) {
    return (
      <div className="card asset-table">
        <div className="card-header">Selected Assets</div>
        <div className="empty-state">No assets selected</div>
      </div>
    );
  }

  return (
    <div className="card asset-table">
      <div className="card-header">
        Selected Assets ({selections.length})
      </div>

      <table>
        <thead>
          <tr>
            <th>Asset</th>
            <th>Type</th>
            <th>Market</th>
            <th className="text-right">Proceeds</th>
            <th className="text-right">SLB Rent</th>
          </tr>
        </thead>
        <tbody>
          {selections.map((selection) => (
            <tr key={selection.asset.asset_id}>
              <td>
                <div className="asset-name">
                  {selection.asset.name || selection.asset.asset_id}
                </div>
                <div className="asset-id text-muted text-xs">
                  {selection.asset.asset_id}
                </div>
              </td>
              <td>
                <span className={`asset-type-badge ${selection.asset.asset_type}`}>
                  {formatAssetType(selection.asset.asset_type)}
                </span>
              </td>
              <td>{selection.asset.market}</td>
              <td className="text-right font-mono">
                {formatCurrency(selection.proceeds)}
              </td>
              <td className="text-right font-mono">
                {formatCurrency(selection.slb_rent)}
              </td>
            </tr>
          ))}
        </tbody>
        <tfoot>
          <tr>
            <td colSpan={3}>
              <strong>Total</strong>
            </td>
            <td className="text-right font-mono">
              <strong>{formatCurrency(totalProceeds)}</strong>
            </td>
            <td className="text-right font-mono">
              <strong>
                {formatCurrency(
                  selections.reduce((sum, s) => sum + s.slb_rent, 0)
                )}
              </strong>
            </td>
          </tr>
        </tfoot>
      </table>
    </div>
  );
}

function formatCurrency(value: number): string {
  if (value >= 1_000_000) {
    return `$${(value / 1_000_000).toFixed(2)}M`;
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
