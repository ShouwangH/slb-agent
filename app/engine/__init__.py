"""
Engine module for SLB Agent.

Contains pure functions for all metric computation and selection logic.
"""

from app.engine.metrics import (
    check_constraints,
    compute_asset_slb_metrics,
    compute_baseline_metrics,
    compute_critical_fraction,
    compute_post_transaction_metrics,
)
from app.engine.selector import (
    apply_filters,
    compute_score,
    select_assets,
)
from app.engine.explanations import generate_explanation_nodes

__all__ = [
    # Metrics (PR4)
    "compute_asset_slb_metrics",
    "compute_baseline_metrics",
    "compute_post_transaction_metrics",
    "compute_critical_fraction",
    "check_constraints",
    # Selection (PR5)
    "compute_score",
    "apply_filters",
    "select_assets",
    # Explanations (PR6)
    "generate_explanation_nodes",
]
