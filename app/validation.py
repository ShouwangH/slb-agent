"""
Input validation for SLB Agent.

This module contains validation functions as defined in DESIGN.md Section 7.
All functions are pure and return lists of error messages.
"""

from typing import Optional

from app.models import Asset, CorporateState, SelectorSpec


class ValidationError(Exception):
    """
    Raised when input validation fails.

    Attributes:
        errors: List of validation error messages
    """

    def __init__(self, errors: list[str]):
        self.errors = errors
        message = "; ".join(errors) if errors else "Validation failed"
        super().__init__(message)


# =============================================================================
# Asset Validation (Section 7.1.1)
# =============================================================================


def validate_asset(asset: Asset) -> list[str]:
    """
    Validate a single asset. Returns list of errors.

    Note: Pydantic already enforces bounds (noi > 0, criticality in [0,1], etc.)
    This function adds semantic validation beyond type constraints.
    """
    errors: list[str] = []

    # noi must be positive (Pydantic enforces > 0, but we check for clarity)
    if asset.noi <= 0:
        errors.append(f"Asset {asset.asset_id}: noi must be > 0 (got {asset.noi})")

    # book_value must be positive
    if asset.book_value <= 0:
        errors.append(f"Asset {asset.asset_id}: book_value must be > 0 (got {asset.book_value})")

    # criticality must be in [0, 1]
    if not (0 <= asset.criticality <= 1):
        errors.append(
            f"Asset {asset.asset_id}: criticality must be in [0, 1] (got {asset.criticality})"
        )

    # leaseability_score must be in [0, 1]
    if not (0 <= asset.leaseability_score <= 1):
        errors.append(
            f"Asset {asset.asset_id}: leaseability_score must be in [0, 1] "
            f"(got {asset.leaseability_score})"
        )

    # asset_id must not be empty
    if not asset.asset_id or not asset.asset_id.strip():
        errors.append("Asset has empty asset_id")

    # market must not be empty
    if not asset.market or not asset.market.strip():
        errors.append(f"Asset {asset.asset_id}: market must not be empty")

    return errors


def validate_assets(assets: list[Asset]) -> list[str]:
    """
    Validate asset list. Returns list of errors.

    Checks:
    - List is not empty
    - Asset IDs are unique
    - Each individual asset is valid
    """
    errors: list[str] = []

    if not assets:
        errors.append("Asset list cannot be empty")
        return errors

    # Check for duplicate IDs
    ids = [a.asset_id for a in assets]
    if len(ids) != len(set(ids)):
        # Find the duplicates for better error message
        seen = set()
        duplicates = set()
        for asset_id in ids:
            if asset_id in seen:
                duplicates.add(asset_id)
            seen.add(asset_id)
        errors.append(f"Asset IDs must be unique; duplicates: {sorted(duplicates)}")

    # Validate each asset
    for asset in assets:
        errors.extend(validate_asset(asset))

    return errors


# =============================================================================
# CorporateState Validation (Section 7.1.2)
# =============================================================================


def validate_corporate_state(state: CorporateState) -> list[str]:
    """
    Validate corporate state. Returns list of errors.

    Note: EBITDA can be negative (distressed company) - no hard error,
    but downstream metrics will be None/undefined.
    """
    errors: list[str] = []

    if state.net_debt < 0:
        errors.append(f"net_debt must be >= 0 (got {state.net_debt})")

    # EBITDA can be negative (distressed company), no error
    # But we could add a warning mechanism if needed

    if state.interest_expense < 0:
        errors.append(f"interest_expense must be >= 0 (got {state.interest_expense})")

    if state.lease_expense is not None and state.lease_expense < 0:
        errors.append(f"lease_expense must be >= 0 (got {state.lease_expense})")

    return errors


# =============================================================================
# SelectorSpec Validation (Section 7.2)
# =============================================================================


def validate_spec(spec: SelectorSpec) -> list[str]:
    """
    Validate selector spec. Returns list of validation errors.
    Empty list = valid.

    Note: Pydantic already enforces most bounds. This function adds
    additional semantic validation and clearer error messages.
    """
    errors: list[str] = []

    # Target amount
    if spec.target_amount <= 0:
        errors.append("target_amount must be positive")

    # Hard constraints ranges
    hc = spec.hard_constraints

    if hc.max_net_leverage is not None:
        if not (0 < hc.max_net_leverage < 10):
            errors.append(
                f"max_net_leverage must be in (0, 10) (got {hc.max_net_leverage})"
            )

    if hc.min_interest_coverage is not None:
        if not (0 < hc.min_interest_coverage < 50):
            errors.append(
                f"min_interest_coverage must be in (0, 50) (got {hc.min_interest_coverage})"
            )

    if hc.min_fixed_charge_coverage is not None:
        if not (0 < hc.min_fixed_charge_coverage < 20):
            errors.append(
                f"min_fixed_charge_coverage must be in (0, 20) "
                f"(got {hc.min_fixed_charge_coverage})"
            )

    if hc.max_critical_fraction is not None:
        if not (0 < hc.max_critical_fraction <= 1):
            errors.append(
                f"max_critical_fraction must be in (0, 1] (got {hc.max_critical_fraction})"
            )

    # Soft preferences weights
    sp = spec.soft_preferences

    if sp.weight_criticality < 0:
        errors.append(
            f"weight_criticality must be non-negative (got {sp.weight_criticality})"
        )

    if sp.weight_leaseability < 0:
        errors.append(
            f"weight_leaseability must be non-negative (got {sp.weight_leaseability})"
        )

    # Filter bounds
    af = spec.asset_filters

    if af.min_leaseability_score is not None:
        if not (0 <= af.min_leaseability_score <= 1):
            errors.append(
                f"min_leaseability_score must be in [0, 1] (got {af.min_leaseability_score})"
            )

    if af.max_criticality is not None:
        if not (0 <= af.max_criticality <= 1):
            errors.append(
                f"max_criticality must be in [0, 1] (got {af.max_criticality})"
            )

    # Max iterations
    if spec.max_iterations < 1:
        errors.append(f"max_iterations must be at least 1 (got {spec.max_iterations})")

    if spec.max_iterations > 10:
        errors.append(
            f"max_iterations should not exceed 10 (got {spec.max_iterations})"
        )

    return errors


# =============================================================================
# Convenience Functions
# =============================================================================


def validate_and_raise(
    assets: Optional[list[Asset]] = None,
    corporate_state: Optional[CorporateState] = None,
    spec: Optional[SelectorSpec] = None,
) -> None:
    """
    Validate inputs and raise ValidationError if any errors found.

    This is a convenience function for API boundary validation.
    """
    all_errors: list[str] = []

    if assets is not None:
        all_errors.extend(validate_assets(assets))

    if corporate_state is not None:
        all_errors.extend(validate_corporate_state(corporate_state))

    if spec is not None:
        all_errors.extend(validate_spec(spec))

    if all_errors:
        raise ValidationError(all_errors)
