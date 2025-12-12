"""
Revision Policy for SLB Agent.

This module enforces constraints on what the LLM may change during spec revision.
It is a standalone, pure logic module with no dependencies on engine or LLM.

Defined in DESIGN.md Section 8.2-8.3.

Policy Rules:
- program_type: Never change (immutable)
- hard_constraints: Cannot relax beyond original (immutable), cannot delete if originally set
- target_amount: Can only decrease, max 20% per iteration, min 75% of original
- asset_filters: Bounded relaxation per iteration (max_criticality, min_leaseability_score only)
  - Cannot delete filter if previously set (None not allowed if was numeric)
  - Default baseline of 0.5 used when previous filter was None
- soft_preferences: Unrestricted
"""

from dataclasses import dataclass
from typing import Optional

from app.config import DEFAULT_REVISION_POLICY_CONFIG, RevisionPolicyConfig
from app.models import HardConstraints, SelectorSpec


@dataclass
class PolicyResult:
    """
    Result of revision policy enforcement.

    Attributes:
        valid: Whether the revision is acceptable (possibly after adjustments)
        spec: The adjusted spec (None if valid=False)
        violations: List of policy violations found (may be non-empty even if valid=True)
    """

    valid: bool
    spec: Optional[SelectorSpec]
    violations: list[str]


def enforce_revision_policy(
    immutable_hard: HardConstraints,
    original_target: float,
    prev_spec: SelectorSpec,
    new_spec: SelectorSpec,
    config: RevisionPolicyConfig = DEFAULT_REVISION_POLICY_CONFIG,
) -> PolicyResult:
    """
    Enforce constraints on what the LLM may change during revision.

    Returns a PolicyResult with potentially adjusted spec. The spec is adjusted
    to comply with policy where possible; if adjustment is impossible, returns
    valid=False.

    Args:
        immutable_hard: The original hard constraints (cannot be relaxed)
        original_target: The original target amount from the first spec
        prev_spec: The previous spec (for per-iteration bounds)
        new_spec: The new spec proposed by the LLM

    Returns:
        PolicyResult with valid flag, adjusted spec, and violation messages

    Policy Rules (from DESIGN.md Section 8.3):
        - program_type: Never change → invalid
        - max_net_leverage: Cannot increase beyond original, cannot delete → clamp/restore
        - min_fixed_charge_coverage: Cannot decrease below original, cannot delete → clamp/restore
        - min_interest_coverage: Cannot decrease below original, cannot delete → clamp/restore
        - max_critical_fraction: Cannot increase beyond original, cannot delete → clamp/restore
        - target_amount: Must decrease or stay same → clamp
        - target_amount: Max 20% drop per iteration → clamp
        - target_amount: Min 50% of original → invalid if below
        - max_criticality: +0.1/iter max, ceiling 0.8, cannot delete → clamp/restore
        - min_leaseability_score: -0.1/iter max, floor 0.2, cannot delete → clamp/restore

    Note:
        When previous filter value was None, a default baseline of 0.5 is used
        for computing bounded relaxation.
    """
    violations: list[str] = []
    adjusted = new_spec.model_copy(deep=True)

    # =========================================================================
    # IMMUTABLE CHECKS (invalid if violated)
    # =========================================================================

    # Program type must not change
    if new_spec.program_type != prev_spec.program_type:
        violations.append(
            f"Cannot change program_type from {prev_spec.program_type.value} "
            f"to {new_spec.program_type.value}"
        )
        return PolicyResult(valid=False, spec=None, violations=violations)

    # =========================================================================
    # HARD CONSTRAINT IMMUTABILITY (clamp if violated)
    # =========================================================================

    # max_net_leverage: cannot increase beyond original or delete if originally set
    if immutable_hard.max_net_leverage is not None:
        if new_spec.hard_constraints.max_net_leverage is None:
            # Cannot delete a constraint that was originally set
            violations.append(
                f"Cannot remove max_net_leverage constraint "
                f"(original: {immutable_hard.max_net_leverage:.2f}x)"
            )
            adjusted.hard_constraints = adjusted.hard_constraints.model_copy(
                update={"max_net_leverage": immutable_hard.max_net_leverage}
            )
        elif new_spec.hard_constraints.max_net_leverage > immutable_hard.max_net_leverage:
            violations.append(
                f"Cannot increase max_net_leverage beyond {immutable_hard.max_net_leverage:.2f}x "
                f"(attempted {new_spec.hard_constraints.max_net_leverage:.2f}x)"
            )
            adjusted.hard_constraints = adjusted.hard_constraints.model_copy(
                update={"max_net_leverage": immutable_hard.max_net_leverage}
            )

    # min_fixed_charge_coverage: cannot decrease below original or delete if originally set
    if immutable_hard.min_fixed_charge_coverage is not None:
        if new_spec.hard_constraints.min_fixed_charge_coverage is None:
            # Cannot delete a constraint that was originally set
            violations.append(
                f"Cannot remove min_fixed_charge_coverage constraint "
                f"(original: {immutable_hard.min_fixed_charge_coverage:.2f}x)"
            )
            adjusted.hard_constraints = adjusted.hard_constraints.model_copy(
                update={"min_fixed_charge_coverage": immutable_hard.min_fixed_charge_coverage}
            )
        elif new_spec.hard_constraints.min_fixed_charge_coverage < immutable_hard.min_fixed_charge_coverage:
            violations.append(
                f"Cannot decrease min_fixed_charge_coverage below {immutable_hard.min_fixed_charge_coverage:.2f}x "
                f"(attempted {new_spec.hard_constraints.min_fixed_charge_coverage:.2f}x)"
            )
            adjusted.hard_constraints = adjusted.hard_constraints.model_copy(
                update={"min_fixed_charge_coverage": immutable_hard.min_fixed_charge_coverage}
            )

    # min_interest_coverage: cannot decrease below original or delete if originally set
    if immutable_hard.min_interest_coverage is not None:
        if new_spec.hard_constraints.min_interest_coverage is None:
            # Cannot delete a constraint that was originally set
            violations.append(
                f"Cannot remove min_interest_coverage constraint "
                f"(original: {immutable_hard.min_interest_coverage:.2f}x)"
            )
            adjusted.hard_constraints = adjusted.hard_constraints.model_copy(
                update={"min_interest_coverage": immutable_hard.min_interest_coverage}
            )
        elif new_spec.hard_constraints.min_interest_coverage < immutable_hard.min_interest_coverage:
            violations.append(
                f"Cannot decrease min_interest_coverage below {immutable_hard.min_interest_coverage:.2f}x "
                f"(attempted {new_spec.hard_constraints.min_interest_coverage:.2f}x)"
            )
            adjusted.hard_constraints = adjusted.hard_constraints.model_copy(
                update={"min_interest_coverage": immutable_hard.min_interest_coverage}
            )

    # max_critical_fraction: cannot increase beyond original or delete if originally set
    if immutable_hard.max_critical_fraction is not None:
        if new_spec.hard_constraints.max_critical_fraction is None:
            # Cannot delete a constraint that was originally set
            violations.append(
                f"Cannot remove max_critical_fraction constraint "
                f"(original: {immutable_hard.max_critical_fraction:.1%})"
            )
            adjusted.hard_constraints = adjusted.hard_constraints.model_copy(
                update={"max_critical_fraction": immutable_hard.max_critical_fraction}
            )
        elif new_spec.hard_constraints.max_critical_fraction > immutable_hard.max_critical_fraction:
            violations.append(
                f"Cannot increase max_critical_fraction beyond {immutable_hard.max_critical_fraction:.1%} "
                f"(attempted {new_spec.hard_constraints.max_critical_fraction:.1%})"
            )
            adjusted.hard_constraints = adjusted.hard_constraints.model_copy(
                update={"max_critical_fraction": immutable_hard.max_critical_fraction}
            )

    # =========================================================================
    # TARGET AMOUNT MONOTONICITY (clamp or invalid)
    # =========================================================================

    # Target amount: must decrease or stay same
    if new_spec.target_amount > prev_spec.target_amount:
        violations.append(
            f"target_amount cannot increase (was ${prev_spec.target_amount:,.0f}, "
            f"attempted ${new_spec.target_amount:,.0f})"
        )
        adjusted.target_amount = prev_spec.target_amount

    # Target amount: bounded per-iteration reduction
    max_drop_fraction = config.max_per_iteration_target_drop_fraction
    min_allowed_target = prev_spec.target_amount * (1.0 - max_drop_fraction)
    if adjusted.target_amount < min_allowed_target:
        violations.append(
            f"target_amount cannot drop more than {max_drop_fraction:.0%} per iteration "
            f"(min ${min_allowed_target:,.0f}, attempted ${new_spec.target_amount:,.0f})"
        )
        adjusted.target_amount = min_allowed_target

    # Target amount: global floor (enforces user intent)
    global_floor = original_target * config.global_target_floor_fraction
    if adjusted.target_amount < global_floor:
        violations.append(
            f"target_amount cannot go below {config.global_target_floor_fraction:.0%} of original "
            f"(${global_floor:,.0f})"
        )
        return PolicyResult(valid=False, spec=None, violations=violations)

    # =========================================================================
    # BOUNDED FILTER RELAXATION (clamp if exceeded)
    # =========================================================================

    # max_criticality: bounded increase per iteration with absolute ceiling
    # Cannot delete if previously set
    prev_crit = prev_spec.asset_filters.max_criticality
    if prev_crit is not None and new_spec.asset_filters.max_criticality is None:
        # Cannot delete a filter that was previously set
        violations.append(
            f"Cannot remove max_criticality filter (previous: {prev_crit:.2f})"
        )
        adjusted.asset_filters = adjusted.asset_filters.model_copy(
            update={"max_criticality": prev_crit}
        )
    elif new_spec.asset_filters.max_criticality is not None:
        if prev_crit is None:
            prev_crit = config.max_criticality_default_baseline

        max_allowed = min(
            prev_crit + config.max_criticality_step,
            config.max_criticality_ceiling,
        )

        if new_spec.asset_filters.max_criticality > max_allowed:
            violations.append(
                f"max_criticality can only increase by {config.max_criticality_step:.2f} per iteration "
                f"(max {max_allowed:.2f}, attempted {new_spec.asset_filters.max_criticality:.2f})"
            )
            adjusted.asset_filters = adjusted.asset_filters.model_copy(
                update={"max_criticality": max_allowed}
            )

        # Absolute ceiling
        if adjusted.asset_filters.max_criticality is not None:
            if adjusted.asset_filters.max_criticality > config.max_criticality_ceiling:
                adjusted.asset_filters = adjusted.asset_filters.model_copy(
                    update={"max_criticality": config.max_criticality_ceiling}
                )

    # min_leaseability_score: bounded decrease per iteration with absolute floor
    # Cannot delete if previously set
    prev_lease = prev_spec.asset_filters.min_leaseability_score
    if prev_lease is not None and new_spec.asset_filters.min_leaseability_score is None:
        # Cannot delete a filter that was previously set
        violations.append(
            f"Cannot remove min_leaseability_score filter (previous: {prev_lease:.2f})"
        )
        adjusted.asset_filters = adjusted.asset_filters.model_copy(
            update={"min_leaseability_score": prev_lease}
        )
    elif new_spec.asset_filters.min_leaseability_score is not None:
        if prev_lease is None:
            prev_lease = config.min_leaseability_default_baseline

        min_allowed = max(
            prev_lease - config.min_leaseability_step,
            config.min_leaseability_floor,
        )

        if new_spec.asset_filters.min_leaseability_score < min_allowed:
            violations.append(
                f"min_leaseability_score can only decrease by {config.min_leaseability_step:.2f} per iteration "
                f"(min {min_allowed:.2f}, attempted {new_spec.asset_filters.min_leaseability_score:.2f})"
            )
            adjusted.asset_filters = adjusted.asset_filters.model_copy(
                update={"min_leaseability_score": min_allowed}
            )

        # Absolute floor
        if adjusted.asset_filters.min_leaseability_score is not None:
            if adjusted.asset_filters.min_leaseability_score < config.min_leaseability_floor:
                adjusted.asset_filters = adjusted.asset_filters.model_copy(
                    update={"min_leaseability_score": config.min_leaseability_floor}
                )

    # =========================================================================
    # RETURN RESULT
    # =========================================================================

    # If we made adjustments, spec is still valid but modified
    if violations:
        return PolicyResult(valid=True, spec=adjusted, violations=violations)

    return PolicyResult(valid=True, spec=new_spec, violations=[])
