"""
Mock LLM Client for testing.

This module provides a deterministic MockLLMClient for testing the orchestrator
and other components without making real LLM calls.

The mock is designed to be:
- Deterministic (same inputs → same outputs)
- Controllable (can configure behavior for different test scenarios)
- Valid (always returns properly structured data)
"""

from typing import Optional

from app.models import (
    AssetFilters,
    ExplanationNode,
    HardConstraints,
    Objective,
    ProgramOutcome,
    ProgramType,
    SelectionStatus,
    SelectorSpec,
    SoftPreferences,
)


class MockLLMClient:
    """
    Mock LLM client for testing.

    Provides deterministic responses that can be configured for different
    test scenarios. All methods return valid structured data.

    Attributes:
        default_target_amount: Default target for generated specs
        revision_target_reduction: Percentage to reduce target on revision
        custom_spec: Optional custom spec to return (overrides generation)
        custom_summary: Optional custom summary to return
        call_counts: Track method call counts for assertions

    Example:
        ```python
        mock = MockLLMClient(default_target_amount=100_000_000)
        spec = mock.generate_selector_spec(
            ProgramType.SLB,
            "Raise $100M via SLB",
            "10 assets, $500M total value"
        )
        assert spec.target_amount == 100_000_000
        ```
    """

    def __init__(
        self,
        default_target_amount: float = 100_000_000,
        revision_target_reduction: float = 0.15,  # 15% reduction per revision
        custom_spec: Optional[SelectorSpec] = None,
        custom_summary: Optional[str] = None,
    ):
        """
        Initialize MockLLMClient.

        Args:
            default_target_amount: Default target for generated specs
            revision_target_reduction: Fraction to reduce target on revision (0.0-1.0)
            custom_spec: If provided, generate_selector_spec returns this
            custom_summary: If provided, generate_explanation_summary returns this
        """
        self.default_target_amount = default_target_amount
        self.revision_target_reduction = revision_target_reduction
        self.custom_spec = custom_spec
        self.custom_summary = custom_summary

        # Track calls for testing
        self.call_counts = {
            "generate_selector_spec": 0,
            "revise_selector_spec": 0,
            "generate_explanation_summary": 0,
        }

        # Store call history for inspection
        self.call_history: list[dict] = []

    def generate_selector_spec(
        self,
        program_type: ProgramType,
        program_description: str,
        asset_summary: str,
    ) -> SelectorSpec:
        """
        Generate initial SelectorSpec from program brief.

        Returns a deterministic spec based on configured defaults.
        If custom_spec is set, returns that instead.

        Args:
            program_type: The type of funding program
            program_description: Natural language description
            asset_summary: Summary of available assets

        Returns:
            SelectorSpec with conservative defaults
        """
        self.call_counts["generate_selector_spec"] += 1
        self.call_history.append({
            "method": "generate_selector_spec",
            "program_type": program_type,
            "program_description": program_description,
            "asset_summary": asset_summary,
        })

        if self.custom_spec is not None:
            return self.custom_spec

        # Parse target from description if mentioned
        target = self._extract_target_from_description(program_description)
        if target is None:
            target = self.default_target_amount

        # Determine objective from description
        objective = self._determine_objective(program_description)

        return SelectorSpec(
            program_type=program_type,
            objective=objective,
            target_amount=target,
            hard_constraints=HardConstraints(
                max_net_leverage=4.0,
                min_fixed_charge_coverage=3.0,
                min_interest_coverage=None,
                max_critical_fraction=None,
            ),
            soft_preferences=SoftPreferences(
                prefer_low_criticality=True,
                prefer_high_leaseability=True,
                weight_criticality=1.0,
                weight_leaseability=1.0,
            ),
            asset_filters=AssetFilters(),
            max_iterations=3,
        )

    def revise_selector_spec(
        self,
        original_description: str,
        previous_spec: SelectorSpec,
        outcome: ProgramOutcome,
    ) -> SelectorSpec:
        """
        Revise SelectorSpec to address infeasibility.

        Reduces target by configured percentage and relaxes filters.
        Does NOT relax hard constraints (immutable per policy).

        Args:
            original_description: The original program description
            previous_spec: The spec that produced infeasible outcome
            outcome: The ProgramOutcome with violations

        Returns:
            Revised SelectorSpec with reduced target
        """
        self.call_counts["revise_selector_spec"] += 1
        self.call_history.append({
            "method": "revise_selector_spec",
            "original_description": original_description,
            "previous_target": previous_spec.target_amount,
            "violations": [v.code for v in outcome.violations],
        })

        # Reduce target by configured percentage
        new_target = previous_spec.target_amount * (1 - self.revision_target_reduction)

        # Relax filters slightly
        new_filters = AssetFilters(
            include_asset_types=previous_spec.asset_filters.include_asset_types,
            exclude_asset_types=previous_spec.asset_filters.exclude_asset_types,
            exclude_markets=previous_spec.asset_filters.exclude_markets,
            # Relax score thresholds
            min_leaseability_score=self._relax_min_threshold(
                previous_spec.asset_filters.min_leaseability_score
            ),
            max_criticality=self._relax_max_threshold(
                previous_spec.asset_filters.max_criticality
            ),
        )

        return SelectorSpec(
            program_type=previous_spec.program_type,
            objective=previous_spec.objective,
            target_amount=new_target,
            # Hard constraints are immutable - copy as-is
            hard_constraints=previous_spec.hard_constraints,
            soft_preferences=previous_spec.soft_preferences,
            asset_filters=new_filters,
            max_iterations=previous_spec.max_iterations,
        )

    def generate_explanation_summary(
        self,
        nodes: list[ExplanationNode],
    ) -> str:
        """
        Generate narrative summary from explanation nodes.

        Produces a deterministic summary based on node categories.

        Args:
            nodes: List of ExplanationNode objects

        Returns:
            2-3 sentence narrative summary
        """
        self.call_counts["generate_explanation_summary"] += 1
        self.call_history.append({
            "method": "generate_explanation_summary",
            "node_count": len(nodes),
            "categories": [n.category for n in nodes],
        })

        if self.custom_summary is not None:
            return self.custom_summary

        # Build summary from nodes
        constraint_nodes = [n for n in nodes if n.category == "constraint"]
        driver_nodes = [n for n in nodes if n.category == "driver"]
        risk_nodes = [n for n in nodes if n.category == "risk"]

        error_nodes = [n for n in constraint_nodes if n.severity == "error"]

        if error_nodes:
            # Infeasible case
            violation_labels = [n.label for n in error_nodes[:2]]
            return (
                f"The selection was unable to meet all requirements. "
                f"Key issues: {', '.join(violation_labels)}. "
                f"Consider reducing the target or relaxing filters."
            )

        # Feasible case
        parts = ["Successfully structured the SLB program."]

        if driver_nodes:
            driver_label = driver_nodes[0].label
            parts.append(f"Key driver: {driver_label}.")

        if risk_nodes:
            risk_label = risk_nodes[0].label
            parts.append(f"Note: {risk_label} should be monitored.")

        return " ".join(parts)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _extract_target_from_description(self, description: str) -> Optional[float]:
        """Extract target amount from description if mentioned."""
        import re

        # Look for patterns like "$100M", "$100 million", "100M"
        patterns = [
            r'\$(\d+(?:\.\d+)?)\s*[Mm](?:illion)?',  # $100M, $100 million
            r'\$(\d+(?:\.\d+)?)\s*[Bb](?:illion)?',  # $1B, $1 billion
            r'(\d+(?:\.\d+)?)\s*[Mm](?:illion)?\s*(?:dollars?)?',  # 100M, 100 million
        ]

        for pattern in patterns:
            match = re.search(pattern, description)
            if match:
                value = float(match.group(1))
                if 'B' in pattern or 'b' in description[match.start():match.end()]:
                    return value * 1_000_000_000
                return value * 1_000_000

        return None

    def _determine_objective(self, description: str) -> Objective:
        """Determine objective from description keywords."""
        description_lower = description.lower()

        if any(kw in description_lower for kw in ["maximize", "max proceeds", "as much as"]):
            return Objective.MAXIMIZE_PROCEEDS
        if any(kw in description_lower for kw in ["minimize risk", "low risk", "conservative"]):
            return Objective.MINIMIZE_RISK
        return Objective.BALANCED

    def _relax_min_threshold(self, value: Optional[float]) -> Optional[float]:
        """Relax a minimum threshold (decrease it)."""
        if value is None:
            return None
        # Decrease by 0.1, but not below 0
        return max(0.0, value - 0.1)

    def _relax_max_threshold(self, value: Optional[float]) -> Optional[float]:
        """Relax a maximum threshold (increase it)."""
        if value is None:
            return None
        # Increase by 0.1, but not above 1
        return min(1.0, value + 0.1)

    # =========================================================================
    # Test Helpers
    # =========================================================================

    def reset_counts(self) -> None:
        """Reset call counts and history."""
        for key in self.call_counts:
            self.call_counts[key] = 0
        self.call_history.clear()

    def get_total_calls(self) -> int:
        """Get total number of method calls."""
        return sum(self.call_counts.values())
