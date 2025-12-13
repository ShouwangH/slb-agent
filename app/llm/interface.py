"""
LLM Client Interface for SLB Agent.

This module defines the abstract interface for LLM clients.
All LLM implementations (mock, OpenAI, etc.) must implement this Protocol.

The interface is designed to be:
- Technology-agnostic (no OpenAI-specific code)
- Async-compatible for production use
- Synchronous for testing simplicity
"""

from typing import Protocol, runtime_checkable

from app.models import (
    ExplanationNode,
    ProgramOutcome,
    ProgramType,
    ScenarioDefinition,
    SelectorSpec,
)


@runtime_checkable
class LLMClient(Protocol):
    """
    Protocol defining the LLM client interface.

    All LLM implementations must provide these methods. The interface
    supports both sync and async implementations - implementers can
    choose based on their needs.

    Methods:
        generate_selector_spec: Generate initial SelectorSpec from program brief
        revise_selector_spec: Revise spec to address infeasibility
        generate_explanation_summary: Generate narrative summary from nodes
    """

    def generate_selector_spec(
        self,
        program_type: ProgramType,
        program_description: str,
        asset_summary: str,
    ) -> SelectorSpec:
        """
        Generate initial SelectorSpec from natural language program brief.

        This is the first LLM call in the agentic loop. The LLM translates
        the user's intent into a structured specification.

        Args:
            program_type: The type of funding program (e.g., SLB)
            program_description: Natural language description of the program
            asset_summary: Summary statistics of available assets

        Returns:
            SelectorSpec with target amount, constraints, preferences, and filters

        Note:
            The spec should use conservative defaults where the description
            is ambiguous. Hard constraints should reflect typical covenant
            requirements unless explicitly stated otherwise.
        """
        ...

    def revise_selector_spec(
        self,
        original_description: str,
        previous_spec: SelectorSpec,
        outcome: ProgramOutcome,
    ) -> SelectorSpec:
        """
        Revise SelectorSpec to address infeasibility.

        Called when the engine returns INFEASIBLE status. The LLM should
        analyze the violations and adjust the spec within policy bounds.

        Args:
            original_description: The original program description
            previous_spec: The spec that produced the infeasible outcome
            outcome: The ProgramOutcome with violations and metrics

        Returns:
            Revised SelectorSpec attempting to achieve feasibility

        Note:
            The LLM should respect revision policy bounds:
            - Cannot relax hard constraints (immutable)
            - Target can only decrease (max 20% per iteration)
            - Filters can relax within bounds
        """
        ...

    def generate_explanation_summary(
        self,
        nodes: list[ExplanationNode],
    ) -> str:
        """
        Generate narrative summary from structured explanation nodes.

        The engine produces machine-readable ExplanationNode objects.
        This method generates a 2-3 sentence executive summary suitable
        for investment committee presentations.

        Args:
            nodes: List of ExplanationNode objects from the engine

        Returns:
            2-3 sentence narrative summary

        Note:
            The summary should:
            - Lead with the key outcome (feasible/infeasible)
            - Highlight binding constraints or key drivers
            - Note any significant risks
        """
        ...

    def generate_scenario_definitions(
        self,
        brief: str,
        asset_summary: str,
        num_scenarios: int,
    ) -> list[ScenarioDefinition]:
        """
        Generate scenario definitions from a brief.

        Creates multiple scenario variants (base, conservative, aggressive, etc.)
        from a single natural language brief. Each scenario represents a different
        capital ask with potentially different constraints.

        Args:
            brief: Natural language program description
            asset_summary: Summary of available assets (from summarize_assets)
            num_scenarios: Number of scenarios to generate (1-5)

        Returns:
            List of ScenarioDefinition objects

        Contract:
            - First scenario MUST be kind=BASE
            - Each scenario has unique label
            - target_amount is required for each scenario
            - max_leverage and min_coverage are optional
        """
        ...
