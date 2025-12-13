"""
OpenAI LLM Client for SLB Agent.

This module provides the real LLM implementation using OpenAI's API.
It implements the LLMClient Protocol and uses structured outputs.

Requires OPENAI_API_KEY environment variable to be set.
"""

import logging
import os
from typing import Optional, TypeVar

logger = logging.getLogger(__name__)

import openai
from openai import OpenAI
from pydantic import BaseModel

from app.config import DEFAULT_LLM_CONFIG, LLMConfig
from app.exceptions import InfrastructureError
from app.llm.prompts import (
    format_explanation_system,
    format_explanation_user,
    format_generate_scenarios_system,
    format_generate_scenarios_user,
    format_generate_spec_system,
    format_generate_spec_user,
    format_revise_spec_system,
    format_revise_spec_user,
)
from app.models import (
    ExplanationNode,
    ProgramOutcome,
    ProgramType,
    ScenarioDefinition,
    ScenarioDefinitionList,
    SelectorSpec,
)


# Type variable for generic structured output
T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Raised when LLM call fails."""

    pass


class OpenAILLMClient:
    """
    OpenAI LLM client implementing the LLMClient Protocol.

    Uses OpenAI's structured output feature to ensure valid responses.
    All methods are synchronous to match the interface contract.

    Attributes:
        config: LLM configuration (model, temperature, max_tokens)
        client: OpenAI client instance

    Example:
        ```python
        client = OpenAILLMClient()
        spec = client.generate_selector_spec(
            ProgramType.SLB,
            "Raise $100M via SLB",
            "10 assets, $500M total value"
        )
        ```
    """

    def __init__(
        self,
        config: LLMConfig = DEFAULT_LLM_CONFIG,
        api_key: Optional[str] = None,
    ):
        """
        Initialize OpenAI client.

        Args:
            config: LLM configuration (defaults to DEFAULT_LLM_CONFIG)
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)

        Raises:
            ValueError: If no API key is provided or found in environment
        """
        self.config = config

        # Get API key from parameter or environment
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = OpenAI(api_key=key)

    def _call_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
    ) -> T:
        """
        Make a structured LLM call.

        Uses OpenAI's structured output feature to parse response
        directly into a Pydantic model.

        Args:
            system_prompt: System message content
            user_prompt: User message content
            response_model: Pydantic model class for response

        Returns:
            Parsed response as the specified model type

        Raises:
            InfrastructureError: If API is unreachable or returns 5xx
            LLMError: If response cannot be parsed or other API errors
        """
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=response_model,
                temperature=self.config.temperature,
                max_completion_tokens=self.config.max_tokens,
            )

            parsed = response.choices[0].message.parsed
            if parsed is None:
                raise LLMError("LLM returned empty response")

            return parsed

        except openai.APIConnectionError as e:
            # Network-level failure - infrastructure error
            raise InfrastructureError(f"Cannot reach OpenAI API: {e}") from e
        except openai.APIStatusError as e:
            # API returned an error status
            if e.status_code >= 500:
                # Server error - infrastructure error
                raise InfrastructureError(f"OpenAI API error ({e.status_code}): {e}") from e
            # 4xx errors are likely our fault (bad request, etc.)
            raise LLMError(f"OpenAI API call failed ({e.status_code}): {e}") from e
        except LLMError:
            raise
        except Exception as e:
            # Wrap unexpected errors in LLMError
            raise LLMError(f"OpenAI API call failed: {e}") from e

    def _call_text(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        Make a text-only LLM call (no structured output).

        Used for generating free-form text like summaries.

        Args:
            system_prompt: System message content
            user_prompt: User message content

        Returns:
            Text response content

        Raises:
            InfrastructureError: If API is unreachable or returns 5xx
            LLMError: If API call fails
        """
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.temperature,
                max_completion_tokens=self.config.max_tokens,
            )

            content = response.choices[0].message.content
            if content is None:
                raise LLMError("LLM returned empty response")

            return content

        except openai.APIConnectionError as e:
            # Network-level failure - infrastructure error
            raise InfrastructureError(f"Cannot reach OpenAI API: {e}") from e
        except openai.APIStatusError as e:
            # API returned an error status
            if e.status_code >= 500:
                # Server error - infrastructure error
                raise InfrastructureError(f"OpenAI API error ({e.status_code}): {e}") from e
            # 4xx errors are likely our fault (bad request, etc.)
            raise LLMError(f"OpenAI API call failed ({e.status_code}): {e}") from e
        except LLMError:
            raise
        except Exception as e:
            # Wrap unexpected errors in LLMError
            raise LLMError(f"OpenAI API call failed: {e}") from e

    def generate_selector_spec(
        self,
        program_type: ProgramType,
        program_description: str,
        asset_summary: str,
    ) -> SelectorSpec:
        """
        Generate initial SelectorSpec from program brief.

        Uses structured output to ensure valid SelectorSpec response.

        Args:
            program_type: The type of funding program
            program_description: Natural language description
            asset_summary: Summary of available assets

        Returns:
            SelectorSpec with target, constraints, and preferences

        Raises:
            LLMError: If API call fails
        """
        system_prompt = format_generate_spec_system(program_type.value)
        user_prompt = format_generate_spec_user(
            program_type=program_type.value,
            program_description=program_description,
            asset_summary=asset_summary,
        )

        spec = self._call_structured(system_prompt, user_prompt, SelectorSpec)

        # Debug logging for asset filters
        debug_msg = (
            f"LLM generated spec: target={spec.target_amount:,.0f}, "
            f"asset_filters.include_types={spec.asset_filters.include_asset_types}, "
            f"asset_filters.exclude_types={spec.asset_filters.exclude_asset_types}"
        )
        logger.info(debug_msg)
        print(f"[DEBUG] {debug_msg}")  # Also print for immediate visibility

        return spec

    def revise_selector_spec(
        self,
        original_description: str,
        previous_spec: SelectorSpec,
        outcome: ProgramOutcome,
    ) -> SelectorSpec:
        """
        Revise SelectorSpec to address infeasibility.

        Uses structured output to ensure valid SelectorSpec response.

        Args:
            original_description: The original program description
            previous_spec: The spec that produced infeasible outcome
            outcome: The ProgramOutcome with violations

        Returns:
            Revised SelectorSpec

        Raises:
            LLMError: If API call fails
        """
        system_prompt = format_revise_spec_system(outcome)
        user_prompt = format_revise_spec_user(
            original_description=original_description,
            previous_spec=previous_spec,
            outcome=outcome,
        )

        return self._call_structured(system_prompt, user_prompt, SelectorSpec)

    def generate_explanation_summary(
        self,
        nodes: list[ExplanationNode],
    ) -> str:
        """
        Generate narrative summary from explanation nodes.

        Uses text completion (not structured output) since
        the result is free-form text.

        Args:
            nodes: List of ExplanationNode objects

        Returns:
            2-3 sentence narrative summary

        Raises:
            LLMError: If API call fails
        """
        system_prompt = format_explanation_system()
        user_prompt = format_explanation_user(nodes)

        return self._call_text(system_prompt, user_prompt)

    def generate_scenario_definitions(
        self,
        brief: str,
        asset_summary: str,
        num_scenarios: int,
    ) -> list[ScenarioDefinition]:
        """
        Generate scenario definitions from a brief.

        Creates multiple scenario variants (base, conservative, aggressive, etc.)
        from a single natural language brief. Uses structured output to ensure
        valid ScenarioDefinition responses.

        Args:
            brief: Natural language program description
            asset_summary: Summary of available assets
            num_scenarios: Number of scenarios to generate (1-5)

        Returns:
            List of ScenarioDefinition objects

        Contract:
            - First scenario MUST be kind=BASE
            - Each scenario has unique label
            - target_amount is required for each scenario
            - max_leverage and min_coverage are optional

        Raises:
            LLMError: If API call fails
        """
        # Clamp num_scenarios to valid range
        num_scenarios = max(1, min(num_scenarios, 5))

        system_prompt = format_generate_scenarios_system(num_scenarios)
        user_prompt = format_generate_scenarios_user(
            brief=brief,
            asset_summary=asset_summary,
            num_scenarios=num_scenarios,
        )

        result = self._call_structured(system_prompt, user_prompt, ScenarioDefinitionList)

        logger.info(
            f"LLM generated {len(result.scenarios)} scenarios: "
            f"{[s.label for s in result.scenarios]}"
        )

        return result.scenarios
