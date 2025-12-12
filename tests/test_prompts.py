"""
Tests for LLM prompts and OpenAI client.

Test categories:
- Prompt template formatting
- OpenAI client initialization
- OpenAI client error handling (mocked)
- Prompt content validation
"""

from unittest.mock import MagicMock, patch

import pytest

from app.config import LLMConfig
from app.llm.openai_client import LLMError, OpenAILLMClient
from app.llm.prompts import (
    GENERATE_EXPLANATION_SYSTEM,
    GENERATE_SPEC_SYSTEM,
    GENERATE_SPEC_USER,
    REVISE_SPEC_SYSTEM,
    REVISE_SPEC_USER,
    format_explanation_system,
    format_explanation_user,
    format_generate_spec_system,
    format_generate_spec_user,
    format_metric,
    format_revise_spec_system,
    format_revise_spec_user,
    format_violations,
)
from app.models import (
    AssetFilters,
    ConstraintViolation,
    ExplanationNode,
    HardConstraints,
    Objective,
    ProgramOutcome,
    ProgramType,
    SelectionStatus,
    SelectorSpec,
    SoftPreferences,
)


# =============================================================================
# Test Data
# =============================================================================


@pytest.fixture
def sample_spec() -> SelectorSpec:
    """Create a sample SelectorSpec for testing."""
    return SelectorSpec(
        program_type=ProgramType.SLB,
        objective=Objective.BALANCED,
        target_amount=100_000_000,
        hard_constraints=HardConstraints(
            max_net_leverage=4.0,
            min_fixed_charge_coverage=3.0,
        ),
        soft_preferences=SoftPreferences(),
        asset_filters=AssetFilters(),
        max_iterations=3,
    )


@pytest.fixture
def sample_outcome() -> ProgramOutcome:
    """Create a sample ProgramOutcome for testing."""
    return ProgramOutcome(
        status=SelectionStatus.INFEASIBLE,
        selected_assets=[],
        proceeds=50_000_000,
        leverage_before=4.0,
        leverage_after=3.5,
        interest_coverage_before=5.0,
        interest_coverage_after=4.5,
        fixed_charge_coverage_before=4.0,
        fixed_charge_coverage_after=3.2,
        critical_fraction=0.15,
        violations=[
            ConstraintViolation(
                code="TARGET_NOT_MET",
                detail="Insufficient proceeds",
                actual=50_000_000,
                limit=100_000_000,
            )
        ],
    )


@pytest.fixture
def sample_nodes() -> list[ExplanationNode]:
    """Create sample ExplanationNodes for testing."""
    return [
        ExplanationNode(
            id="node_1",
            label="Fixed charge coverage binding",
            severity="warning",
            category="constraint",
            metric="fixed_charge_coverage",
            baseline_value=4.0,
            post_value=3.2,
            threshold=3.0,
        ),
        ExplanationNode(
            id="node_2",
            label="Low criticality preferred",
            severity="info",
            category="driver",
        ),
    ]


# =============================================================================
# Format Violations Tests
# =============================================================================


class TestFormatViolations:
    """Tests for format_violations helper."""

    def test_empty_violations(self) -> None:
        """Empty violations list returns 'None'."""
        result = format_violations([])
        assert result == "None"

    def test_single_violation(self) -> None:
        """Single violation is formatted correctly."""
        violations = [
            ConstraintViolation(
                code="TARGET_NOT_MET",
                detail="Insufficient proceeds",
                actual=50_000_000,
                limit=100_000_000,
            )
        ]
        result = format_violations(violations)

        assert "TARGET_NOT_MET" in result
        assert "Insufficient proceeds" in result
        assert "50000000.00" in result
        assert "100000000.00" in result

    def test_multiple_violations(self) -> None:
        """Multiple violations are formatted as list."""
        violations = [
            ConstraintViolation(
                code="TARGET_NOT_MET",
                detail="Too low",
                actual=50,
                limit=100,
            ),
            ConstraintViolation(
                code="MAX_NET_LEVERAGE",
                detail="Too high",
                actual=5.0,
                limit=4.0,
            ),
        ]
        result = format_violations(violations)

        assert "TARGET_NOT_MET" in result
        assert "MAX_NET_LEVERAGE" in result
        assert result.count("-") == 2  # Two bullet points


# =============================================================================
# Format Metric Tests
# =============================================================================


class TestFormatMetric:
    """Tests for format_metric helper."""

    def test_none_value(self) -> None:
        """None value returns 'N/A'."""
        assert format_metric(None) == "N/A"

    def test_numeric_value(self) -> None:
        """Numeric value is formatted with suffix."""
        assert format_metric(4.0) == "4.00x"
        assert format_metric(3.14159) == "3.14x"

    def test_custom_suffix(self) -> None:
        """Custom suffix is applied."""
        assert format_metric(50.0, suffix="%") == "50.00%"


# =============================================================================
# Generate Spec Prompt Tests
# =============================================================================


class TestGenerateSpecPrompts:
    """Tests for generate_selector_spec prompt formatting."""

    def test_system_prompt_contains_program_type(self) -> None:
        """System prompt includes program type."""
        result = format_generate_spec_system("slb")
        assert "slb" in result
        assert "commercial real estate" in result.lower()

    def test_system_prompt_contains_defaults(self) -> None:
        """System prompt lists default values."""
        result = format_generate_spec_system("slb")
        assert "max_net_leverage: 4.0" in result
        assert "min_fixed_charge_coverage: 3.0" in result
        assert "max_iterations: 3" in result

    def test_system_prompt_contains_rules(self) -> None:
        """System prompt includes important rules."""
        result = format_generate_spec_system("slb")
        assert "DO NOT ADJUST USER-PROVIDED NUMBERS" in result
        assert "valid JSON" in result
        assert "EXACT EXTRACTION" in result

    def test_user_prompt_contains_all_inputs(self) -> None:
        """User prompt includes all input parameters."""
        result = format_generate_spec_user(
            program_type="slb",
            program_description="Raise $100M via SLB",
            asset_summary="10 assets, $500M total value",
        )

        assert "slb" in result
        assert "Raise $100M via SLB" in result
        assert "10 assets" in result
        assert "$500M total value" in result

    def test_user_prompt_is_not_empty(self) -> None:
        """User prompt is generated even with minimal inputs."""
        result = format_generate_spec_user(
            program_type="slb",
            program_description="Test",
            asset_summary="",
        )
        assert len(result) > 0
        assert "Test" in result


# =============================================================================
# Revise Spec Prompt Tests
# =============================================================================


class TestReviseSpecPrompts:
    """Tests for revise_selector_spec prompt formatting."""

    def test_system_prompt_contains_rules(self, sample_outcome: ProgramOutcome) -> None:
        """System prompt includes revision rules."""
        result = format_revise_spec_system(sample_outcome)

        # Check for revision priority rules (case insensitive match)
        assert "Reduce target_amount" in result or "reduce target" in result.lower()
        assert "Relax" in result  # "Relax SOFT filters"
        # Check what cannot be changed
        assert "CANNOT CHANGE" in result or "cannot change" in result.lower()
        assert "Program type" in result or "program_type" in result.lower()

    def test_system_prompt_contains_violations(
        self, sample_outcome: ProgramOutcome
    ) -> None:
        """System prompt includes violation details."""
        result = format_revise_spec_system(sample_outcome)
        assert "TARGET_NOT_MET" in result

    def test_system_prompt_contains_strategy(
        self, sample_outcome: ProgramOutcome
    ) -> None:
        """System prompt includes revision strategy."""
        result = format_revise_spec_system(sample_outcome)
        assert "TARGET_NOT_MET" in result
        # Check for revision strategy (case insensitive - prompt uses "reduce target" or "relax")
        assert "reduce target" in result.lower() or "relax" in result.lower()

    def test_user_prompt_contains_spec_json(
        self, sample_spec: SelectorSpec, sample_outcome: ProgramOutcome
    ) -> None:
        """User prompt includes previous spec as JSON."""
        result = format_revise_spec_user(
            original_description="Raise capital",
            previous_spec=sample_spec,
            outcome=sample_outcome,
        )

        assert '"program_type"' in result
        assert '"target_amount"' in result
        assert "100000000" in result

    def test_user_prompt_contains_outcome_metrics(
        self, sample_spec: SelectorSpec, sample_outcome: ProgramOutcome
    ) -> None:
        """User prompt includes outcome metrics."""
        result = format_revise_spec_user(
            original_description="Raise capital",
            previous_spec=sample_spec,
            outcome=sample_outcome,
        )

        assert "Status: infeasible" in result
        assert "$50,000,000" in result
        assert "3.50x" in result  # leverage_after


# =============================================================================
# Explanation Prompt Tests
# =============================================================================


class TestExplanationPrompts:
    """Tests for generate_explanation_summary prompt formatting."""

    def test_system_prompt_is_constant(self) -> None:
        """System prompt is the constant template."""
        result = format_explanation_system()
        assert result == GENERATE_EXPLANATION_SYSTEM

    def test_system_prompt_contains_rules(self) -> None:
        """System prompt includes formatting rules."""
        result = format_explanation_system()
        assert "2-3 sentences" in result
        assert "binding constraints" in result

    def test_user_prompt_contains_nodes(
        self, sample_nodes: list[ExplanationNode]
    ) -> None:
        """User prompt includes node details."""
        result = format_explanation_user(sample_nodes)

        assert "node_1" in result
        assert "Fixed charge coverage binding" in result
        assert "constraint" in result
        assert "driver" in result

    def test_user_prompt_empty_nodes(self) -> None:
        """User prompt handles empty nodes list."""
        result = format_explanation_user([])
        assert "[]" in result or "Explanation Nodes:" in result


# =============================================================================
# OpenAI Client Initialization Tests
# =============================================================================


class TestOpenAIClientInit:
    """Tests for OpenAILLMClient initialization."""

    def test_requires_api_key(self) -> None:
        """Client requires API key."""
        # Clear environment variable if set
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                OpenAILLMClient()

            assert "API key" in str(exc_info.value)

    def test_accepts_api_key_parameter(self) -> None:
        """Client accepts API key as parameter."""
        with patch("app.llm.openai_client.OpenAI") as mock_openai:
            client = OpenAILLMClient(api_key="test-key")
            mock_openai.assert_called_once_with(api_key="test-key")

    def test_uses_environment_variable(self) -> None:
        """Client uses OPENAI_API_KEY environment variable."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            with patch("app.llm.openai_client.OpenAI") as mock_openai:
                client = OpenAILLMClient()
                mock_openai.assert_called_once_with(api_key="env-key")

    def test_uses_custom_config(self) -> None:
        """Client uses custom LLMConfig."""
        custom_config = LLMConfig(model="gpt-4o", temperature=0.5)

        with patch("app.llm.openai_client.OpenAI"):
            client = OpenAILLMClient(api_key="test-key", config=custom_config)

            assert client.config.model == "gpt-4o"
            assert client.config.temperature == 0.5


# =============================================================================
# OpenAI Client Error Handling Tests
# =============================================================================


class TestOpenAIClientErrors:
    """Tests for OpenAI client error handling."""

    def test_api_error_wrapped_in_llm_error(self) -> None:
        """API errors are wrapped in LLMError."""
        with patch("app.llm.openai_client.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.beta.chat.completions.parse.side_effect = Exception("API Error")

            client = OpenAILLMClient(api_key="test-key")

            with pytest.raises(LLMError) as exc_info:
                client.generate_selector_spec(
                    ProgramType.SLB,
                    "Test description",
                    "Test summary",
                )

            assert "API Error" in str(exc_info.value)

    def test_empty_response_raises_error(self) -> None:
        """Empty response raises LLMError."""
        with patch("app.llm.openai_client.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            # Mock response with None parsed content
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.parsed = None
            mock_client.beta.chat.completions.parse.return_value = mock_response

            client = OpenAILLMClient(api_key="test-key")

            with pytest.raises(LLMError) as exc_info:
                client.generate_selector_spec(
                    ProgramType.SLB,
                    "Test description",
                    "Test summary",
                )

            assert "empty response" in str(exc_info.value).lower()


# =============================================================================
# OpenAI Client Method Tests (Mocked)
# =============================================================================


class TestOpenAIClientMethods:
    """Tests for OpenAI client methods with mocked API."""

    def test_generate_selector_spec_calls_api(self) -> None:
        """generate_selector_spec calls the API with correct parameters."""
        with patch("app.llm.openai_client.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            # Mock successful response
            mock_spec = SelectorSpec(
                program_type=ProgramType.SLB,
                objective=Objective.BALANCED,
                target_amount=100_000_000,
                hard_constraints=HardConstraints(),
                soft_preferences=SoftPreferences(),
                asset_filters=AssetFilters(),
                max_iterations=3,
            )
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.parsed = mock_spec
            mock_client.beta.chat.completions.parse.return_value = mock_response

            client = OpenAILLMClient(api_key="test-key")
            result = client.generate_selector_spec(
                ProgramType.SLB,
                "Raise $100M",
                "10 assets",
            )

            # Verify API was called
            mock_client.beta.chat.completions.parse.assert_called_once()

            # Verify result
            assert result.program_type == ProgramType.SLB
            assert result.target_amount == 100_000_000

    def test_revise_selector_spec_calls_api(
        self, sample_spec: SelectorSpec, sample_outcome: ProgramOutcome
    ) -> None:
        """revise_selector_spec calls the API correctly."""
        with patch("app.llm.openai_client.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            # Mock successful response with reduced target
            mock_revised = sample_spec.model_copy(
                update={"target_amount": 80_000_000}
            )
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.parsed = mock_revised
            mock_client.beta.chat.completions.parse.return_value = mock_response

            client = OpenAILLMClient(api_key="test-key")
            result = client.revise_selector_spec(
                "Raise capital",
                sample_spec,
                sample_outcome,
            )

            mock_client.beta.chat.completions.parse.assert_called_once()
            assert result.target_amount == 80_000_000

    def test_generate_explanation_summary_calls_api(
        self, sample_nodes: list[ExplanationNode]
    ) -> None:
        """generate_explanation_summary calls text completion API."""
        with patch("app.llm.openai_client.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            # Mock successful text response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test summary"
            mock_client.chat.completions.create.return_value = mock_response

            client = OpenAILLMClient(api_key="test-key")
            result = client.generate_explanation_summary(sample_nodes)

            # Verify text API was called (not structured)
            mock_client.chat.completions.create.assert_called_once()
            assert result == "Test summary"


# =============================================================================
# Prompt Template Validation Tests
# =============================================================================


class TestPromptTemplateValidation:
    """Tests that prompt templates are well-formed."""

    def test_generate_spec_system_is_valid_template(self) -> None:
        """GENERATE_SPEC_SYSTEM has required placeholders."""
        assert "{program_type}" in GENERATE_SPEC_SYSTEM

    def test_generate_spec_user_is_valid_template(self) -> None:
        """GENERATE_SPEC_USER has required placeholders."""
        assert "{program_type}" in GENERATE_SPEC_USER
        assert "{program_description}" in GENERATE_SPEC_USER
        assert "{asset_summary}" in GENERATE_SPEC_USER

    def test_revise_spec_system_is_valid_template(self) -> None:
        """REVISE_SPEC_SYSTEM has required placeholders."""
        assert "{violations}" in REVISE_SPEC_SYSTEM

    def test_revise_spec_user_is_valid_template(self) -> None:
        """REVISE_SPEC_USER has required placeholders."""
        assert "{original_description}" in REVISE_SPEC_USER
        assert "{previous_spec_json}" in REVISE_SPEC_USER
        assert "{status}" in REVISE_SPEC_USER
        assert "{proceeds" in REVISE_SPEC_USER
