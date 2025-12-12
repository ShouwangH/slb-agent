"""
Custom exceptions for the SLB Agent.

This module defines exceptions that are distinct from validation errors
and represent infrastructure or external service failures.
"""


class InfrastructureError(Exception):
    """
    Raised when external services (LLM, etc.) are unreachable or fail unexpectedly.

    This exception type signals that the failure is due to infrastructure issues,
    not invalid input or business logic violations. API endpoints should return
    503 Service Unavailable when catching this exception.

    Examples:
        - LLM API connection timeout
        - LLM API returns 5xx error
        - Response parsing failure due to malformed LLM output
    """

    pass
