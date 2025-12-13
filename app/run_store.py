"""
In-memory run store for tracking program runs.

This module provides a thread-safe in-memory store for run records.
Runs are stored with UUID-based IDs and can be queried by ID or fund_id.

Note: This is an in-memory implementation for MVP. Data is lost on restart.
For production, replace with a persistent store (database, etc.).
"""

import threading
from dataclasses import dataclass
from typing import Optional

from app.models import ProgramResponse, ScenarioKind


@dataclass
class RunRecord:
    """
    Record of a single program run.

    Attributes:
        run_id: Unique identifier (UUID4 format)
        fund_id: Optional fund identifier for grouping runs
        program_description: Original program description
        response: ProgramResponse if run completed successfully
        error: Error message if run failed (validation error, etc.)
        created_at: ISO format timestamp of run creation

        # Scenario metadata (None = single-scenario run, not part of any set)
        scenario_set_id: ID of the scenario set this run belongs to
        scenario_kind: Classification of this scenario variant
        scenario_label: Human-readable label for this scenario
    """

    # Required fields (no defaults)
    run_id: str
    fund_id: Optional[str]
    program_description: str
    response: Optional[ProgramResponse]
    error: Optional[str]
    created_at: str

    # Optional scenario metadata (defaults = None for backward compatibility)
    scenario_set_id: Optional[str] = None
    scenario_kind: Optional[ScenarioKind] = None
    scenario_label: Optional[str] = None


class RunStore:
    """
    Thread-safe in-memory store for run records.

    Uses a dict for O(1) lookup by run_id and a lock for thread safety.
    Runs are stored in insertion order (Python 3.7+ dict behavior).
    """

    def __init__(self) -> None:
        """Initialize empty store with lock."""
        self._runs: dict[str, RunRecord] = {}
        self._lock = threading.Lock()

    def create(self, record: RunRecord) -> None:
        """
        Store a new run record.

        Args:
            record: RunRecord to store

        Note: Overwrites if run_id already exists (should not happen with UUIDs)
        """
        with self._lock:
            self._runs[record.run_id] = record

    def get(self, run_id: str) -> Optional[RunRecord]:
        """
        Retrieve a run record by ID.

        Args:
            run_id: The run's unique identifier

        Returns:
            RunRecord if found, None otherwise
        """
        with self._lock:
            return self._runs.get(run_id)

    def list_runs(
        self,
        fund_id: Optional[str] = None,
        limit: int = 10,
    ) -> list[RunRecord]:
        """
        List runs, optionally filtered by fund_id.

        Args:
            fund_id: If provided, filter to runs with this fund_id
            limit: Maximum number of runs to return (default 10)

        Returns:
            List of RunRecord objects, most recent first
        """
        with self._lock:
            runs = list(self._runs.values())

        # Filter by fund_id if provided
        if fund_id is not None:
            runs = [r for r in runs if r.fund_id == fund_id]

        # Return most recent first, limited
        runs.reverse()
        return runs[:limit]

    def clear(self) -> None:
        """Clear all run records. Useful for testing."""
        with self._lock:
            self._runs.clear()


# Singleton instance for use across the application
run_store = RunStore()
