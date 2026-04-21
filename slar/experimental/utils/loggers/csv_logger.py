"""
csv_logger.py

A lightweight CSV-based metrics logger that serves as a drop-in fallback
for Weights & Biases (wandb). Supports `init()` as a context manager
and `log(metrics, step)` for logging.

Usage:
    import csv_logger

    with csv_logger.init(project="my_project", name="run_01", dir="logs"):
        for step in range(100):
            csv_logger.log({"loss": 0.5, "acc": 0.9}, step=step)

    # Or without context manager:
    run = csv_logger.init(project="my_project", name="run_01")
    csv_logger.log({"loss": 0.5}, step=0)
    run.finish()
"""

from __future__ import annotations

import csv
import io
import json
import os
import threading
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any


class Run:
    """Represents a single logging run, analogous to a wandb.Run."""

    def __init__(
        self,
        project: str = "default_project",
        name: str | None = None,
        dir: str | Path = ".",
        config: dict[str, Any] | None = None,
        flush_every: int = 1,
        **kwargs: Any,
    ) -> None:
        self.project = project
        self.name = name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.config = config or {}
        self.flush_every = flush_every
        self._extra_kwargs = kwargs  # absorb unknown wandb kwargs gracefully

        # Build output directory: <dir>/<project>/<name>/
        self._run_dir = Path(dir) / self.project / self.name
        self._run_dir.mkdir(parents=True, exist_ok=True)

        self._csv_path = self._run_dir / "metrics.csv"
        self._config_path = self._run_dir / "config.json"

        # Internal state
        self._fieldnames: list[str] = ["_step", "_timestamp"]
        self._rows_buffer: list[dict[str, Any]] = []
        self._rows_written: int = 0
        self._file: io.TextIOWrapper | None = None
        self._writer: csv.DictWriter | None = None
        self._lock = threading.Lock()
        self._finished = False

        # Persist config
        self._save_config()

        # Open CSV file
        self._open()

        print(
            f"[csv_logger] Run initialized.\n"
            f"  Project : {self.project}\n"
            f"  Name    : {self.name}\n"
            f"  Dir     : {self._run_dir}\n"
            f"  CSV     : {self._csv_path}"
        )

    # ------------------------------------------------------------------
    # File handling
    # ------------------------------------------------------------------

    def _save_config(self) -> None:
        with open(self._config_path, "w") as f:
            json.dump(self.config, f, indent=2, default=str)

    def _open(self) -> None:
        self._file = open(self._csv_path, "w", newline="")
        # We'll defer writing the header until the first log call,
        # because we don't know the metric keys yet.
        self._writer = None

    def _ensure_writer(self, keys: list[str]) -> None:
        """Lazily create (or recreate) the CSV writer when new columns appear."""
        new_keys = [k for k in keys if k not in self._fieldnames]
        if self._writer is None:
            # First time — create writer with current fieldnames + new keys
            self._fieldnames.extend(new_keys)
            assert self._file is not None
            self._writer = csv.DictWriter(
                self._file, fieldnames=self._fieldnames, extrasaction="ignore"
            )
            self._writer.writeheader()
        elif new_keys:
            # New columns discovered — we need to rewrite the entire CSV
            self._fieldnames.extend(new_keys)
            self._rewrite_csv()

    def _rewrite_csv(self) -> None:
        """Rewrite the CSV file with an updated header (new columns added)."""
        assert self._file is not None
        self._file.close()

        # Read all existing rows
        existing_rows: list[dict[str, str]] = []
        with open(self._csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)

        # Rewrite with expanded fieldnames
        self._file = open(self._csv_path, "w", newline="")
        self._writer = csv.DictWriter(
            self._file, fieldnames=self._fieldnames, extrasaction="ignore"
        )
        self._writer.writeheader()
        for row in existing_rows:
            self._writer.writerow(row)
        self._file.flush()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """
        Log a dictionary of metrics.

        Args:
            metrics: Key-value pairs of metric names and their values.
            step: Optional step/iteration number. If None, auto-increments.
        """
        if self._finished:
            raise RuntimeError(
                "Cannot log to a finished run. Call init() to start a new run."
            )

        with self._lock:
            if step is None:
                step = self._rows_written + len(self._rows_buffer)

            row: dict[str, Any] = {
                "_step": step,
                "_timestamp": datetime.now().isoformat(),
                **metrics,
            }

            metric_keys = [k for k in metrics.keys()]
            self._ensure_writer(metric_keys)

            assert self._writer is not None
            self._rows_buffer.append(row)

            if len(self._rows_buffer) >= self.flush_every:
                self._flush()

    def _flush(self) -> None:
        """Write buffered rows to disk."""
        if not self._rows_buffer or self._writer is None:
            return
        for row in self._rows_buffer:
            self._writer.writerow(row)
        assert self._file is not None
        self._file.flush()
        self._rows_written += len(self._rows_buffer)
        self._rows_buffer.clear()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def finish(self) -> None:
        """Flush remaining data and close the CSV file."""
        if self._finished:
            return
        with self._lock:
            self._flush()
            if self._file and not self._file.closed:
                self._file.close()
            self._finished = True
        print(
            f"[csv_logger] Run finished. {self._rows_written} rows written to {self._csv_path}"
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "Run":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.finish()

    def __del__(self) -> None:
        if not self._finished:
            self.finish()


# ======================================================================
# Module-level API (mirrors wandb's global interface)
# ======================================================================

_active_run: Run | None = None


def init(
    project: str = "default_project",
    name: str | None = None,
    dir: str | Path = ".",
    config: dict[str, Any] | None = None,
    flush_every: int = 1,
    **kwargs: Any,
) -> Run:
    """
    Initialize a new logging run. Mirrors `wandb.init(...)`.

    Args:
        project: Project name (used as subdirectory).
        name: Run name (used as subdirectory under project).
        dir: Root directory for log output.
        config: Hyperparameters / config dict to persist alongside metrics.
        flush_every: Flush to disk every N log calls (default: 1 for safety).
        **kwargs: Additional keyword arguments (absorbed for wandb compat).

    Returns:
        A Run instance that can also be used as a context manager.
    """
    global _active_run
    if _active_run is not None and not _active_run._finished:
        _active_run.finish()

    _active_run = Run(
        project=project,
        name=name,
        dir=dir,
        config=config,
        flush_every=flush_every,
        **kwargs,
    )
    return _active_run


def log(metrics: dict[str, Any], step: int | None = None) -> None:
    """Log metrics to the active run. Mirrors `wandb.log(...)`."""
    if _active_run is None:
        raise RuntimeError("No active run. Call csv_logger.init() first.")
    _active_run.log(metrics, step=step)


def finish() -> None:
    """Finish the active run."""
    global _active_run
    if _active_run is not None:
        _active_run.finish()
        _active_run = None
