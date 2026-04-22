"""
PyTorch Checkpoint Manager
==========================

A production-ready checkpoint manager for PyTorch training loops
that handles:

- **Periodic saving** – numbered snapshots every *N* epochs.
- **Last checkpoint** – saved once at the end of training (via
  :meth:`finalize` or :meth:`close`), with a hard link kept
  up-to-date during training so ``checkpoint_last.pt`` always
  points to the most recent numbered checkpoint on disk.
- **Best checkpoint** – hard-links to the file with the best
  tracked metric, avoiding disk-space duplication.
- **Metric tracking** – supports both ``"min"`` (e.g. loss) and
  ``"max"`` (e.g. accuracy) optimisation directions.
- **Atomic writes** – uses tmp-file-then-rename to prevent
  corruption from interrupted saves.
- **Optional pruning** – caps the number of numbered checkpoint
  files on disk.
- **Background I/O** – serialization and disk writes happen in a
  dedicated thread so the training loop is not blocked.

Saving strategy
---------------

To minimise I/O, the manager distinguishes three file roles:

``checkpoint_epoch_NNNNN.pt``
    Physical files written every *save_every_n_epochs* epochs.
    These are the only files that trigger a full ``torch.save``
    during training.

``checkpoint_last.pt``
    A **hard link** that always points to the most recently
    written numbered checkpoint.  Updated cheaply (inode
    operation, no data copied) after every numbered save.  When
    :meth:`finalize` / :meth:`close` is called, if the final
    epoch did *not* coincide with a numbered save, one final
    physical write is performed so that no work is lost.

``checkpoint_best.pt``
    A hard link to whichever numbered checkpoint achieved the
    best value of the tracked metric.

Typical directory layout during training::

    checkpoints/
    ├── checkpoint_epoch_00010.pt
    ├── checkpoint_epoch_00020.pt
    ├── checkpoint_epoch_00030.pt  ← physical file
    ├── checkpoint_last.pt         ← hard link → epoch 00030
    └── checkpoint_best.pt         ← hard link → epoch 00020

After :meth:`finalize` (if training ended at epoch 37)::

    checkpoints/
    ├── ...
    ├── checkpoint_epoch_00030.pt
    ├── checkpoint_epoch_00037.pt  ← written by finalize
    ├── checkpoint_last.pt         ← hard link → epoch 00037
    └── checkpoint_best.pt         ← hard link → epoch 00020

Quick start
-----------

.. code-block:: python

    from checkpoint_manager import CheckpointManager

    manager = CheckpointManager(
        dir="./checkpoints",
        save_every_n_epochs=10,
        metric_name="val_loss",
        metric_mode="min",
    )

    for epoch in range(1, num_epochs + 1):
        train_one_epoch(model, optimizer, train_loader)
        val_loss = validate(model, val_loader)

        manager.save(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            metric_value=val_loss,
            extra={
                "scheduler": scheduler.state_dict(),
            },
        )

    # Writes the final checkpoint_last.pt (if needed) and
    # shuts down the background thread.
    manager.finalize(model=model, optimizer=optimizer)

    # Later — restore the best model for evaluation
    ckpt = manager.load_best(model)
    print(
        f"Best val_loss: {ckpt['metric_value']:.4f} "
        f"@ epoch {ckpt['epoch']}"
    )

Checkpoint payload schema
-------------------------

Every ``.pt`` file written by the manager contains a dictionary
with at least the following keys:

.. code-block:: python

    {
        "epoch":                int,
        "model_state_dict":     OrderedDict,
        "optimizer_state_dict": OrderedDict,
        "metric_name":          str,
        "metric_mode":          str,
        "metric_value":         float | None,
        "best_metric":          float | None,
        "best_epoch":           int   | None,
        "extra":                dict  | None,
    }
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages saving, linking, and loading of PyTorch
    training checkpoints.

    Parameters
    ----------
    dir : str | Path
        Root directory for all checkpoint files.  Created
        automatically (including parents) if it does not exist.
    save_every_n_epochs : int, default ``1``
        A **numbered** checkpoint
        (``checkpoint_epoch_00010.pt``) is written every *N*
        epochs.  ``checkpoint_last.pt`` is maintained as a hard
        link to the most recent numbered file — it is **not**
        rewritten from scratch each epoch.
    metric_name : str, default ``"val_loss"``
        Human-readable name of the metric being tracked.  Stored
        inside each checkpoint for provenance and used in log
        messages.
    metric_mode : ``"min"`` | ``"max"``, default ``"min"``
        Optimisation direction for the tracked metric:

        * ``"min"`` – lower is better (e.g. loss, error rate).
        * ``"max"`` – higher is better (e.g. accuracy, F1).
    max_checkpoints : int | None, default ``None``
        If set, only the most recent *N* **numbered** checkpoint
        files are kept on disk.  Older files are deleted
        automatically after each periodic save.
        ``checkpoint_last.pt`` and ``checkpoint_best.pt`` are
        **never** pruned.  ``None`` disables pruning entirely.
    async_save : bool, default ``True``
        If ``True``, state dicts are snapshotted (moved to CPU)
        in the calling thread, then serialization + disk I/O
        happens in a background thread so the GPU can resume
        training immediately.  If ``False``, all work is done
        synchronously in the caller.

    Attributes
    ----------
    best_metric : float | None
        The best metric value observed so far, or ``None``
        before the first metric is recorded.
    best_epoch : int | None
        The epoch at which :attr:`best_metric` was observed.

    Raises
    ------
    ValueError
        If ``save_every_n_epochs < 1`` or ``metric_mode`` is
        not ``"min"`` / ``"max"``.

    Notes
    -----
    **Hard links vs. copies** – ``checkpoint_best.pt`` and
    ``checkpoint_last.pt`` are created with :func:`os.link`,
    sharing the same inode (and disk blocks) as the numbered
    file they point to.  This is instant and uses no additional
    disk space.  If the target file is later deleted by pruning,
    the hard link keeps the data alive until the link itself is
    replaced.

    **Atomic writes** – Each file is first written to a
    temporary path (``*.pt.tmp``) and then renamed.  On POSIX
    systems ``rename(2)`` is atomic, so a crash mid-write will
    never leave a corrupt checkpoint at the final path.

    **Background saving** – When ``async_save=True`` the manager
    keeps a single-thread
    :class:`~concurrent.futures.ThreadPoolExecutor`.  The
    ``state_dict()`` snapshot (GPU → CPU copy) still happens in
    the calling thread to guarantee a consistent snapshot, but
    the expensive :func:`torch.save` serialization and disk
    write are offloaded.  :meth:`save` will **block** if a
    previous background save has not yet finished, so at most
    one write is in flight at any time.

    **Finalization** – Call :meth:`finalize` (or :meth:`close`,
    or use the context-manager protocol) at the end of training.
    If the last epoch was not a numbered-save epoch, a final
    physical write ensures no work is lost.

    **Thread / process safety** – The manager is *not*
    thread-safe.  In distributed training, only the rank-0
    process should call :meth:`save`.

    Examples
    --------
    Save every 5 epochs, track accuracy (higher is better):

    >>> manager = CheckpointManager(
    ...     "./ckpts",
    ...     save_every_n_epochs=5,
    ...     metric_name="val_acc",
    ...     metric_mode="max",
    ... )
    >>> for epoch in range(1, 26):
    ...     manager.save(
    ...         epoch=epoch,
    ...         model=model,
    ...         optimizer=optim,
    ...         metric_value=acc,
    ...     )
    >>> manager.finalize(model, optim)
    >>> manager.best_metric
    0.95

    As a context manager (call ``finalize`` before exit if you
    need the final-epoch write):

    >>> with CheckpointManager(
    ...     "./ckpts", save_every_n_epochs=10,
    ... ) as mgr:
    ...     for epoch in range(1, 101):
    ...         mgr.save(
    ...             epoch=epoch,
    ...             model=model,
    ...             optimizer=optim,
    ...         )
    ...     mgr.finalize(model, optim)
    """

    #: Filename for the most-recent checkpoint (hard link
    #: during training; physical file only from finalize).
    LAST_FILENAME: str = "checkpoint_last.pt"

    #: Filename for the best-metric checkpoint (hard link).
    BEST_FILENAME: str = "checkpoint_best.pt"

    def __init__(
        self,
        dir: str | Path,
        save_every_n_epochs: int = 1,
        metric_name: str = "val_loss",
        metric_mode: Literal["min", "max"] = "min",
        max_checkpoints: Optional[int] = None,
        async_save: bool = True,
    ) -> None:
        if save_every_n_epochs < 1:
            raise ValueError(
                f"save_every_n_epochs must be >= 1, "
                f"got {save_every_n_epochs}"
            )
        if metric_mode not in ("min", "max"):
            raise ValueError(
                f"metric_mode must be 'min' or 'max', "
                f"got '{metric_mode}'"
            )

        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)

        self.save_every_n_epochs = save_every_n_epochs
        self.metric_name = metric_name
        self.metric_mode = metric_mode
        self.max_checkpoints = max_checkpoints
        self.async_save = async_save

        self._best_metric: Optional[float] = None
        self._best_epoch: Optional[int] = None
        self._numbered_checkpoints: list[Path] = []

        # Most recent payload snapshot (CPU tensors).  Retained
        # so that ``finalize`` can write a final checkpoint
        # without the caller having to pass model/optimizer
        # again — unless they want the absolute latest state.
        self._last_payload: Optional[Dict[str, Any]] = None
        self._last_saved_epoch: Optional[int] = None
        self._last_snapshot_epoch: Optional[int] = None

        # Background writer — max 1 concurrent save to bound
        # memory usage.
        self._executor: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="ckpt_writer",
            )
            if async_save
            else None
        )
        self._pending_future: Optional[Future] = None

    # -------------------------------------------------------------
    # Context-manager protocol
    # -------------------------------------------------------------

    def __enter__(self) -> "CheckpointManager":
        return self

    def __exit__(
        self, exc_type, exc_val, exc_tb,
    ) -> None:
        self.close()

    # -------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------

    @property
    def best_metric(self) -> Optional[float]:
        """Best metric value observed so far, or ``None``."""
        return self._best_metric

    @property
    def best_epoch(self) -> Optional[int]:
        """Epoch of :attr:`best_metric`, or ``None``."""
        return self._best_epoch

    # -------------------------------------------------------------
    # Saving
    # -------------------------------------------------------------

    def save(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metric_value: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record state for the current epoch.

        Two things happen every call:

        1. A **CPU snapshot** of the model and optimizer state
           dicts is taken and cached internally (cheap — no disk
           I/O).  This cached snapshot is used by
           :meth:`finalize` if the final epoch doesn't coincide
           with a numbered save.
        2. If ``epoch`` is divisible by
           :attr:`save_every_n_epochs`, a **numbered checkpoint**
           is written to disk and ``checkpoint_last.pt`` is
           updated as a hard link.

        Additionally, if *metric_value* is provided and
        represents an improvement, ``checkpoint_best.pt`` is
        hard-linked to the numbered file.

        When ``async_save=True``, disk I/O is offloaded to a
        background thread.  If the previous background save is
        still in progress, this method **blocks** until it
        completes before starting the new save.  This ensures at
        most one save is in flight and bounds peak memory to
        ≈ 2× one checkpoint.

        Parameters
        ----------
        epoch : int
            Current epoch number (1-based by convention).
        model : torch.nn.Module
            Model whose ``state_dict()`` will be saved.
        optimizer : torch.optim.Optimizer
            Optimizer whose ``state_dict()`` will be saved.
        metric_value : float | None, optional
            Value of the tracked metric for this epoch.  If
            ``None``, the best-checkpoint logic is skipped.
        extra : dict[str, Any] | None, optional
            Arbitrary additional objects to store, e.g.::

                {
                    "scheduler": scheduler.state_dict(),
                    "scaler":    scaler.state_dict(),
                }

            Each value should be serialisable by
            :func:`torch.save`.

        Examples
        --------
        >>> manager.save(
        ...     epoch=5,
        ...     model=model,
        ...     optimizer=optimizer,
        ...     metric_value=0.032,
        ...     extra={
        ...         "scheduler": scheduler.state_dict(),
        ...     },
        ... )
        """
        # --- Snapshot state dicts (GPU → CPU) in the calling
        #     thread -----------------------------------------------
        t0 = time.monotonic()
        payload = self._build_payload(
            epoch, model, optimizer, metric_value, extra,
        )
        snapshot_ms = (time.monotonic() - t0) * 1000
        logger.debug(
            "State-dict snapshot took %.1f ms", snapshot_ms,
        )

        # Cache for finalize
        self._last_payload = payload
        self._last_snapshot_epoch = epoch

        # --- Determine actions ------------------------------------
        save_numbered = (
            epoch % self.save_every_n_epochs == 0
        )
        numbered_path = (
            self.dir / self._epoch_filename(epoch)
            if save_numbered
            else None
        )

        is_new_best = (
            metric_value is not None
            and self._is_improvement(metric_value)
        )
        if is_new_best:
            self._best_metric = metric_value
            self._best_epoch = epoch

        # Nothing to write to disk this epoch
        if not save_numbered and not is_new_best:
            return

        if numbered_path is not None:
            self._numbered_checkpoints.append(numbered_path)

        # --- Dispatch disk I/O ------------------------------------
        write_task = _WriteTask(
            payload=payload,
            dir=self.dir,
            numbered_path=numbered_path,
            update_last_link=save_numbered,
            last_filename=self.LAST_FILENAME,
            is_new_best=is_new_best,
            best_filename=self.BEST_FILENAME,
            metric_name=self.metric_name,
            metric_value=metric_value,
            epoch=epoch,
        )

        if self._executor is not None:
            self._wait_for_pending()
            self._pending_future = self._executor.submit(
                write_task.run,
            )
        else:
            write_task.run()

        if save_numbered:
            self._last_saved_epoch = epoch

        # --- Prune old numbered checkpoints -----------------------
        self._cleanup_old_checkpoints()

    # -------------------------------------------------------------
    # Finalization
    # -------------------------------------------------------------

    def finalize(
        self,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write the final ``checkpoint_last.pt`` and shut down.

        Call this at the end of the training loop.  If the last
        epoch already coincided with a numbered save, only the
        hard link is verified and no redundant write occurs.
        Otherwise a new physical checkpoint file is written for
        the final epoch.

        If *model* and *optimizer* are provided, a fresh snapshot
        is taken (useful if state has changed since the last
        :meth:`save` call, e.g. an LR-scheduler step).  If
        omitted, the most recent cached snapshot from
        :meth:`save` is reused.

        Parameters
        ----------
        model : torch.nn.Module | None, optional
            If provided together with *optimizer*, a fresh CPU
            snapshot is taken.
        optimizer : torch.optim.Optimizer | None, optional
            Required if *model* is provided.
        extra : dict[str, Any] | None, optional
            Additional state to include (only used when *model*
            and *optimizer* are provided).

        Raises
        ------
        RuntimeError
            If no snapshot is available (i.e. :meth:`save` was
            never called) and *model* / *optimizer* are not
            provided.

        Examples
        --------
        End-of-training with a fresh snapshot:

        >>> manager.finalize(
        ...     model=model, optimizer=optimizer,
        ... )

        End-of-training reusing the last cached snapshot:

        >>> manager.finalize()
        """
        # Flush any in-flight background write first
        self._wait_for_pending()

        # Optionally take a fresh snapshot
        if model is not None and optimizer is not None:
            epoch = (
                self._last_snapshot_epoch
                if self._last_snapshot_epoch is not None
                else 0
            )
            metric_value = (
                self._last_payload.get("metric_value")
                if self._last_payload
                else None
            )
            self._last_payload = self._build_payload(
                epoch, model, optimizer,
                metric_value=metric_value,
                extra=extra,
            )
            self._last_snapshot_epoch = epoch

        if self._last_payload is None:
            raise RuntimeError(
                "finalize() called but no snapshot is "
                "available.  Either call save() at least "
                "once, or pass model and optimizer to "
                "finalize()."
            )

        # If the last snapshot was already written as a numbered
        # file, just ensure the hard link is current.
        already_saved = (
            self._last_snapshot_epoch == self._last_saved_epoch
        )
        if already_saved:
            logger.info(
                "Last epoch %d was already saved as a "
                "numbered checkpoint; verifying "
                "checkpoint_last.pt hard link.",
                self._last_snapshot_epoch,
            )
            numbered_path = (
                self.dir
                / self._epoch_filename(self._last_saved_epoch)
            )
            if numbered_path.exists():
                _hardlink(
                    target=numbered_path,
                    link=(
                        self.dir
                        / self.LAST_FILENAME
                    ),
                )
        else:
            # Write a new numbered file for the final epoch and
            # link it.
            epoch = self._last_snapshot_epoch or 0
            final_path = (
                self.dir
                / self._epoch_filename(epoch)
            )
            logger.info(
                "Writing final checkpoint for epoch %d "
                "→ %s",
                epoch,
                final_path,
            )
            _atomic_save(self._last_payload, final_path)
            _hardlink(
                target=final_path,
                link=(
                    self.dir / self.LAST_FILENAME
                ),
            )
            self._last_saved_epoch = epoch
            logger.info(
                "Hard-linked %s → %s",
                self.LAST_FILENAME,
                final_path.name,
            )

        # Release cached payload to free memory
        self._last_payload = None

        # Shut down the thread pool
        self._shutdown_executor()

    def close(self) -> None:
        """Flush pending writes and shut down the background
        thread pool.

        Unlike :meth:`finalize`, this does **not** write a new
        physical checkpoint for the final epoch.  Use this only
        if you have already called :meth:`finalize`, or if you
        are certain the last epoch coincided with a numbered
        save.

        For the safest end-of-training cleanup, prefer
        :meth:`finalize`.  The context-manager ``__exit__``
        calls :meth:`close`.
        """
        self._wait_for_pending()
        self._shutdown_executor()

    def flush(self) -> None:
        """Block until any in-flight background save completes.

        Useful when you need to guarantee that the latest
        numbered checkpoint is on disk before proceeding — for
        example, before evaluation or before starting a new
        distributed-training phase.

        Does nothing if no background save is pending or if
        ``async_save=False``.
        """
        self._wait_for_pending()

    # -------------------------------------------------------------
    # Loading
    # -------------------------------------------------------------

    def load(
        self,
        path: str | Path,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        map_location: Any = None,
    ) -> Dict[str, Any]:
        """Load a checkpoint and restore model (and optionally
        optimizer) state.

        After loading, the manager's internal :attr:`best_metric`
        and :attr:`best_epoch` are synchronised with the values
        stored in the checkpoint so that training can resume
        without losing the notion of "best so far".

        Parameters
        ----------
        path : str | Path
            Path to a ``.pt`` file produced by :meth:`save`.
        model : torch.nn.Module
            Model whose ``state_dict`` will be replaced.
        optimizer : torch.optim.Optimizer | None, optional
            If provided, its ``state_dict`` is restored too.
        map_location : Any, optional
            Forwarded to :func:`torch.load` (e.g. ``"cpu"``,
            ``torch.device("cuda:0")``).

        Returns
        -------
        dict[str, Any]
            The full checkpoint dictionary.  Callers can use
            this to retrieve ``epoch``, ``metric_value``,
            ``extra``, etc.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.

        Examples
        --------
        Resume training from the latest state:

        >>> ckpt = manager.load(
        ...     "checkpoints/checkpoint_last.pt",
        ...     model, optimizer,
        ... )
        >>> start_epoch = ckpt["epoch"] + 1

        Load only the model (e.g. for inference):

        >>> ckpt = manager.load(
        ...     "checkpoints/checkpoint_best.pt",
        ...     model,
        ... )
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}"
            )

        # Ensure any pending write is finished before reading
        self.flush()

        checkpoint: Dict[str, Any] = torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"],
        )

        if (
            optimizer is not None
            and "optimizer_state_dict" in checkpoint
        ):
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"],
            )

        # Restore manager best-metric tracking state
        if checkpoint.get("best_metric") is not None:
            self._best_metric = checkpoint["best_metric"]
            self._best_epoch = checkpoint.get("best_epoch")

        logger.info(
            "Loaded checkpoint from %s (epoch %d)",
            path,
            checkpoint.get("epoch", -1),
        )
        return checkpoint

    def load_best(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        map_location: Any = None,
    ) -> Dict[str, Any]:
        """Convenience wrapper: load ``checkpoint_best.pt``.

        Parameters
        ----------
        model : torch.nn.Module
            Model to restore.
        optimizer : torch.optim.Optimizer | None, optional
            Optimizer to restore.
        map_location : Any, optional
            Forwarded to :func:`torch.load`.

        Returns
        -------
        dict[str, Any]
            Full checkpoint dictionary.

        Raises
        ------
        FileNotFoundError
            If ``checkpoint_best.pt`` does not exist (no metric
            has improved yet, or the directory was cleared).
        """
        return self.load(
            self.dir / self.BEST_FILENAME,
            model,
            optimizer,
            map_location,
        )

    def load_last(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        map_location: Any = None,
    ) -> Dict[str, Any]:
        """Convenience wrapper: load ``checkpoint_last.pt``.

        Parameters
        ----------
        model : torch.nn.Module
            Model to restore.
        optimizer : torch.optim.Optimizer | None, optional
            Optimizer to restore.
        map_location : Any, optional
            Forwarded to :func:`torch.load`.

        Returns
        -------
        dict[str, Any]
            Full checkpoint dictionary.

        Raises
        ------
        FileNotFoundError
            If ``checkpoint_last.pt`` does not exist
            (:meth:`save` was never called with a numbered
            epoch, or :meth:`finalize` was not called).
        """
        return self.load(
            self.dir / self.LAST_FILENAME,
            model,
            optimizer,
            map_location,
        )

    # -------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------

    def _build_payload(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metric_value: Optional[float],
        extra: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Assemble the checkpoint dict with all tensors on CPU.

        The CPU copy happens here (in the calling thread) so
        that we get a consistent snapshot *before* the next
        forward pass mutates any buffers.  Moving to CPU also
        releases GPU memory immediately.

        Returns
        -------
        dict[str, Any]
            Checkpoint payload matching the schema described
            in the module docstring.
        """
        model_sd = {
            k: v.cpu()
            for k, v in model.state_dict().items()
        }
        optimizer_sd = _cpu_optimizer_state_dict(optimizer)

        payload: Dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": model_sd,
            "optimizer_state_dict": optimizer_sd,
            "metric_name": self.metric_name,
            "metric_mode": self.metric_mode,
            "metric_value": metric_value,
            "best_metric": self._best_metric,
            "best_epoch": self._best_epoch,
        }
        if extra is not None:
            payload["extra"] = extra
        return payload

    @staticmethod
    def _epoch_filename(epoch: int) -> str:
        """Return the numbered filename for a given epoch.

        Examples
        --------
        >>> CheckpointManager._epoch_filename(42)
        'checkpoint_epoch_00042.pt'
        """
        return f"checkpoint_epoch_{epoch:05d}.pt"

    def _is_improvement(self, metric_value: float) -> bool:
        """Return ``True`` if *metric_value* improves on
        :attr:`best_metric`.

        On the very first call (when :attr:`best_metric` is
        ``None``), any finite value counts as an improvement.
        """
        if self._best_metric is None:
            return True
        if self.metric_mode == "min":
            return metric_value < self._best_metric
        return metric_value > self._best_metric

    def _wait_for_pending(self) -> None:
        """Block until the previous background save completes.

        Re-raises any exception from the background thread.
        """
        if self._pending_future is not None:
            self._pending_future.result()
            self._pending_future = None

    def _shutdown_executor(self) -> None:
        """Shut down the thread pool if it exists."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def _cleanup_old_checkpoints(self) -> None:
        """Delete oldest numbered checkpoints exceeding
        :attr:`max_checkpoints`.

        ``checkpoint_last.pt`` and ``checkpoint_best.pt`` are
        never affected.  If a deleted file is the current
        hard-link target of one of those links, the data remains
        accessible through the hard link until it too is
        replaced.
        """
        if self.max_checkpoints is None:
            return
        while (
            len(self._numbered_checkpoints)
            > self.max_checkpoints
        ):
            old = self._numbered_checkpoints.pop(0)
            if old.exists():
                old.unlink()
                logger.debug(
                    "Pruned old checkpoint: %s", old,
                )


# =============================================================
# Free-standing helpers
# =============================================================


def _cpu_optimizer_state_dict(
    optimizer: torch.optim.Optimizer,
) -> Dict[str, Any]:
    """Return optimizer state dict with all tensors on CPU.

    Optimizer ``state`` entries often contain GPU tensors (e.g.
    Adam's ``exp_avg`` and ``exp_avg_sq``).  Moving them to CPU
    frees GPU memory and makes the dict safe to hand off to a
    background thread.
    """
    raw = optimizer.state_dict()
    cpu_state: Dict[str, Any] = {}
    for param_id, state_entry in raw.get("state", {}).items():
        cpu_state[param_id] = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v
            for k, v in state_entry.items()
        }
    return {
        "state": cpu_state,
        "param_groups": raw["param_groups"],
    }


def _atomic_save(
    payload: Dict[str, Any], path: Path,
) -> None:
    """Write *payload* to *path* atomically via tmp + rename.

    Guarantees that *path* is never left in a half-written state
    even if the process is killed during :func:`torch.save`.
    """
    tmp_path = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp_path)
    tmp_path.rename(path)


def _hardlink(target: Path, link: Path) -> None:
    """Create or replace *link* as a hard link to *target*.

    If *link* already exists it is removed first so that
    :func:`os.link` does not raise ``FileExistsError``.
    """
    if link.exists() or link.is_symlink():
        link.unlink()
    os.link(src=target, dst=link)


# =============================================================
# _WriteTask – disk I/O for background execution
# =============================================================


class _WriteTask:
    """Bundles all disk I/O for a single numbered save.

    Instances are submitted to the background thread pool.  All
    data (the payload dict) has already been moved to CPU, so
    this class never touches the GPU.

    Parameters
    ----------
    payload : dict
        Fully-assembled, CPU-resident checkpoint dict.
    dir : Path
        Root checkpoint directory.
    numbered_path : Path | None
        Destination for the numbered file, or ``None`` if this
        epoch does not get a numbered checkpoint (only possible
        when called for a best-only update on a non-numbered
        epoch).
    update_last_link : bool
        Whether to update ``checkpoint_last.pt`` as a hard link
        to the newly written numbered file.
    last_filename : str
        Filename for the last checkpoint link.
    is_new_best : bool
        Whether to create/update ``checkpoint_best.pt``.
    best_filename : str
        Filename for the best checkpoint link.
    metric_name : str
        For logging only.
    metric_value : float | None
        For logging only.
    epoch : int
        For logging only.
    """

    def __init__(
        self,
        payload: Dict[str, Any],
        dir: Path,
        numbered_path: Optional[Path],
        update_last_link: bool,
        last_filename: str,
        is_new_best: bool,
        best_filename: str,
        metric_name: str,
        metric_value: Optional[float],
        epoch: int,
    ) -> None:
        self.payload = payload
        self.dir = dir
        self.numbered_path = numbered_path
        self.update_last_link = update_last_link
        self.last_filename = last_filename
        self.is_new_best = is_new_best
        self.best_filename = best_filename
        self.metric_name = metric_name
        self.metric_value = metric_value
        self.epoch = epoch

    def run(self) -> None:
        """Execute all disk writes.  Thread-safe."""
        t0 = time.monotonic()

        # 1. Write numbered checkpoint
        if self.numbered_path is not None:
            _atomic_save(self.payload, self.numbered_path)
            logger.info(
                "Saved periodic checkpoint → %s",
                self.numbered_path,
            )

        # 2. Update checkpoint_last.pt hard link
        if (
            self.update_last_link
            and self.numbered_path is not None
        ):
            _hardlink(
                target=self.numbered_path,
                link=(
                    self.dir
                    / self.last_filename
                ),
            )
            logger.debug(
                "Hard-linked %s → %s",
                self.last_filename,
                self.numbered_path.name,
            )

        # 3. Update checkpoint_best.pt hard link
        if (
            self.is_new_best
            and self.numbered_path is not None
        ):
            _hardlink(
                target=self.numbered_path,
                link=(
                    self.dir
                    / self.best_filename
                ),
            )
            logger.info(
                "New best %s=%.6f at epoch %d "
                "→ hard-linked %s",
                self.metric_name,
                self.metric_value,
                self.epoch,
                self.best_filename,
            )

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.debug(
            "Checkpoint I/O for epoch %d took %.1f ms",
            self.epoch,
            elapsed_ms,
        )
