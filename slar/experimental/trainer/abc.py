import torch
import torch.nn as nn
import time
import copy
import logging

from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable

from ..utils.loggers improt create_logger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer(ABC):
    """
    Base Trainer class for training PyTorch models.
    
    Subclass this and implement `compute_loss()` for custom loss calculation.
    
    Usage:
        class MyTrainer(Trainer):
            def compute_loss(self, batch, model):
                inputs, targets = batch
                outputs = model(inputs)
                return nn.functional.cross_entropy(outputs, targets), outputs
                
        trainer = MyTrainer(model, train_loader, val_loader, optimizer, device='cuda')
        history = trainer.fit(epochs=10)
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[str] = None,
        gradient_clip_val: Optional[float] = None,
        accumulation_steps: int = 1,
        callbacks: Optional[list] = None,
        logger: Optional[Any] = None,
    ):
        """
            Args:
            model: PyTorch model to train.
            train_dataloader: DataLoader for training data.
            val_dataloader: DataLoader for validation data (optional).
            optimizer: Optimizer (default: Adam with lr=1e-3).
            scheduler: Learning rate scheduler (optional).
            device: Device string ('cuda', 'cpu', 'mps'). Auto-detected if None.
            gradient_clip_val: Max norm for gradient clipping (optional).
            accumulation_steps: Number of steps to accumulate gradients.
            callbacks: List of TrainerCallback instances (optional).
            logger: Logger instance (optional).
        """
        self.device = device or self._auto_detect_device()
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=1e-3)
        self.scheduler = scheduler
        self.gradient_clip_val = gradient_clip_val
        self.accumulation_steps = accumulation_steps
        self.callbacks = callbacks or []


        # --- Resolve logger --- 
        self.logger, self.callbacks = self._resolve_logger(
            logger, self.callbacks
        )

        # State tracking
        self.current_epoch = 0
        self.global_step = 0

    # ----------------------------------------------------------------------------
    # Resolve single logger
    # ----------------------------------------------------------------------------

    @staticmethod
    def _resolve_logger(
        logger: Optional[Any],
        callbacks: list,
    ) -> tuple[Any, list]:
        """
        Resolve a single logger from `logger` arg and `callbacks` list.

        Rules:
            1. Both provided         → raise ValueError
            2. Only `logger`          → wrap into LoggerCallback, append to callbacks
            3. Only in `callbacks`    → extract logger from it
            4. Neither                → use NullLogger

        Returns:
            (resolved_logger, updated_callbacks)
        """
        # Find any LoggerCallback already in the callbacks list
        logger_callbacks = [
            cb for cb in callbacks if isinstance(cb, LoggerCallback)
        ]
        other_callbacks = [
            cb for cb in callbacks if not isinstance(cb, LoggerCallback)
        ]

        has_logger_arg = logger is not None
        has_logger_cb = len(logger_callbacks) > 0

        # --- Case 1: Both provided → conflict ---
        if has_logger_arg and has_logger_cb:
            raise ValueError(
                "Received a logger via both `logger=` and `callbacks=`. "
                "Provide it through one or the other, not both."
            )

        # --- Case 2: Only `logger` arg → wrap into callback ---
        if has_logger_arg:
            cb = LoggerCallback(logger)
            return logger, other_callbacks + [cb]

        # --- Case 3: Only in callbacks → extract logger from it ---
        if has_logger_cb:
            if len(logger_callbacks) > 1:
                raise ValueError(
                    f"Found {len(logger_callbacks)} LoggerCallbacks in "
                    f"callbacks list. Only one logger is allowed."
                )
            resolved_logger = logger_callbacks[0].logger
            return resolved_logger, callbacks  # keep original list intact

        # --- Case 4: Neither → NullLogger ---
        null = create_logger(None) # NullLogger
        cb = LoggerCallback(null)
        return null, other_callbacks + [cb]

    # ------------------------------------------------------------------
    # Abstract / Hook methods — override in subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def compute_loss(
        self, batch: Any, model: nn.Module
    ) -> tuple[torch.Tensor, Any]:
        """
        Compute the loss for a single batch.

        Args:
            batch: A single batch from the DataLoader.
            model: The model (already on the correct device).

        Returns:
            (loss, outputs): loss is a scalar Tensor; outputs can be
            anything useful (logits, dict of tensors, etc.)
        """
        raise NotImplementedError

    def compute_metrics(
        self, outputs: Any, batch: Any
    ) -> Dict[str, float]:
        """
        (Optional) Compute additional metrics. Override for accuracy, F1, etc.

        Returns:
            Dictionary of metric_name -> value.
        """
        return {}

    def on_epoch_start(self, epoch: int) -> None:
        """Hook called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, epoch: int, logs: Dict[str, float]) -> None:
        """Hook called at the end of each epoch."""
        pass

    def on_train_start(self) -> None:
        """Hook called once before training begins."""
        pass

    def on_train_end(self, history: Dict[str, list]) -> None:
        """Hook called once after training ends."""
        pass

    def on_batch_start(self, batch_idx: int, batch: Any) -> None:
        """Hook called before each training batch."""
        pass

    def on_batch_end(
        self, batch_idx: int, batch: Any, loss: float, outputs: Any
    ) -> None:
        """Hook called after each training batch."""
        pass

    # ------------------------------------------------------------------
    # Core training logic
    # ------------------------------------------------------------------

    def fit(
        self,
        epochs: int,
        early_stopping_patience: Optional[int] = None,
        save_best_model: bool = True,
        verbose: bool = True,
    ) -> Dict[str, list]:
        """
        Train the model.

        Args:
            epochs: Number of epochs to train.
            early_stopping_patience: Stop if val_loss doesn't improve
                for this many epochs. None = disabled.
            save_best_model: Keep a copy of the best model weights.
            verbose: Print progress each epoch.

        Returns:
            Training history dict.
        """
        self._stop_training = False
        patience_counter = 0
        self.on_train_start()
        self._fire_callbacks("on_train_start", trainer=self)

        for epoch in range(1, epochs + 1):
            if self._stop_training:
                logger.info("Training stopped early by flag.")
                break

            self.current_epoch = epoch
            self.on_epoch_start(epoch)
            self._fire_callbacks("on_epoch_start", trainer=self, epoch=epoch)

            epoch_start = time.time()

            # --- Training phase ---
            train_loss, train_metrics = self._train_one_epoch()

            # --- Validation phase ---
            val_loss, val_metrics = (
                self._validate() if self.val_dataloader else (None, {})
            )

            epoch_time = time.time() - epoch_start
            current_lr = self._get_current_lr()

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(current_lr)
            self.history["epoch_time"].append(epoch_time)

            # Store extra metrics in history
            for k, v in {**train_metrics, **val_metrics}.items():
                self.history.setdefault(k, []).append(v)

            logs = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": current_lr,
                "epoch_time": epoch_time,
                **train_metrics,
                **val_metrics,
            }

            # --- Best model tracking & early stopping ---
            if val_loss is not None:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    patience_counter = 0
                    if save_best_model:
                        self.best_model_state = copy.deepcopy(
                            self.model.state_dict()
                        )
                else:
                    patience_counter += 1

                if (
                    early_stopping_patience is not None
                    and patience_counter >= early_stopping_patience
                ):
                    logger.info(
                        f"Early stopping triggered at epoch {epoch} "
                        f"(patience={early_stopping_patience})."
                    )
                    self._stop_training = True

            # --- Logging ---
            if verbose:
                self._log_epoch(logs)

            self.on_epoch_end(epoch, logs)
            self._fire_callbacks("on_epoch_end", trainer=self, epoch=epoch, logs=logs)

        # Restore best model if available
        if save_best_model and self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info("Restored best model weights.")

        self.on_train_end(self.history)
        self._fire_callbacks("on_train_end", trainer=self, history=self.history)

        return self.history

    def _train_one_epoch(self) -> tuple[float, Dict[str, float]]:
        """Run one training epoch. Returns (avg_loss, aggregated_metrics)."""
        self.model.train()
        running_loss = 0.0
        total_samples = 0
        all_metrics: Dict[str, float] = {}
        num_batches = len(self.train_dataloader)

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_dataloader):
            self.on_batch_start(batch_idx, batch)
            batch = self._move_batch_to_device(batch)

            # Forward
            loss, outputs = self.compute_loss(batch, self.model)

            # Scale loss for gradient accumulation
            scaled_loss = loss / self.accumulation_steps
            scaled_loss.backward()

            # Step every `accumulation_steps` or at the last batch
            if (batch_idx + 1) % self.accumulation_steps == 0 or (
                batch_idx + 1
            ) == num_batches:
                if self.gradient_clip_val is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_val
                    )
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            # Book-keeping
            batch_size = self._get_batch_size(batch)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            metrics = self.compute_metrics(outputs, batch)
            for k, v in metrics.items():
                key = f"train_{k}"
                all_metrics[key] = all_metrics.get(key, 0.0) + v * batch_size

            self.on_batch_end(batch_idx, batch, loss.item(), outputs)
            self._fire_callbacks(
                "on_batch_end",
                trainer=self,
                batch_idx=batch_idx,
                loss=loss.item(),
            )

        # LR scheduler step (per-epoch schedulers)
        if self.scheduler is not None:
            self.scheduler.step()

        avg_loss = running_loss / max(total_samples, 1)
        avg_metrics = {k: v / max(total_samples, 1) for k, v in all_metrics.items()}
        return avg_loss, avg_metrics

    @torch.no_grad()
    def _validate(self) -> tuple[float, Dict[str, float]]:
        """Run one validation pass. Returns (avg_loss, aggregated_metrics)."""
        self.model.eval()
        running_loss = 0.0
        total_samples = 0
        all_metrics: Dict[str, float] = {}

        for batch in self.val_dataloader:
            batch = self._move_batch_to_device(batch)
            loss, outputs = self.compute_loss(batch, self.model)

            batch_size = self._get_batch_size(batch)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            metrics = self.compute_metrics(outputs, batch)
            for k, v in metrics.items():
                key = f"val_{k}"
                all_metrics[key] = all_metrics.get(key, 0.0) + v * batch_size

        avg_loss = running_loss / max(total_samples, 1)
        avg_metrics = {k: v / max(total_samples, 1) for k, v in all_metrics.items()}
        return avg_loss, avg_metrics

    # ------------------------------------------------------------------
    # Prediction / evaluation helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, dataloader: DataLoader) -> list:
        """Run inference and return a list of model outputs."""
        self.model.eval()
        results = []
        for batch in dataloader:
            batch = self._move_batch_to_device(batch)
            _, outputs = self.compute_loss(batch, self.model)
            results.append(outputs)
        return results

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Save full training state to a file."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load training state from a file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint.get("current_epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.history = checkpoint.get("history", self.history)
        logger.info(f"Checkpoint loaded from {path}")

    # ------------------------------------------------------------------
    # Utility / private methods
    # ------------------------------------------------------------------

    @staticmethod
    def _auto_detect_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _move_batch_to_device(self, batch: Any) -> Any:
        """Recursively move tensors in batch to self.device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        elif isinstance(batch, (list, tuple)):
            moved = [self._move_batch_to_device(b) for b in batch]
            return type(batch)(moved)
        elif isinstance(batch, dict):
            return {k: self._move_batch_to_device(v) for k, v in batch.items()}
        return batch  # leave non-tensor objects as-is

    @staticmethod
    def _get_batch_size(batch: Any) -> int:
        """Best-effort extraction of batch size."""
        if isinstance(batch, torch.Tensor):
            return batch.size(0)
        elif isinstance(batch, (list, tuple)) and len(batch) > 0:
            first = batch[0]
            if isinstance(first, torch.Tensor):
                return first.size(0)
        elif isinstance(batch, dict):
            for v in batch.values():
                if isinstance(v, torch.Tensor):
                    return v.size(0)
        return 1

    def _get_current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def _log_epoch(self, logs: Dict[str, Any]) -> None:
        parts = [f"Epoch {logs['epoch']:>3d}"]
        parts.append(f"train_loss={logs['train_loss']:.4f}")
        if logs.get("val_loss") is not None:
            parts.append(f"val_loss={logs['val_loss']:.4f}")
        parts.append(f"lr={logs['lr']:.2e}")
        parts.append(f"time={logs['epoch_time']:.1f}s")

        # Any extra metrics
        skip = {"epoch", "train_loss", "val_loss", "lr", "epoch_time"}
        for k, v in logs.items():
            if k not in skip and isinstance(v, (int, float)):
                parts.append(f"{k}={v:.4f}")

        logger.info(" | ".join(parts))

    def _fire_callbacks(self, event_name: str, **kwargs) -> None:
        for cb in self.callbacks:
            fn = getattr(cb, event_name, None)
            if callable(fn):
                fn(**kwargs)

    def stop_training(self) -> None:
        """Call this (e.g. from a callback) to stop after the current epoch."""
        self._stop_training = True


# ======================================================================
# Callback base class
# ======================================================================

class TrainerCallback:
    """
    Base class for Trainer callbacks. Override any method you need.
    """

    def on_train_start(self, trainer: Trainer, **kwargs) -> None:
        pass

    def on_train_end(self, trainer: Trainer, **kwargs) -> None:
        pass

    def on_epoch_start(self, trainer: Trainer, epoch: int, **kwargs) -> None:
        pass

    def on_epoch_end(
        self, trainer: Trainer, epoch: int, logs: Dict[str, float], **kwargs
    ) -> None:
        pass

    def on_batch_end(
        self, trainer: Trainer, batch_idx: int, loss: float, **kwargs
    ) -> None:
        pass

# -----------------------------------------------------------------------------
# Logger Callback
# -----------------------------------------------------------------------------
class LoggerCallback(TrainerCallback):
    """Exception-safe context-manager logger integration."""

    def __init__(self, logger_instance):
        self.logger = logger_instance
        self._context = None

    def on_train_start(self, trainer, **kwargs):
        self._context = self.logger.init()
        self._context.__enter__()

    def on_train_end(self, trainer, **kwargs):
        self._safe_exit()

    def on_epoch_end(self, trainer, epoch, logs, **kwargs):
        self.logger.log(logs)

    def _safe_exit(self):
        if self._context is not None:
            try:
                self._context.__exit__(None, None, None)
            except Exception:
                pass
            finally:
                self._context = None
