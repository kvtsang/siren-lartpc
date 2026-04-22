from __future__ import annotations

import torch
import wandb
import numpy as np
import boost_histogram as bh
import matplotlib.pyplot as plt

from contextlib import nullcontext
from typing import Dict, Any, Optional
from tqdm.auto import tqdm

from . import SirenVis
from .factories import (
    create_dataloader,
    create_optimizer,
    create_scheduler,
    create_logger,
    create_checkpoint_manager,
)

def weighted_logit_mse_loss(logit_pred, p_hat, eps=1e-7):
    # Clip to avoid inf in logit
    p_hat = p_hat.clamp(eps, 1 - eps)
    target_logit = torch.logit(p_hat)
    w = p_hat * (1 - p_hat)
    return (w * (logit_pred - target_logit) ** 2).mean()

class SirenTrainer:

    """Trainer class for training a neural network model in PyTorch."""
    
    def __init__(self, cfg: Dict[str, Any]):
        """
        Initialize the trainer from a single nested config dictionary.

        Args:
            cfg: Nested dictionary with keys 'trainer', 'model', 'dataoader',
                 'criterion', 'optimizer', and optionally 'logger'.
        """
        self.cfg = cfg
        self.trainer_cfg = cfg.get("trainer", {})

        # -- Device -------------------------------------------------------
        self.device = torch.device(
            self.trainer_cfg.get(
                "device", "cuda" if torch.cuda.is_available() else "cpu"
            )
        )

        # -- Build components from factories ------------------------------
        self.model = SirenVis.create(cfg['model']).to(self.device)
        self.dataloader = create_dataloader(**cfg["dataloader"])
        self.optimizer = create_optimizer(self.model, cfg['optimizer'])
        
        if 'scheduler' in cfg:
            self.scheduler = create_scheduler(self.optimizer, cfg['scheduler'])
        else:
            self.scheduler = None

        self.logger = create_logger(cfg.get('logger', None))
        self.loss_fn = weighted_logit_mse_loss
        
        # -- Trainer hyper-parameters -------------------------------------
        self.max_epochs = self.trainer_cfg.get("max_epochs", 100)
        self.grad_clip_max_norm = self.trainer_cfg.get("grad_clip_max_norm", None)
        self.log_every_n_steps = self.trainer_cfg.get("log_every_n_steps", 50)
        self.val_every_n_epochs = self.trainer_cfg.get("val_every_n_epochs", 1)
        self.plot_every_n_vals = self.trainer_cfg.get("plot_every_n_vals", 1)

        # -- State tracking -----------------------------------------------
        self.current_epoch = 0
        self.global_step = 0

    # -- Helpers --------------------------------------------------------------
    def _move_batch_to_device(
        self, batch: Tuple[torch.Tensor],
    ) -> Tuple[torch.Tensor]:
        
        """
        Convert and move batch to ``self.device``. 
        """
        
        return tuple(v.to(self.device) for v in batch)


    def _compute_loss(self, batch: Tuple[torch.Tensor]) -> torch.Tensor:
        coords, target = batch
        out = self._forward(coords)
        loss = self.loss_fn(out, target)
        
        return loss, out


    def _step_scheduler(self, metric: Optional[float] = None) -> None:
        """Step the LR scheduler, handling ReduceLROnPlateau specially."""
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            # No metric provided this epoch — skip plateau check intentionally.
            if metric is None:
                return
            self.scheduler.step(metric)
        else:
            self.scheduler.step()

    def _forward(self, x: torch.Tensor):
        return self.model.siren_at(x)

           
    # -- Training / Evaluation loops -----------------------------------------
    def _run_train_epoch(self, pbar: tqdm) -> float:
        """Run one training epoch using a reusable progress bar."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
    
        pbar.reset()
        pbar.set_description(f"Epoch {self.current_epoch+1}/{self.max_epochs} [train]")
    
        for batch in self.dataloader:
            batch = self._move_batch_to_device(batch)
    
            loss, out = self._compute_loss(batch)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.grad_clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_max_norm
                )
            self.optimizer.step()
    
            loss_val = loss.item()
            total_loss += loss_val
            num_batches += 1
            self.global_step += 1
    
            pbar.set_postfix(loss=f"{loss_val:.6f}", step=self.global_step)
            pbar.update(1)
    
            if self.global_step % self.log_every_n_steps == 0:
                self.logger.log(
                    {
                        "train/step_loss": loss_val,
                        "lr": self.optimizer.param_groups[0]["lr"],
                    },
                    step=self.global_step,
                )
    
        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _run_eval_epoch(self, pbar: tqdm) -> float:
        """Run one eval epoch using a reusable progress bar."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
    
        pbar.reset()
        pbar.set_description(
            f"Epoch {self.current_epoch+1}/{self.max_epochs} [val]"
        )

        pbar.update(1)
        for batch in self.dataloader:
            batch = self._move_batch_to_device(batch)
            loss, out = self._compute_loss(batch)
            loss_val = loss.item()
            total_loss += loss_val
            num_batches += 1
    
            pbar.set_postfix(loss=f"{loss_val:.6f}")
            pbar.update(1)

        return total_loss / max(num_batches, 1)
        
    def validate(self) -> float:
        with tqdm(
            total=len(self.loaders["val"]), 
            desc="[val]", dynamic_ncols=True
        ) as pbar:
            return self._run_eval_epoch(pbar)


    # -- Main entry point ----------------------------------------------------
    def fit(self) -> None:
        with self.logger.init():
            self.checkpoint_manager = self._init_checkpoint_manager()
            self._fit()

        if self.checkpoint_manager is not None:
            self.checkpoint_manager.close()
            
    def _fit(self) -> None:
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"▸ Training on {self.device}  |  Parameters: {n_params:,}")
        self.logger.log({"model/parameters": n_params})
    
        # -- Create reusable progress bars ----------------------------
        train_pbar = tqdm(
            total=len(self.dataloader),
            desc="[train]",
            leave=True,
            dynamic_ncols=True,
            mininterval=1.,
        )

        val_pbar = tqdm(
            total=len(self.dataloader),
            desc="[val]",
            leave=True,
            dynamic_ncols=True,
            mininterval=1.,
        )

        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
    
            # -- train --------------------------------------------
            train_loss = self._run_train_epoch(train_pbar)
            self.logger.log(
                {"train/epoch_loss": train_loss, "epoch": epoch},
                step=self.global_step,
            )

    
            # -- validate -----------------------------------------
            val_loss = None
            if (epoch + 1) % self.val_every_n_epochs == 0:
                val_loss = self._run_eval_epoch(val_pbar)
                self.logger.log(
                    {"val/epoch_loss": val_loss, "epoch": epoch},
                    step=self.global_step,
                )
    
            # -- checkpoint ---------------------------------------
            self.save_checkpoint(val_loss)
    
            # -- scheduler ----------------------------------------
            self._step_scheduler(val_loss)
    
            # -- epoch summary -----------------------------------
            lr = self.optimizer.param_groups[0]["lr"]
            self.logger.log({'lr': lr}, step=self.global_step)

        train_pbar.close()
        val_pbar.close()
        tqdm.write("✔ Training complete.")

    # -- Checkpoint ----------------------------------------------
    def _init_checkpoint_manager(self):
        cfg = self.cfg.get('checkpoint')

        if cfg is None:
            return None

        if self.logger is None:
            return create_checkpoint_manager(cfg)

        return create_checkpoint_manager(
            cfg, self.logger.run.project, self.logger.run.name
        )
    
    def save_checkpoint(
        self,
        val_loss: float = None
    ) -> None:
        if self.checkpoint_manager is None:
            return

        if self.scheduler is not None:
            extra = {'scheduler': self.scheduler}
        else:
            extra = None

        self.checkpoint_manager.save(
            epoch=self.current_epoch,
            model=self.model,
            optimizer=self.optimizer,
            metric_value=val_loss,
            extra=extra,
        )
