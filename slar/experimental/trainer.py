from __future__ import annotations

import torch
import wandb
import numpy as np
import boost_histogram as bh
import matplotlib.pyplot as plt

from contextlib import nullcontext
from typing import Dict, Any, Optional
from tqdm.auto import tqdm

from .factories import (
    create_dataloader,
    create_optimizer,
    create_scheduler,
    create_checkpoint_manager,
)

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
        self.model = create_model(cfg).to(self.device)
        self.loaders = create_data_loaders(**cfg["dataloader"])
        self.loss_fn = create_instance(cfg["criterion"])
        self.optimizer = create_optimizer(self.model, cfg['optimizer'])
        
        if 'scheduler' in cfg:
            self.scheduler = create_scheduler(self.optimizer, cfg['scheduler'])
        else:
            self.scheduler = None

        self.logger = wandb if 'wandb' in cfg else None
        self.histogram = self._make_histogram()
        
        # -- Trainer hyper-parameters -------------------------------------
        self.max_epochs = self.trainer_cfg.get("max_epochs", 100)
        self.grad_clip_max_norm = self.trainer_cfg.get("grad_clip_max_norm", None)
        self.log_every_n_steps = self.trainer_cfg.get("log_every_n_steps", 50)
        self.val_every_n_epochs = self.trainer_cfg.get("val_every_n_epochs", 1)
        self.log_histo_every_n_vals = self.trainer_cfg.get(
            'log_histo_every_n_vals', 1
        )

        # -- State tracking -----------------------------------------------
        self.current_epoch = 0
        self.global_step = 0

    # -- Helpers --------------------------------------------------------------
    def _move_batch_to_device(self, batch: Dict[str, Any],) -> Dict[str, Any]:
        
        """
        Convert and move only the keys listed in ``BATCH_TENSOR_KEYS`` to
        ``self.device``. Remaining keys are passed through unchanged.
        """
        
        out = {}
        for k, v in batch.items():
            if k in self.BATCH_TENSOR_KEYS:
                if  isinstance(v, np.ndarray):
                    v = torch.from_numpy(v)
                out[k] = v.to(self.device)
            else:
                out[k] = v
        return out

    def _compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Forward pass → masked loss."""
        out = self._forward(batch)
        
        target = batch["flash"]
        target_mask = batch["flash_mask"]
        loss = self.loss_fn(out[target_mask], target[target_mask])
        
        return loss, out

    def _log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log *metrics* to the wandb-compatible logger (if configured)."""
        if self.logger is not None:
            self.logger.log(metrics, step=step)

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

    def _forward(self, batch):
        return MultiFlashHypothesis.apply(
            0, batch["charge"], batch["charge_lengths"], self.model
        )

    # -- Histogram ------------------------------------------------------------
    def _make_histogram(self):
        if 'histogram' not in self.cfg:
            return None

        h_cfg = self.cfg['histogram']
        xmax = h_cfg.get('xmax', 100_000)
        bins = h_cfg.get('bins', 100)

        x_edges = np.logspace(1, np.log10(xmax), bins+1)
        x_edges[0] = 0

        return bh.Histogram(
            bh.axis.Variable(x_edges),
            storage=bh.storage.Mean(),
        )

    def _reset_histogram(self):
        if self._on_histogram_epoch:
            self.histogram.reset()
        
    def _fill_histogram(self, batch, output):
        if not self._on_histogram_epoch:
            return
            
        diff2 = (batch['flash'] - output.detach()) ** 2
        mask = batch['flash_mask']
        self.histogram.fill(
            batch['flash'][mask].cpu(),
            sample=diff2[mask].cpu(),
        )

    def _log_histogram(self):
        if not self._on_histogram_epoch or self.logger is None:
            return

        xs = self.histogram.axes.centers[0]
        ys = self.histogram.values()
        
        #table = self.logger.Table(
        #    data=np.column_stack([xs, ys]),
        #    columns=['flash_data', 'l2_loss'],
        #)
        #line = self.logger.plot.line(
        #    table, 'flash_data', 'l2_loss',
        #    title='Loss Profile',
        #)
        #self._log({'loss_profile' : line}, step=self.global_step)

        fig, ax = plt.subplots()
        ax.loglog(xs, ys)
        ax.set_xlabel('flash data (p.e.)')
        ax.set_ylabel('L2 loss')
        ax.grid(True, which='both', ls='--', lw=0.5)
        self._log({'loss_profile': wandb.Image(fig)}, step=self.global_step)
        plt.close()

    @property
    def _on_histogram_epoch(self):
        if self.histogram is None:
            return False
            
        n = self.val_every_n_epochs * self.log_histo_every_n_vals
        return (self.current_epoch + 1) % n == 0
            
    # -- Training / Evaluation loops -----------------------------------------
    def _run_train_epoch(self, pbar: tqdm) -> float:
        """Run one training epoch using a reusable progress bar."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
    
        pbar.reset()
        pbar.set_description(f"Epoch {self.current_epoch+1}/{self.max_epochs} [train]")
    
        for batch in self.loaders["train"]:
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
                self._log(
                    {
                        "train/step_loss": loss_val,
                        "lr": self.optimizer.param_groups[0]["lr"],
                    },
                    step=self.global_step,
                )
    
        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _run_eval_epoch(self, name: str, pbar: tqdm) -> float:
        """Run one eval epoch using a reusable progress bar."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
    
        pbar.reset()
        pbar.set_description(f"Epoch {self.current_epoch+1}/{self.max_epochs} [{name}]")

        self._reset_histogram()
        for batch in self.loaders[name]:
            batch = self._move_batch_to_device(batch)
            loss, out = self._compute_loss(batch)
            loss_val = loss.item()
            total_loss += loss_val
            num_batches += 1
    
            pbar.set_postfix(loss=f"{loss_val:.6f}")
            pbar.update(1)

            self._fill_histogram(batch, out)
        self._log_histogram()
        
        return total_loss / max(num_batches, 1)
        
    def validate(self) -> float:
        with tqdm(
            total=len(self.loaders["val"]), 
            desc="[val]", dynamic_ncols=True
        ) as pbar:
            return self._run_eval_epoch("val", pbar)

    def test(self) -> float:
        with tqdm(
            total=len(self.loaders["test"]), 
            desc="[test]", dynamic_ncols=True
        ) as pbar:
            return self._run_eval_epoch("test", pbar)
        
    # -- Main entry point ----------------------------------------------------
    def fit(self) -> None:
        if self.logger is not None:
            ctx = self.logger.init(**self.cfg["wandb"])
        else:
            ctx = nullcontext()
        with ctx:
            self.checkpoint_manager = self._init_checkpoint_manager()
            self._fit()

        if self.checkpoint_manager is not None:
            self.checkpoint_manager.close()
            
    def _fit(self) -> None:
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"▸ Training on {self.device}  |  Parameters: {n_params:,}")
        self._log({"model/parameters": n_params})
    
        # -- Create reusable progress bars ----------------------------
        train_pbar = tqdm(
            total=len(self.loaders["train"]),
            desc="[train]",
            leave=True,
            dynamic_ncols=True,
        )
        val_pbar = tqdm(
            total=len(self.loaders["val"]),
            desc="[val]",
            leave=True,
            dynamic_ncols=True,
        )
    
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
    
            # -- train --------------------------------------------
            train_loss = self._run_train_epoch(train_pbar)
            self._log(
                {"train/epoch_loss": train_loss, "epoch": epoch},
                step=self.global_step,
            )
    
            # -- validate -----------------------------------------
            val_loss = None
            if (epoch + 1) % self.val_every_n_epochs == 0:
                val_loss = self._run_eval_epoch("val", val_pbar)
                self._log(
                    {"val/epoch_loss": val_loss, "epoch": epoch},
                    step=self.global_step,
                )
    
            # -- checkpoint ---------------------------------------
            self.save_checkpoint(val_loss)
    
            # -- scheduler ----------------------------------------
            self._step_scheduler(val_loss)
    
            # -- epoch summary ------------------------------------
            lr = self.optimizer.param_groups[0]["lr"]
            self._log({'lr': lr}, step=self.global_step)
    
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
