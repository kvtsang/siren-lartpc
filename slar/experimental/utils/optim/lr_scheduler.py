import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt


class CosineAnnealingWithLinearWarmup(_LRScheduler):
    """
    Cosine Annealing learning rate scheduler with linear warmup.
    
    During warmup phase (steps 0 to warmup_steps):
        lr = base_lr * (current_step / warmup_steps)
    
    During cosine annealing phase (steps warmup_steps to total_steps):
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress))
        where progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
    
    Args:
        optimizer (torch.optim.Optimizer): Wrapped optimizer.
        warmup_steps (int): Number of steps for linear warmup.
        total_steps (int): Total number of training steps (warmup + cosine annealing).
        min_lr (float): Minimum learning rate at the end of cosine annealing. Default: 0.0.
        last_epoch (int): The index of the last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if total_steps < warmup_steps:
            raise ValueError(
                f"total_steps ({total_steps}) must be >= warmup_steps ({warmup_steps})"
            )

        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

        # Call parent init last — it triggers get_lr() internally
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate for each parameter group."""
        step = self.last_epoch  # _LRScheduler increments last_epoch on each step

        if step < self.warmup_steps:
            # Linear warmup: scale from 0 to base_lr
            if self.warmup_steps == 0:
                warmup_factor = 1.0
            else:
                warmup_factor = step / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            cosine_steps = self.total_steps - self.warmup_steps
            if cosine_steps == 0:
                return [self.min_lr for _ in self.base_lrs]

            progress = (step - self.warmup_steps) / cosine_steps
            progress = min(progress, 1.0)  # Clamp to avoid issues past total_steps

            return [
                self.min_lr
                + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]
