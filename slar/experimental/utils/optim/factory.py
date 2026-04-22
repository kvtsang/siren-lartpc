import torch.nn as nn

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from ..helper import create_instance

def create_optimizer(
    model: nn.Module,
    config: dict = None,
    **kwargs,
) -> Optimizer:
    """
    Factory function that creates a PyTorch optimizer for
    a given model.

    Args:
        model:    A torch.nn.Module whose parameters will
                  be optimized.
        config:   A dictionary containing 'class' (the
                  optimizer dotted path) and any optimizer
                  hyperparameters (lr, weight_decay, etc.).
        **kwargs: Alternatively, provide the optimizer class
                  and hyperparameters as keyword arguments.

    Returns:
        An instance of the specified PyTorch optimizer.

    Raises:
        ValueError: If no class path is provided.
        TypeError:  If the created instance is not a
                    torch.optim.Optimizer.

    Examples:
        >>> import torch.nn as nn
        >>> model = nn.Linear(10, 2)

        >>> optimizer = create_optimizer(model, {
        ...     "class": "torch.optim.Adam",
        ...     "lr": 1e-3,
        ...     "weight_decay": 1e-5,
        ... })

        >>> optimizer = create_optimizer(
        ...     model,
        ...     class_path="torch.optim.SGD",
        ...     lr=0.01,
        ...     momentum=0.9,
        ... )
    """
    # ---------------------------------------------------- #
    # 1. Merge config dict and keyword arguments
    # ---------------------------------------------------- #
    params: dict = {}
    if config is not None:
        params.update(config)
    params.update(kwargs)

    # ---------------------------------------------------- #
    # 2. Inject model parameters as the first argument
    # ---------------------------------------------------- #
    params["args"] = [model.parameters()]

    # ---------------------------------------------------- #
    # 3. Default to 'torch.optim.Adam' if no class given
    # ---------------------------------------------------- #
    if "class" not in params and "class_path" not in params:
        params["class"] = "torch.optim.Adam"

    # ---------------------------------------------------- #
    # 4. Create the optimizer via the generic factory
    # ---------------------------------------------------- #
    optimizer = create_instance(params)

    # ---------------------------------------------------- #
    # 5. Validate the result
    # ---------------------------------------------------- #
    if not isinstance(optimizer, Optimizer):
        raise TypeError(
            f"Expected a torch.optim.Optimizer instance, "
            f"got {type(optimizer).__name__}."
        )

    return optimizer

def create_scheduler(
    optimizer: Optimizer,
    config: dict = None,
    **kwargs,
) -> LRScheduler:
    """
    Factory function that creates a PyTorch learning rate
    scheduler for a given optimizer.

    Args:
        optimizer: A torch.optim.Optimizer instance.
        config:    A dictionary containing 'class' (the
                   scheduler dotted path) and any scheduler
                   parameters (step_size, gamma, etc.).
        **kwargs:  Alternatively, provide the scheduler
                   class and parameters as keyword
                   arguments.

    Returns:
        An instance of the specified PyTorch scheduler.

    Raises:
        ValueError: If no class path is provided.
        TypeError:  If the created instance is not a
                    LRScheduler.
    """
    # ---------------------------------------------------- #
    # 1. Merge config dict and keyword arguments
    # ---------------------------------------------------- #
    params: dict = {}
    if config is not None:
        params.update(config)
    params.update(kwargs)

    # ---------------------------------------------------- #
    # 2. Inject optimizer as the first argument
    # ---------------------------------------------------- #
    params["args"] = [optimizer]

    # ---------------------------------------------------- #
    # 3. Default to CosineAnnealingLR if no class given
    # ---------------------------------------------------- #
    if "class" not in params and "class_path" not in params:
        params["class"] = (
            "torch.optim.lr_scheduler.CosineAnnealingLR"
        )

    # ---------------------------------------------------- #
    # 4. Create the scheduler via the generic factory
    # ---------------------------------------------------- #
    scheduler = create_instance(params)

    # ---------------------------------------------------- #
    # 5. Validate the result
    # ---------------------------------------------------- #
    if not isinstance(scheduler, LRScheduler):
        raise TypeError(
            f"Expected a LRScheduler instance, "
            f"got {type(scheduler).__name__}."
        )

    return scheduler
