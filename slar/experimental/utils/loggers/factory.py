"""
Logger interfaces and factory.
"""

class Logger:
    """Example logger with context manager."""
    def __init__(self, module, **kwargs):
        self._module = module
        self._kwargs = kwargs

    def init(self):
        return self  # self is the context manager

    def __enter__(self):
        self._module.init(**self._kwargs)
        return self

    def __exit__(self, *exc_info):
        self._module.finish()

    def log(self, *args, **kwargs):
        self._module.log(*args, **kwargs)

class NullLogger:
    """
    No-op logger used when user doesn't provide one.
    Satisfies the same interface so no if-checks are needed.
    """

    def init(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        pass

    def log(self, *args, **kwargs):
        pass

def create_logger(cfg):
    if cfg is None:
        return NullLogger()

    cfg = cfg.copy()
    if cfg.pop('use_wandb', False):
        import wandb
        return Logger(wandb, **cfg)

    from . import csv_logger
    return Logger(csv_logger, **cfg)
