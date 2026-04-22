import os
import datetime
import random
import string

from .manager import CheckpointManager


def create_checkpoint_manager(
    cfg: dict,
    project: str = 'no_name',
    name: str = None,
):
    """
    Factory function to create a CheckpointManager.

    Creates and returns an initialized CheckpointManager
    instance. If no name identifier is provided, one is
    generated using the current date-time and a random
    alphanumeric suffix. The checkpoint directory in the
    configuration is updated by appending the project
    and name identifier.

    Parameters
    ----------
    cfg : dict
        Keyword arguments to be passed to the
        ``CheckpointManager`` constructor. May contain
        the key ``'dir'`` specifying the base
        directory for storing checkpoints. If not
        provided, defaults to the current working
        directory. The ``'dir'`` value will be
        modified to append ``{project}/{name}`` before
        being passed to the constructor.
    project : str, optional
        Name of the project. Used solely to construct
        the checkpoint directory path. Not passed to the
        ``CheckpointManager`` constructor.
        Defaults to ``'no_name'``.
    name : str, optional
        Identifier for the specific run. Used solely to
        construct the checkpoint directory path. Not
        passed to the ``CheckpointManager`` constructor.
        If ``None``, a unique identifier is generated
        using the current date-time and a random
        8-character alphanumeric string.
        Defaults to ``None``.

    Returns
    -------
    CheckpointManager
        An initialized instance with the checkpoint
        directory path set to
        ``{dir}/{project}/{name}``.

    Examples
    --------
    >>> cfg = {
    ...     'dir': '/data/checkpoints',
    ...     'save_best': True,
    ...     'max_to_keep': 5,
    ... }
    >>> manager = create_checkpoint_manager(
    ...     cfg,
    ...     project='my_project',
    ...     name='run_01',
    ... )
    >>> print(manager.dir)
    /data/checkpoints/my_project/run_01
    """
    cfg = cfg.copy()
    cfg.setdefault('dir', os.getcwd())

    if name is None:
        timestamp = datetime.datetime.now().strftime(
            '%Y%m%d_%H%M%S'
        )
        random_suffix = ''.join(
            random.choices(
                string.ascii_lowercase + string.digits,
                k=8
            )
        )
        name = f"{timestamp}_{random_suffix}"

    cfg['dir'] = os.path.join(
        cfg['dir'],
        project,
        name 
    )

    return CheckpointManager(**cfg)
