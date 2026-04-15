import importlib
from typing import Any

def create_instance(config: dict = None, **kwargs) -> Any:
    """
    Factory function that creates a class instance from a
    configuration dict or keyword arguments.

    The 'class' key (or 'class_path') should be a dotted string
    like 'module.submodule.ClassName'. All remaining keys are
    passed as keyword arguments to the class constructor.

    Args:
        config:   A dictionary containing 'class' and
                  constructor arguments.
        **kwargs: Alternatively, provide 'class' and
                  constructor args as keyword arguments.

    Returns:
        An instance of the specified class.

    Raises:
        ValueError:      If no class path is provided.
        ImportError:      If the module cannot be imported.
        AttributeError:  If the class is not found in the
                         module.

    Examples:
        >>> obj = create_instance(
        ...     {"class": "collections.OrderedDict"}
        ... )

        >>> obj = create_instance(
        ...     class_path="datetime.datetime",
        ...     year=2024, month=1, day=1,
        ... )
    """
    # -------------------------------------------------------- #
    # 1. Merge config dict and keyword arguments
    # -------------------------------------------------------- #
    params: dict = {}
    if config is not None:
        params.update(config)
    params.update(kwargs)

    # -------------------------------------------------------- #
    # 2. Extract the class path string
    # -------------------------------------------------------- #
    class_path: str | None = (
        params.pop("class", None)
        or params.pop("class_path", None)
    )

    if not class_path:
        raise ValueError(
            "A class path must be provided via the 'class' "
            "or 'class_path' key. "
            "Example: 'collections.OrderedDict'"
        )

    # -------------------------------------------------------- #
    # 3. Resolve the class from the dotted path
    # -------------------------------------------------------- #
    cls = _import_class(class_path)

    # -------------------------------------------------------- #
    # 4. Separate positional args from keyword args
    # -------------------------------------------------------- #
    positional_args = params.pop("args", [])

    # -------------------------------------------------------- #
    # 5. Instantiate and return
    # -------------------------------------------------------- #
    return cls(*positional_args, **params)


def _import_class(dotted_path: str) -> type:
    """
    Import a class from a fully-qualified dotted path string.

    Supports paths like:
        - 'collections.OrderedDict'
        - 'package.module.ClassName'
        - 'package.module.ClassName.NestedClass'

    It walks the dotted path from right to left, trying to
    split into (module, attribute_chain) until a valid module
    is found.
    """
    parts = dotted_path.rsplit(".", 1)

    if len(parts) == 1:
        import builtins
        if hasattr(builtins, parts[0]):
            return getattr(builtins, parts[0])
        raise ImportError(
            f"Cannot import '{dotted_path}' "
            f"— no module path provided."
        )

    module_path, attr_chain = parts[0], parts[1]

    segments = module_path.split(".")
    obj = None

    for i in range(len(segments), 0, -1):
        candidate_module = ".".join(segments[:i])
        remaining_attrs = segments[i:] + [attr_chain]
        try:
            obj = importlib.import_module(candidate_module)
            for attr in remaining_attrs:
                obj = getattr(obj, attr)
            return obj
        except (ImportError, AttributeError):
            continue

    raise ImportError(
        f"Could not import class '{dotted_path}'. "
        f"Verify that the module is installed and the "
        f"class name is correct."
    )
