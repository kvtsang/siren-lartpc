"""
Transforms with hparams support for SirenVis."""

from __future__ import annotations

import math
from typing import Union

import torch
import torch.nn as nn

from .utils.factory import create_instance


class Identity(nn.Module):
    """nn.Module identity – a no‐op transform.

    Implements the ``hparams`` / ``from_hparams`` protocol so that
    :class:`SirenVis` can serialise and reconstruct it uniformly
    alongside real transforms.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def inverse(self) -> "Identity":
        """Return the inverse transform (also identity)."""
        return Identity()

    @property
    def hparams(self) -> dict:
        return { "class": _get_class_path(self) }

    @classmethod
    def from_hparams(cls, hparams: dict) -> "Identity":
        return cls()


class LogTransform(nn.Module):
    r"""Element‐wise logarithmic transform: ``y = log(x + eps)``.

    The natural logarithm is used by default.  An arbitrary base can be
    selected, in which case the transform becomes::

        y = log_b(x + eps) = ln(x + eps) / ln(b)

    Parameters
    ----------
    eps : float, optional
        Small positive constant added before taking the log to avoid
        ``log(0)``.  Default ``1e-10``.
    base : float or None, optional
        Logarithm base.  ``None`` (default) means natural log (base *e*).
    """

    def __init__(
        self,
        eps: float = 1e-10,
        base: Union[float, None] = None,
    ):
        super().__init__()
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        if base is not None and base <= 0:
            raise ValueError(f"base must be positive, got {base}")
        if base is not None and base == 1.0:
            raise ValueError("base must not be 1")

        self.eps = eps
        self.base = base
        self._log_base = math.log(base) if base is not None else 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ``log_b(x + eps)``."""
        return torch.log(x + self.eps) / self._log_base

    def inverse(self) -> "ExpTransform":
        """Return the corresponding inverse transform."""
        return ExpTransform(eps=self.eps, base=self.base)

    @property
    def hparams(self) -> dict:
        """Serialisable dict for reconstruction.

        Returns
        -------
        dict
            ``{"class": "...", "eps": ..., "base": ...}``
        """
        return {
            "class": _get_class_path(self),
            "eps": self.eps,
            "base": self.base,
        }

    @classmethod
    def from_hparams(cls, hparams: dict) -> "LogTransform":
        """Reconstruct from a ``hparams`` dict.

        Parameters
        ----------
        hparams : dict
            Must contain ``"eps"``; ``"base"`` is optional.
        """
        return cls(
            eps=hparams.get("eps", 1e-10),
            base=hparams.get("base", None),
        )

    def extra_repr(self) -> str:
        base_str = self.base if self.base is not None else "e"
        return f"eps={self.eps}, base={base_str}"


class ExpTransform(nn.Module):
    r"""Element‐wise inverse of :class:`LogTransform`: ``x = b^y - eps``.

    Undoes ``y = log_b(x + eps)`` so that::

        ExpTransform(LogTransform(x)) ≈ x

    Parameters
    ----------
    eps : float, optional
        The same ``eps`` used in the corresponding :class:`LogTransform`.
        Default ``1e-10``.
    base : float or None, optional
        Logarithm base.  ``None`` (default) means natural log (base *e*),
        so the inverse is ``exp(y) - eps``.
    """

    def __init__(
        self,
        eps: float = 1e-10,
        base: Union[float, None] = None,
    ):
        super().__init__()
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        if base is not None and base <= 0:
            raise ValueError(f"base must be positive, got {base}")
        if base is not None and base == 1.0:
            raise ValueError("base must not be 1")

        self.eps = eps
        self.base = base
        self._log_base = math.log(base) if base is not None else 1.0

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Compute ``b^y - eps``."""
        return torch.exp(y * self._log_base) - self.eps

    def inverse(self) -> "LogTransform":
        """Return the corresponding forward :class:`LogTransform`."""
        return LogTransform(eps=self.eps, base=self.base)

    @property
    def hparams(self) -> dict:
        """Serialisable dict for reconstruction.

        Returns
        -------
        dict
            ``{"class": "...", "eps": ..., "base": ...}``
        """
        return {
            "class": _get_class_path(self),
            "eps": self.eps,
            "base": self.base,
        }

    @classmethod
    def from_hparams(cls, hparams: dict) -> "ExpTransform":
        """Reconstruct from a ``hparams`` dict.

        Parameters
        ----------
        hparams : dict
            Must contain ``"eps"``; ``"base"`` is optional.
        """
        return cls(
            eps=hparams.get("eps", 1e-10),
            base=hparams.get("base", None),
        )

    def extra_repr(self) -> str:
        base_str = self.base if self.base is not None else "e"
        return f"eps={self.eps}, base={base_str}"


# ---------------------------------------------------------------------------
# Output‐transform factory
# ---------------------------------------------------------------------------
def create_output_transform(
    cfg: Union[Dict[str, Any], None] = None,
) -> Tuple[nn.Module, nn.Module]:
    """Factory that builds a forward output transform and its inverse.

    Parameters
    ----------
    cfg : dict or None
        Configuration dict with the structure::

            {
                "class": "slar.transform.LogTransform",
                "eps": 1e-10,
                "base": 10.0
            }

        ``"class"`` is required.  All other keys are forwarded as keyword
        arguments to the constructor via :func:`create_instance`.

        If ``None``, returns a pair of :class:`Identity` transforms.

    Returns
    -------
    xfmr : nn.Module
        Forward output transformation.
    xfmr_inv : nn.Module
        Inverse output transformation.

    Raises
    ------
    TypeError
        If the constructed object does not support any of the recognised
        conventions for obtaining an inverse.

    Notes
    -----
    The resolved class must follow **one** of these conventions
    (checked in order):

    1. **``nn.Module`` with ``.inverse()``** — the instance *is* the
       forward transform and ``.inverse()`` returns the inverse module.
    2. **Attribute‐based** — the instance exposes ``.forward_transform``
       and ``.inverse_transform`` (both ``nn.Module``).
    3. **Tuple return** — the constructor directly returns a
       ``(xfmr, xfmr_inv)`` tuple.

    Examples
    --------
    ::

        # With config
        cfg = {"class": "slar.transform.LogTransform", "eps": 1e-10}
        xfmr, xfmr_inv = create_output_transform(cfg)

        # Without config — returns Identity pair
        xfmr, xfmr_inv = create_output_transform()

        # Round-trip through hparams
        xfmr2, _ = create_output_transform(xfmr.hparams)
    """
    if cfg is None:
        return Identity(), Identity()

    obj = create_instance(cfg)

    # Convention 1: nn.Module with .inverse()
    if isinstance(obj, nn.Module) and callable(getattr(obj, "inverse", None)):
        return obj, obj.inverse()

    # Convention 2: attribute-based
    if hasattr(obj, "forward_transform") and hasattr(obj, "inverse_transform"):
        return obj.forward_transform, obj.inverse_transform

    # Convention 3: tuple return
    if isinstance(obj, tuple) and len(obj) == 2:
        return obj[0], obj[1]

    cls_name = cfg.get("class", cfg.get("class_path", "unknown"))
    raise TypeError(
        f"Cannot extract (xfmr, xfmr_inv) from {cls_name!r}. "
        "The class should either: "
        "(1) be an nn.Module with an .inverse() method, "
        "(2) expose .forward_transform and .inverse_transform attributes, or "
        "(3) return a (xfmr, xfmr_inv) tuple from its constructor."
    )


# ------------------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------------------

def _get_class_path(obj):
    return f"{obj.__class__.__module__}.{self.__class__.__qualname__}"
