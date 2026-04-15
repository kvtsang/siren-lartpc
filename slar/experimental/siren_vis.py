"""SirenVis – A visibility‐aware wrapper around the base Siren network.

Key design decisions
--------------------
* ``SirenVis(x) = output_scale * xfmr(Siren(meta.norm_coord(x)))``
  for in‐bound coordinates; out‐of‐bound coordinates produce zeros.
* Input coordinates are normalised via ``meta`` (:class:`AABox`) which also
  provides containment checks.
* ``xfmr`` / ``xfmr_inv`` are transformations on the **SIREN output**
  (not the input coordinates).
* ``output_scale`` is either a registered **buffer** (default) or a learnable
  ``nn.Parameter``, controlled by ``learnable_output_scale``.
* Each component (``AABox``, transforms) owns its own ``hparams`` —
  ``SirenVis`` collects them at checkpoint time and delegates reconstruction
  back to each component.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import yaml

from slar.base import Siren
from photonlib.experimental import AABox
from photonlib.experimental.transform import Identity, create_output_transform


class SirenVis(nn.Module):
    """Siren network extended with AABox coordinate normalisation, an optional
    output transformation, and an optional output scale for visibility
    prediction in LArTPC detectors.

    The forward pass is::

        mask   = meta.contain(x)
        x_norm = meta.norm_coord(x[mask])
        output[mask] = output_scale * xfmr(Siren(x_norm))
        output[~mask] = 0

    Parameters
    ----------
    in_features : int
        Dimensionality of the input coordinates.
    hidden_features : int
        Width of every hidden layer.
    hidden_layers : int
        Number of hidden layers.
    out_features : int
        Dimensionality of the network output.
    meta : AABox
        Axis‐aligned bounding box used to normalise input coordinates
        and check containment.
    outermost_linear : bool, optional
        If *True* the last layer is a plain linear layer (no sine
        activation).  Default ``True``.
    first_omega_0 : float, optional
        Frequency multiplier for the first layer.  Default ``30.0``.
    hidden_omega_0 : float, optional
        Frequency multiplier for hidden layers.  Default ``30.0``.
    output_scale : array_like or None, optional
        Per‐output multiplicative scale applied **after** the output
        transform.  Shape must be broadcast‐compatible with
        ``(out_features,)``.  When *None*, no scaling is applied
        (equivalent to all‐ones).
    learnable_output_scale : bool, optional
        If *True* ``output_scale`` is stored as a learnable
        ``nn.Parameter``; otherwise as a non‐learnable buffer.
        Default ``False``.
    xfmr : nn.Module or None, optional
        Transformation applied to the **Siren output** before
        multiplication by ``output_scale``.  Default ``None`` (identity).
    xfmr_inv : nn.Module or None, optional
        Inverse of ``xfmr``.  Stored for convenience but never called
        inside :meth:`forward`.  Default ``None`` (identity).
    """

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        meta: AABox,
        outermost_linear: bool = True,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
        output_scale: Optional[Any] = None,
        learnable_output_scale: bool = False,
        xfmr: Optional[nn.Module] = None,
        xfmr_inv: Optional[nn.Module] = None,
    ):
        super().__init__()

        # ---- AABox for coordinate normalisation & containment ---- #
        self._meta = meta

        # ---- core Siren network ---- #
        self.siren = Siren(
            in_features=in_features,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=out_features,
            outermost_linear=outermost_linear,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
        )

        # ---- output scale ---- #
        self.learnable_output_scale = learnable_output_scale
        self._init_output_scale(output_scale, out_features, learnable_output_scale)

        # ---- output transforms ---- #
        self.xfmr: nn.Module = xfmr if xfmr is not None else Identity()
        self.xfmr_inv: nn.Module = xfmr_inv if xfmr_inv is not None else Identity()

    # ------------------------------------------------------------------ #
    #  Properties
    # ------------------------------------------------------------------ #
    @property
    def meta(self) -> AABox:
        """Axis‐aligned bounding box used for coordinate normalisation."""
        return self._meta

    @property
    def hparams(self) -> Dict[str, Any]:
        """Return a serialisable dict sufficient to fully reconstruct the
        model architecture (not the weights).

        Each sub‐component contributes its own ``hparams``.
        """
        return dict(
            in_features=self.siren.net[0].linear.in_features,
            hidden_features=self.siren.net[0].linear.out_features,
            hidden_layers=len(self.siren.net) - 2,
            out_features=self.siren.net[-1].linear.out_features
            if hasattr(self.siren.net[-1], "linear")
            else self.siren.net[-1].out_features,
            outermost_linear=not hasattr(self.siren.net[-1], "omega_0"),
            first_omega_0=self.siren.net[0].omega_0,
            hidden_omega_0=self.siren.net[1].omega_0
            if len(self.siren.net) > 2
            else 30.0,
            learnable_output_scale=self.learnable_output_scale,
            meta=self._meta.hparams,
            output_transform=self.xfmr.hparams,
        )

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #
    def _init_output_scale(
        self,
        output_scale: Optional[Any],
        out_features: int,
        learnable: bool,
    ) -> None:
        """Register ``output_scale`` as a buffer or parameter."""
        if output_scale is not None:
            if isinstance(output_scale, np.ndarray):
                tensor = torch.from_numpy(output_scale).float()
            elif isinstance(output_scale, torch.Tensor):
                tensor = output_scale.float()
            else:
                tensor = torch.tensor(output_scale, dtype=torch.float32)
            tensor = tensor.reshape(-1)
        else:
            tensor = torch.ones(out_features, dtype=torch.float32)

        if learnable:
            self.output_scale = nn.Parameter(tensor)
        else:
            self.register_buffer("output_scale", tensor)

    # ------------------------------------------------------------------ #
    #  Forward
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the model.  Out‐of‐bound points are zero‐filled.

        Parameters
        ----------
        x : torch.Tensor
            Input coordinates of shape ``(N, in_features)`` in physical
            (un‐normalised) space.

        Returns
        -------
        torch.Tensor
            Network output of shape ``(N, out_features)``.
        """
        mask = self._meta.contain(x)

        out_shape = x.shape[:-1] + (self.output_scale.shape[-1],)
        output = torch.zeros(out_shape, dtype=x.dtype, device=x.device)

        if not mask.any():
            return output

        x_in = x[mask]
        x_norm = self._meta.norm_coord(x_in)
        siren_out = self.siren(x_norm)
        transformed = self.xfmr(siren_out)
        scaled = self.output_scale * transformed

        output[mask] = scaled
        return output

    # ------------------------------------------------------------------ #
    #  Checkpoint save / load
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, path: Union[str, Path], **extra) -> None:
        """Save model to a self‐contained checkpoint.

        The checkpoint dict contains:

        * ``hparams``    – architecture + sub‐component hparams
        * ``state_dict`` – ``self.state_dict()``
        * any additional keyword arguments supplied by the caller

        Parameters
        ----------
        path : str or Path
            Destination file path.
        **extra
            Arbitrary additional entries (e.g. ``epoch``, ``loss``).
        """
        ckpt: Dict[str, Any] = dict(
            hparams=self.hparams,
            state_dict=self.state_dict(),
        )
        ckpt.update(extra)
        torch.save(ckpt, path)

    @classmethod
    def from_checkpoint(
        cls,
        path: Union[str, Path],
        map_location: Optional[Any] = None,
    ) -> "SirenVis":
        """Construct a ``SirenVis`` directly from a checkpoint file.

        No YAML / dict config is required — the checkpoint contains
        everything needed.

        Parameters
        ----------
        path : str or Path
            Path to the checkpoint file saved with :meth:`save_checkpoint`.
        map_location : optional
            Passed through to :func:`torch.load`.

        Returns
        -------
        SirenVis
            Fully initialised model with weights loaded.
        """
        ckpt = torch.load(path, map_location=map_location)
        hparams = ckpt["hparams"]

        meta = AABox.from_hparams(hparams["meta"])

        xfmr, xfmr_inv = create_output_transform(
            hparams.get("output_transform", None)
        )

        model = cls(
            in_features=hparams["in_features"],
            hidden_features=hparams["hidden_features"],
            hidden_layers=hparams["hidden_layers"],
            out_features=hparams["out_features"],
            meta=meta,
            outermost_linear=hparams.get("outermost_linear", True),
            first_omega_0=hparams.get("first_omega_0", 30.0),
            hidden_omega_0=hparams.get("hidden_omega_0", 30.0),
            learnable_output_scale=hparams.get("learnable_output_scale", False),
            xfmr=xfmr,
            xfmr_inv=xfmr_inv,
            # output_scale restored from state_dict
        )

        model.load_state_dict(ckpt["state_dict"])
        return model

    # ------------------------------------------------------------------ #
    #  Class factory – from dict / YAML
    # ------------------------------------------------------------------ #
    @classmethod
    def create(
        cls,
        cfg: Union[str, Path, Dict[str, Any]],
        output_scale: Optional[Any] = None,
        output_scale_path: Optional[Union[str, Path]] = None,
    ) -> "SirenVis":
        """Class factory that builds a ``SirenVis`` from a config.

        Parameters
        ----------
        cfg : str, Path, or dict
            If a *str* or *Path* it is interpreted as the path to a YAML
            file.  If a *dict* it is used directly.

            Recognised keys:

            * **in_features** *(int, required)*
            * **hidden_features** *(int, required)*
            * **hidden_layers** *(int, required)*
            * **out_features** *(int, required)*
            * **outermost_linear** *(bool)* – default ``True``
            * **first_omega_0** *(float)* – default ``30.0``
            * **hidden_omega_0** *(float)* – default ``30.0``
            * **learnable_output_scale** *(bool)* – default ``False``
            * **output_scale_path** *(str)* – ``.npy`` file
            * **meta** *(dict)* – forwarded to ``AABox.from_hparams``

              - ``{"ranges": [[lo, hi], ...]}``  or
              - ``{"h5_file": "<path>"}``

            * **output_transform** *(dict or None)* – forwarded to
              :func:`create_output_transform`

              - ``{"class": "slar.transform.LogTransform", "eps": 1e-10}``

        output_scale : array_like or None, optional
            Per‐output scale (highest priority).
        output_scale_path : str, Path or None, optional
            Path to a ``.npy`` file.  Overrides in‐config value.

        Returns
        -------
        SirenVis

        Examples
        --------
        **YAML** (``config.yaml``)::

            in_features: 3
            hidden_features: 256
            hidden_layers: 5
            out_features: 180
            first_omega_0: 30.0
            hidden_omega_0: 30.0
            outermost_linear: true
            learnable_output_scale: false
            output_scale_path: "visibility_scale.npy"
            meta:
              ranges:
                - [-200.0, 200.0]
                - [-200.0, 200.0]
                - [0.0, 500.0]
            output_transform:
              class: "slar.transform.LogTransform"
              eps: 1.0e-10
              base: 10.0

        **Python**::

            model = SirenVis.create("config.yaml")
        """
        # ---- load YAML if necessary ---- #
        if isinstance(cfg, (str, Path)):
            with open(cfg, "r") as fh:
                cfg = yaml.safe_load(fh)
        cfg = dict(cfg)  # shallow copy

        # ---- resolve output_scale ---- #
        if output_scale is None:
            output_scale = cfg.pop("output_scale", None)
        else:
            cfg.pop("output_scale", None)

        if output_scale is None:
            scale_path = output_scale_path or cfg.pop("output_scale_path", None)
            if scale_path is not None:
                output_scale = np.load(scale_path)
        else:
            cfg.pop("output_scale_path", None)

        cfg.pop("output_scale_path", None)

        # ---- resolve meta (AABox) ---- #
        meta_cfg = cfg.pop("meta", None)
        if meta_cfg is None:
            raise ValueError(
                "Configuration must contain a 'meta' section with either "
                "'h5_file' or 'ranges' to construct an AABox."
            )
        meta = AABox.from_hparams(meta_cfg)

        # ---- resolve output transforms ---- #
        output_transform_cfg = cfg.pop("output_transform", None)
        xfmr, xfmr_inv = create_output_transform(output_transform_cfg)

        # ---- build model ---- #
        return cls(
            meta=meta,
            output_scale=output_scale,
            xfmr=xfmr,
            xfmr_inv=xfmr_inv,
            **cfg,
        )

    # ------------------------------------------------------------------ #
    #  Pretty repr
    # ------------------------------------------------------------------ #
    def extra_repr(self) -> str:
        parts = [
            f"learnable_output_scale={self.learnable_output_scale}",
            f"xfmr={self.xfmr.__class__.__name__}",
            f"xfmr_inv={self.xfmr_inv.__class__.__name__}",
            f"meta={self._meta}",
        ]
        return ", ".join(parts)
