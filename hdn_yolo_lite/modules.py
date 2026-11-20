"""LMSCSP-DeepOnly modules for HDN-YOLO-Lite.

Paper reference: Section 3.2, Eqs. (6)-(12).

The implementation contains three pieces:

* :class:`LMSConv`       -- Lightweight Multi-Scale Convolution (Eqs. 6-8).
* :class:`LMSBottleneck` -- 1x1 channel adjustment + LMSConv, optional residual.
* :class:`C3k2LMS`       -- C2f/CSP-style block whose bottlenecks are LMSBottlenecks
                            (Eqs. 9-12).

The block is applied only to the deep P5 stages in
``configs/hdn_yolo_lite_yolo11m.yaml`` (backbone layer 8 and neck layer 22).
The shallow P3/P4 paths and the detection head are left unchanged.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from ultralytics.nn.modules import C2f, Conv

__all__ = [
    "LMSConv",
    "LMSBottleneck",
    "C3k2LMS",
    "install_model_builder",
]


class LMSConv(nn.Module):
    """Lightweight Multi-Scale Convolution with a half-channel bypass.

    The input channels are split into a main branch and a bypass branch.
    The main branch is further split into 3x3 and 5x5 convolution branches.
    The bypass is transferred by identity (stride 1) or average pooling
    (stride > 1). All branches are concatenated and fused by a 1x1 conv.
    """

    def __init__(self, c1: int, c2: int, s: int = 1, act: bool = True):
        super().__init__()
        self.main_ch = max(c1 // 2, 1)
        self.skip_ch = c1 - self.main_ch
        self.k3_ch = max(self.main_ch // 2, 1)
        self.k5_ch = self.main_ch - self.k3_ch

        self.cv3 = Conv(self.k3_ch, self.k3_ch, 3, s, act=act)
        self.cv5 = Conv(self.k5_ch, self.k5_ch, 5, s, act=act) if self.k5_ch > 0 else None
        self.skip = nn.AvgPool2d(kernel_size=s, stride=s) if s > 1 and self.skip_ch > 0 else nn.Identity()
        self.fuse = Conv(c1, c2, 1, 1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_main, x_skip = torch.split(x, [self.main_ch, self.skip_ch], dim=1)
        x3, x5 = torch.split(x_main, [self.k3_ch, self.k5_ch], dim=1)

        y = [self.cv3(x3)]
        if self.k5_ch > 0:
            y.append(self.cv5(x5))
        if self.skip_ch > 0:
            y.append(self.skip(x_skip))
        return self.fuse(torch.cat(y, dim=1))


class LMSBottleneck(nn.Module):
    """Lightweight bottleneck: 1x1 channel adjustment followed by LMSConv."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 1.0):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = LMSConv(c_, c2)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3k2LMS(C2f):
    """C2f/CSP-style block that replaces standard bottlenecks with LMSBottlenecks.

    The constructor mirrors the YOLO11 ``C3k2`` signature so that the block can
    be described in a model YAML. ``c3k`` and ``attn`` are accepted for
    signature compatibility but are intentionally unused: the paper instantiates
    one LMSBottleneck per replaced deep-stage block.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(LMSBottleneck(self.c, self.c, shortcut, e=1.0) for _ in range(n))


_ORIGINAL_PARSE_MODEL = None


def _c3k2_to_lms(block: nn.Module) -> C3k2LMS:
    """Create a C3k2LMS block with the same external channels/repeats as ``block``."""
    c1 = block.cv1.conv.in_channels
    c2 = block.cv2.conv.out_channels
    n = len(block.m)
    e = block.c / c2
    new = C3k2LMS(c1, c2, n, e=e)

    # Copy the bookkeeping attributes set by ultralytics.nn.tasks.parse_model.
    for attr in ("i", "f"):
        if hasattr(block, attr):
            setattr(new, attr, getattr(block, attr))
    new.type = str(C3k2LMS)[8:-2].replace("__main__.", "")
    new.np = sum(p.numel() for p in new.parameters())
    return new


def _hdn_parse_model(d, ch, verbose: bool = True):
    """Parse a model dict and swap every ``C3k2LMS`` entry to an LMS block.

    Ultralytics' parser only knows its built-in module classes when deciding
    channel widths. To avoid forking the framework, the YAML entry is parsed as
    a standard ``C3k2`` (which has the same external channel geometry) and the
    resulting block is replaced by ``C3k2LMS`` immediately afterwards.
    """
    layers = d.get("backbone", []) + d.get("head", [])
    lms_indices = [i for i, layer in enumerate(layers) if layer[2] == "C3k2LMS"]
    if not lms_indices:
        return _ORIGINAL_PARSE_MODEL(d, ch, verbose)

    d = copy.deepcopy(d)
    n_backbone = len(d["backbone"])
    for i in lms_indices:
        layer = d["backbone"][i] if i < n_backbone else d["head"][i - n_backbone]
        layer[2] = "C3k2"

    model, save = _ORIGINAL_PARSE_MODEL(d, ch, verbose)
    for i in lms_indices:
        model[i] = _c3k2_to_lms(model[i])
    return model, save


def install_model_builder() -> None:
    """Enable ``C3k2LMS`` entries in Ultralytics model YAML files.

    The patch is idempotent and is applied at runtime, so no file inside the
    installed ``ultralytics`` package needs to be modified. Call this before
    constructing a model from ``configs/hdn_yolo_lite_yolo11m.yaml``.
    """
    global _ORIGINAL_PARSE_MODEL
    if _ORIGINAL_PARSE_MODEL is not None:
        return

    from ultralytics.nn import tasks

    _ORIGINAL_PARSE_MODEL = tasks.parse_model
    tasks.parse_model = _hdn_parse_model

    # Allow checkpoints containing these classes to be loaded with
    # torch.load(..., weights_only=True) on newer PyTorch versions.
    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is not None:
        add_safe_globals([LMSConv, LMSBottleneck, C3k2LMS])
