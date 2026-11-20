"""Reference implementation of HDN-YOLO-Lite.

HDN-YOLO-Lite combines three components on top of YOLO11m:

* LMSCSP-DeepOnly deep-stage lightweight refinement (Section 3.2).
* Hard-distribution-targeted training (Section 3.3).
* NWD-based bounding-box regression (Section 3.4).

See ``README.md`` for usage.
"""

from .loss import NWDBboxLoss, install_nwd_loss, normalized_wasserstein
from .modules import C3k2LMS, LMSBottleneck, LMSConv, install_model_builder

__all__ = [
    "LMSConv",
    "LMSBottleneck",
    "C3k2LMS",
    "install_model_builder",
    "normalized_wasserstein",
    "NWDBboxLoss",
    "install_nwd_loss",
]

__version__ = "1.0.0"
