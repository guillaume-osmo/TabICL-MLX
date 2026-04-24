"""TabICL MLX -- Apple Silicon native TabICL inference.

Converted from the PyTorch tabicl package (soda-inria/tabicl).
Inference-only, supports regression (max_classes=0).
"""

from .model import TabICL
from .regressor import TabICLRegressorMLX
from .convert import convert_checkpoint, convert_from_huggingface

__all__ = [
    "TabICL",
    "TabICLRegressorMLX",
    "convert_checkpoint",
    "convert_from_huggingface",
]
