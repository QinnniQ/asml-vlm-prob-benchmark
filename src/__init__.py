"""
ASML Vision-Language Benchmark Package
Provides modular components for dataset handling, model inference,
and probabilistic uncertainty metrics.
"""

from .dataset import WaferDataset
from .model import CLIPWrapper
from .metrics import (
    compute_entropy,
    compute_confidence,
    compute_brier_score,
)
