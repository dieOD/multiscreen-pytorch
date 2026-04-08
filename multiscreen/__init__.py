"""Multiscreen: a screening-based language model architecture.

Reference: "Screening Is Enough" (Nakanishi, 2026; arXiv:2604.01178).
"""

from multiscreen.config import MultiscreenConfig
from multiscreen.model import MultiscreenModel, MultiscreenLayer, GatedScreeningBlock

__version__ = "0.1.0"
__all__ = [
    "MultiscreenConfig",
    "MultiscreenModel",
    "MultiscreenLayer",
    "GatedScreeningBlock",
]
