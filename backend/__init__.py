"""System V4 Backend - Quantitative Trading Dashboard Backend."""

__version__ = "0.1.0"
__author__ = "System V4 Team"

from .config import settings, IndicatorConfiguration
from .data_loader import KLineDataLoader
from .indicators.registry import IndicatorRegistry, IndicatorPipeline

__all__ = [
    "settings",
    "IndicatorConfiguration",
    "KLineDataLoader",
    "IndicatorRegistry",
    "IndicatorPipeline"
]
