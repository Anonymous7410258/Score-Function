"""Inference package for Meta-TTA-TSM."""

from inference.tta import TestTimeAdapter
from inference.predict import ScorePredictor

__all__ = ["TestTimeAdapter", "ScorePredictor"]
