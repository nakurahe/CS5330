"""
Jigsaw Puzzle Solver Package
Computer vision-based puzzle solving system.
"""

__version__ = "1.0.0"

from .piece_detector import PieceDetector, PuzzlePiece
from .feature_extractor import FeatureExtractor, PieceFeatures, EdgeType, PieceType
from .piece_matcher import PieceMatcher, EdgeMatch
from .solver import PuzzleSolver, PuzzleGrid, AssemblyStep
from .visualizer import PuzzleVisualizer

__all__ = [
    'PieceDetector',
    'PuzzlePiece',
    'FeatureExtractor',
    'PieceFeatures',
    'EdgeType',
    'PieceType',
    'PieceMatcher',
    'EdgeMatch',
    'PuzzleSolver',
    'PuzzleGrid',
    'AssemblyStep',
    'PuzzleVisualizer',
]
