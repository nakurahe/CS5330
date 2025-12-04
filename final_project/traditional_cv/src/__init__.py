"""
Jigsaw Puzzle Solver Package
Computer vision-based puzzle solving system.
"""

__version__ = "1.0.0"

from .feature_extractor import PieceFeatures, PieceType, create_features_from_annotations
from .solver import PuzzleSolver, PuzzleGrid, AssemblyStep
from .visualizer import PuzzleVisualizer
from .puzzle_loader import PuzzleDataLoader, PuzzleInfo, PuzzlePiece
from .evaluator import PuzzleEvaluator, EvaluationMetrics

__all__ = [
    'PuzzlePiece',
    'PieceFeatures',
    'PieceType',
    'create_features_from_annotations',
    'PuzzleSolver',
    'PuzzleGrid',
    'AssemblyStep',
    'PuzzleVisualizer',
    'PuzzleDataLoader',
    'PuzzleInfo',
    'PuzzleEvaluator',
    'EvaluationMetrics',
]
