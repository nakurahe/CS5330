"""
Feature Extraction Module
Defines data structures for puzzle piece features.

Note: Most feature extraction has been removed as the solver now uses:
- Ground truth piece types (corner/edge/interior) from annotations
- Visual matching computed directly in the solver
"""

from typing import List
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PieceType(Enum):
    """Types of puzzle pieces based on number of flat edges."""
    CORNER = 2      # 2 flat edges
    EDGE = 1        # 1 flat edge
    INTERIOR = 0    # 0 flat edges


class PieceFeatures:
    """Container for puzzle piece features used by the solver."""
    
    def __init__(self, piece_id: int, piece_type: PieceType = PieceType.INTERIOR):
        """
        Initialize piece features.
        
        Args:
            piece_id: Unique identifier for the piece
            piece_type: Type of piece (CORNER, EDGE, or INTERIOR)
        """
        self.piece_id: int = piece_id
        self.piece_type: PieceType = piece_type


def create_features_from_annotations(pieces: List) -> List[PieceFeatures]:
    """
    Create PieceFeatures from puzzle pieces with ground truth annotations.
    
    This is the primary way to create features for the solver - using
    ground truth piece types from puzzle_info.json annotations.
    
    Args:
        pieces: List of PuzzlePiece objects with annotations
        
    Returns:
        List of PieceFeatures with piece types from ground truth
    """
    logger.info("Creating features from ground truth annotations...")
    
    features_list = []
    for piece in pieces:
        # Get piece type from annotation, default to INTERIOR if not available
        piece_type = PieceType.INTERIOR
        if hasattr(piece, 'annotation') and piece.annotation is not None:
            piece_type = piece.annotation.piece_type
        elif hasattr(piece, 'ground_truth_type') and piece.ground_truth_type is not None:
            piece_type = piece.ground_truth_type
        
        features = PieceFeatures(piece.id, piece_type)
        features_list.append(features)
    
    # Log statistics
    corners = sum(1 for f in features_list if f.piece_type == PieceType.CORNER)
    edges = sum(1 for f in features_list if f.piece_type == PieceType.EDGE)
    interior = sum(1 for f in features_list if f.piece_type == PieceType.INTERIOR)
    logger.info(f"Created features: {corners} corners, {edges} edges, {interior} interior")
    
    return features_list
