"""
Puzzle Data Loader Module
Loads pre-separated puzzle pieces and annotations from the new input format.

Expected directory structure:
    puzzle_dir/
    ├── pieces/
    │   ├── piece_0.png
    │   ├── piece_1.png
    │   └── ...
    └── annotations/
        └── puzzle_info.json
"""

import cv2
import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

from .feature_extractor import PieceType


# Mapping from string piece types to PieceType enum
PIECE_TYPE_MAP = {
    "corner": PieceType.CORNER,
    "edge": PieceType.EDGE,
    "interior": PieceType.INTERIOR
}


@dataclass
class PieceAnnotation:
    """Ground truth annotation for a puzzle piece."""
    piece_id: int
    row: int
    col: int
    piece_type: PieceType
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in original image
    
    @classmethod
    def from_json(cls, data: dict) -> 'PieceAnnotation':
        """Create PieceAnnotation from JSON data."""
        piece_type = PIECE_TYPE_MAP.get(data.get('type', 'interior'), PieceType.INTERIOR)
        
        return cls(
            piece_id=data['piece_id'],
            row=data['row'],
            col=data['col'],
            piece_type=piece_type,
            bbox=tuple(data.get('bbox', [0, 0, 0, 0]))
        )


@dataclass
class PuzzleInfo:
    """Complete puzzle metadata from annotations."""
    puzzle_name: str
    grid_size: int
    image_size: Tuple[int, int]  # (width, height)
    piece_size: Tuple[int, int]  # (width, height)
    pieces: List[PieceAnnotation] = field(default_factory=list)
    original_image_path: Optional[str] = None
    template_id: Optional[int] = None
    
    @classmethod
    def from_json(cls, data: dict) -> 'PuzzleInfo':
        """Create PuzzleInfo from JSON data."""
        pieces = [PieceAnnotation.from_json(p) for p in data.get('pieces', [])]
        
        return cls(
            puzzle_name=data.get('puzzle_name', 'unknown'),
            grid_size=data.get('grid_size', 3),
            image_size=tuple(data.get('image_size', [0, 0])),
            piece_size=tuple(data.get('piece_size', [0, 0])),
            pieces=pieces,
            original_image_path=data.get('original_image'),
            template_id=data.get('template_id')
        )
    
    def get_piece_annotation(self, piece_id: int) -> Optional[PieceAnnotation]:
        """Get annotation for a specific piece by ID."""
        for piece in self.pieces:
            if piece.piece_id == piece_id:
                return piece
        return None
    
    def get_piece_at_position(self, row: int, col: int) -> Optional[PieceAnnotation]:
        """Get annotation for the piece at a specific grid position."""
        for piece in self.pieces:
            if piece.row == row and piece.col == col:
                return piece
        return None


class PuzzlePiece:
    """Represents a single puzzle piece with its properties."""
    
    def __init__(self, piece_id: int, image: np.ndarray, 
                 mask: Optional[np.ndarray] = None,
                 bbox: Tuple[int, int, int, int] = (0, 0, 0, 0),
                 contour: Optional[np.ndarray] = None,
                 annotation: Optional[PieceAnnotation] = None):
        """
        Initialize a puzzle piece.
        
        Args:
            piece_id: Unique identifier for the piece
            image: Piece image (BGR format)
            mask: Binary mask (optional, will be created if not provided)
            bbox: Bounding box (x, y, w, h)
            contour: Contour points (optional)
            annotation: Ground truth annotation (optional)
        """
        self.id = piece_id
        self.image = image
        self.bbox = bbox
        self.annotation = annotation
        self.angle = 0
        
        # Create mask if not provided (assume entire image is the piece)
        if mask is None:
            self.mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255
        else:
            self.mask = mask
        
        # Create contour from mask if not provided
        if contour is None:
            contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.contour = contours[0] if contours else np.array([])
        else:
            self.contour = contour
        
        self.centroid = self._calculate_centroid()
    
    def _calculate_centroid(self) -> Tuple[float, float]:
        """Calculate the centroid of the piece."""
        if len(self.contour) > 0:
            M = cv2.moments(self.contour)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                return (cx, cy)
        
        # Fallback to image center
        return (self.image.shape[1] / 2, self.image.shape[0] / 2)
    
    @property
    def ground_truth_type(self) -> Optional[PieceType]:
        """Get ground truth piece type if annotation is available."""
        if self.annotation:
            return self.annotation.piece_type
        return None


class PuzzleDataLoader:
    """Loads puzzle data from the new pre-separated format."""
    
    def __init__(self, puzzle_dir: str):
        """
        Initialize the loader with a puzzle directory.
        
        Args:
            puzzle_dir: Path to the puzzle directory containing pieces/ and annotations/
        """
        self.puzzle_dir = Path(puzzle_dir)
        self.pieces_dir = self.puzzle_dir / "pieces"
        self.annotations_dir = self.puzzle_dir / "annotations"
        self.puzzle_info: Optional[PuzzleInfo] = None
        self._pieces: List[PuzzlePiece] = []
        
    def load(self) -> Tuple[List[PuzzlePiece], PuzzleInfo]:
        """
        Load all puzzle pieces and annotations.
        
        Returns:
            Tuple of (list of PuzzlePiece objects, PuzzleInfo metadata)
        """
        logger.info(f"Loading puzzle from: {self.puzzle_dir}")
        
        # Load annotations
        self.puzzle_info = self._load_annotations()
        logger.info(f"Loaded puzzle info: {self.puzzle_info.puzzle_name}, "
                   f"grid: {self.puzzle_info.grid_size}x{self.puzzle_info.grid_size}, "
                   f"{len(self.puzzle_info.pieces)} pieces")
        
        # Load piece images
        self._pieces = self._load_pieces()
        logger.info(f"Loaded {len(self._pieces)} piece images")
        
        return self._pieces, self.puzzle_info
    
    def _load_annotations(self) -> PuzzleInfo:
        """Load puzzle annotations from JSON file."""
        json_path = self.annotations_dir / "puzzle_info.json"
        
        if not json_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {json_path}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        return PuzzleInfo.from_json(data)
    
    def _load_pieces(self) -> List[PuzzlePiece]:
        """Load all piece images and create PuzzlePiece objects."""
        pieces = []
        
        # Get expected number of pieces from annotations
        num_pieces = len(self.puzzle_info.pieces) if self.puzzle_info else 0
        
        if num_pieces == 0:
            # Try to detect pieces from directory
            piece_files = sorted(self.pieces_dir.glob("piece_*.png"))
            num_pieces = len(piece_files)
            logger.warning(f"No pieces in annotations, found {num_pieces} piece files")
        
        for i in range(num_pieces):
            piece_path = self.pieces_dir / f"piece_{i}.png"
            
            if not piece_path.exists():
                logger.warning(f"Piece file not found: {piece_path}")
                continue
            
            # Load image
            image = cv2.imread(str(piece_path))
            if image is None:
                logger.warning(f"Failed to load piece image: {piece_path}")
                continue
            
            # Get annotation if available
            annotation = None
            if self.puzzle_info:
                annotation = self.puzzle_info.get_piece_annotation(i)
            
            # Create bounding box (for pre-separated pieces, it's the full image)
            h, w = image.shape[:2]
            bbox = (0, 0, w, h)
            
            # Create puzzle piece
            piece = PuzzlePiece(
                piece_id=i,
                image=image,
                bbox=bbox,
                annotation=annotation
            )
            pieces.append(piece)
        
        return pieces
    
    @property
    def pieces(self) -> List[PuzzlePiece]:
        """Get loaded pieces."""
        return self._pieces
    
    @property 
    def grid_size(self) -> int:
        """Get puzzle grid size (assumes square grid)."""
        if self.puzzle_info:
            return self.puzzle_info.grid_size
        return int(np.sqrt(len(self._pieces)))
    
    @property
    def rows(self) -> int:
        """Get number of rows."""
        return self.grid_size
    
    @property
    def cols(self) -> int:
        """Get number of columns."""
        return self.grid_size
    
    def get_ground_truth_grid(self) -> Dict[Tuple[int, int], int]:
        """
        Get the ground truth grid mapping.
        
        Returns:
            Dict mapping (row, col) -> piece_id
        """
        grid = {}
        if self.puzzle_info:
            for piece in self.puzzle_info.pieces:
                grid[(piece.row, piece.col)] = piece.piece_id
        return grid


def load_puzzle(puzzle_dir: str) -> Tuple[List[PuzzlePiece], PuzzleInfo]:
    """
    Convenience function to load a puzzle from directory.
    
    Args:
        puzzle_dir: Path to puzzle directory
        
    Returns:
        Tuple of (pieces, puzzle_info)
    """
    loader = PuzzleDataLoader(puzzle_dir)
    return loader.load()
