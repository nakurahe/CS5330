"""
Puzzle Solver Module
Assembles puzzle pieces using greedy algorithm with visual matching.

Uses piece type (corner/edge/interior) classification from the annotation file.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from itertools import permutations
import logging

from src.feature_extractor import PieceType

logger = logging.getLogger(__name__)


class PuzzleGrid:
    """Represents the puzzle grid with placed pieces."""
    
    def __init__(self, rows: int, cols: int):
        """
        Initialize the puzzle grid.
        
        Args:
            rows: Number of rows
            cols: Number of columns
        """
        self.rows = rows
        self.cols = cols
        self.grid = {}  # (row, col) -> piece_id
        self.placed_pieces = set()
    
    def place_piece(self, row: int, col: int, piece_id: int):
        """Place a piece at the specified position."""
        self.grid[(row, col)] = piece_id
        self.placed_pieces.add(piece_id)
    
    def remove_piece(self, row: int, col: int) -> Optional[int]:
        """Remove and return piece at position, or None if empty."""
        if (row, col) in self.grid:
            piece_id = self.grid.pop((row, col))
            self.placed_pieces.discard(piece_id)
            return piece_id
        return None
    
    def get_piece(self, row: int, col: int) -> Optional[int]:
        """Get piece ID at position, or None if empty."""
        return self.grid.get((row, col))
    
    def is_position_empty(self, row: int, col: int) -> bool:
        """Check if position is empty."""
        return (row, col) not in self.grid
    
    def is_valid_position(self, row: int, col: int) -> bool:
        """Check if position is within grid bounds."""
        return 0 <= row < self.rows and 0 <= col < self.cols
    
    def get_neighbors(self, row: int, col: int) -> Dict[str, Optional[int]]:
        """Get piece IDs of neighbors (top, right, bottom, left)."""
        neighbors = {
            'top': self.get_piece(row - 1, col) if row > 0 else None,
            'right': self.get_piece(row, col + 1) if col < self.cols - 1 else None,
            'bottom': self.get_piece(row + 1, col) if row < self.rows - 1 else None,
            'left': self.get_piece(row, col - 1) if col > 0 else None,
        }
        return neighbors
    
    def get_empty_positions(self) -> List[Tuple[int, int]]:
        """Get all empty positions in the grid."""
        empty = []
        for row in range(self.rows):
            for col in range(self.cols):
                if self.is_position_empty(row, col):
                    empty.append((row, col))
        return empty
    
    def clear(self):
        """Clear the grid."""
        self.grid.clear()
        self.placed_pieces.clear()


class AssemblyStep:
    """Represents a single step in puzzle assembly."""
    
    def __init__(self, piece_id: int, row: int, col: int, reason: str = ""):
        self.piece_id = piece_id
        self.row = row
        self.col = col
        self.reason = reason
    
    def __repr__(self):
        return f"Step: Place piece #{self.piece_id} at ({self.row}, {self.col}) - {self.reason}"


class PuzzleSolver:
    """Solves the puzzle using greedy assembly with visual matching.
    
    Uses piece-type-only mode:
    - Corner pieces (2 flat edges) go to corner positions
    - Edge pieces (1 flat edge) go to border positions
    - Interior pieces (0 flat edges) go to center
    
    Positions are determined by trying all corner permutations and
    greedily placing edges based on visual matching with neighbors.
    """
    
    def __init__(self, rows: int = 3, cols: int = 3):
        """
        Initialize the solver.
        
        Args:
            rows: Number of rows in puzzle (default 3)
            cols: Number of columns in puzzle (default 3)
        """
        self.rows = rows
        self.cols = cols
        self.grid = PuzzleGrid(rows, cols)
        self.assembly_steps = []
        self.pieces_dict = {}  # piece_id -> piece object
        self.features_dict = {}  # piece_id -> features
    
    def solve(self, features_list: List, pieces: List) -> Tuple[PuzzleGrid, List[AssemblyStep]]:
        """
        Solve the puzzle using piece-type-only mode with visual matching.
        
        Args:
            features_list: List of PieceFeatures
            pieces: List of PuzzlePiece objects (required for visual matching)
            
        Returns:
            Tuple of (PuzzleGrid, assembly steps)
        """
        logger.info("Starting puzzle solving...")
        
        # Create lookup dictionaries
        self.features_dict = {f.piece_id: f for f in features_list}
        self.pieces_dict = {p.id: p for p in pieces}
        
        # Use piece-type-only mode with visual matching
        logger.info("Mode: piece-type-only (brute force with visual matching)")
        return self._solve_piece_type_only(features_list, pieces)
    
    def _solve_piece_type_only(self, features_list: List, pieces: List) -> Tuple[PuzzleGrid, List[AssemblyStep]]:
        """
        Solve puzzle using only piece types and visual matching.
        
        For a 3x3 puzzle:
        - 4 corners at positions (0,0), (0,2), (2,0), (2,2)
        - 4 edges at positions (0,1), (1,0), (1,2), (2,1)
        - 1 interior at position (1,1)
        """
        # Classify pieces by type
        corners = [f for f in features_list if f.piece_type == PieceType.CORNER]
        edges = [f for f in features_list if f.piece_type == PieceType.EDGE]
        interiors = [f for f in features_list if f.piece_type == PieceType.INTERIOR]
        
        logger.info(f"Pieces by type: {len(corners)} corners, {len(edges)} edges, {len(interiors)} interior")
        
        # Define positions by type for 3x3 grid
        corner_positions = [(0, 0), (0, 2), (2, 0), (2, 2)]
        edge_positions = [(0, 1), (1, 0), (1, 2), (2, 1)]
        interior_positions = [(1, 1)]
        
        # Strategy: Try all permutations of corners and find best overall match
        # Then greedily place edges and interior
        
        best_grid = None
        best_score = -1
        best_steps = []
        
        # Try different corner arrangements
        corner_perms = list(permutations(corners, min(4, len(corners))))
        logger.info(f"Trying {len(corner_perms)} corner permutations...")
        
        for corner_perm in corner_perms:
            self.grid.clear()
            self.assembly_steps = []
            
            # Place corners
            for i, corner in enumerate(corner_perm):
                if i < len(corner_positions):
                    row, col = corner_positions[i]
                    self.grid.place_piece(row, col, corner.piece_id)
                    self.assembly_steps.append(AssemblyStep(
                        corner.piece_id, row, col, "Corner piece"
                    ))
            
            # Greedily place edges based on visual matching with corners
            remaining_edges = list(edges)
            for row, col in edge_positions:
                best_edge, best_edge_score = self._find_best_piece_for_position(
                    remaining_edges, row, col, pieces
                )
                if best_edge:
                    self.grid.place_piece(row, col, best_edge.piece_id)
                    self.assembly_steps.append(AssemblyStep(
                        best_edge.piece_id, row, col, f"Edge piece (score: {best_edge_score:.3f})"
                    ))
                    remaining_edges.remove(best_edge)
            
            # Place interior
            for row, col in interior_positions:
                if interiors:
                    interior = interiors[0]
                    self.grid.place_piece(row, col, interior.piece_id)
                    self.assembly_steps.append(AssemblyStep(
                        interior.piece_id, row, col, "Interior piece"
                    ))
            
            # Calculate total grid score
            total_score = self._calculate_grid_score()
            
            if total_score > best_score:
                best_score = total_score
                best_grid = dict(self.grid.grid)
                best_steps = list(self.assembly_steps)
        
        # Restore best solution
        self.grid.clear()
        for (row, col), piece_id in best_grid.items():
            self.grid.place_piece(row, col, piece_id)
        self.assembly_steps = best_steps
        
        logger.info(f"Best solution score: {best_score:.3f}")
        logger.info(f"Puzzle solving complete. Placed {len(self.grid.placed_pieces)} pieces.")
        
        return self.grid, self.assembly_steps
    
    def _find_best_piece_for_position(self, candidates: List, row: int, col: int, 
                                       pieces: List) -> Tuple[Optional[object], float]:
        """
        Find the best piece for a position based on visual matching with neighbors.
        
        Args:
            candidates: List of candidate PieceFeatures
            row, col: Target position
            pieces: List of PuzzlePiece objects
            
        Returns:
            Tuple of (best_piece, best_score)
        """
        if not candidates:
            return None, 0.0
        
        best_piece = None
        best_score = -1
        
        for candidate in candidates:
            if candidate.piece_id in self.grid.placed_pieces:
                continue
            
            score = self._calculate_visual_match_score(candidate.piece_id, row, col, pieces)
            
            if score > best_score:
                best_score = score
                best_piece = candidate
        
        return best_piece, best_score
    
    def _calculate_visual_match_score(self, piece_id: int, row: int, col: int, 
                                       pieces: List) -> float:
        """
        Calculate visual matching score for placing a piece at a position.
        
        Compares edge color profiles with adjacent placed pieces.
        
        Args:
            piece_id: ID of piece to evaluate
            row, col: Target position
            pieces: List of PuzzlePiece objects
            
        Returns:
            Match score (higher is better)
        """
        neighbors = self.grid.get_neighbors(row, col)
        
        # Map: if neighbor is in direction X, we compare our edge X with their opposite edge
        opposite_edge = {'top': 'bottom', 'bottom': 'top', 'left': 'right', 'right': 'left'}
        
        scores = []
        
        piece = self.pieces_dict.get(piece_id)
        if piece is None:
            return 0.0
        
        for direction, neighbor_id in neighbors.items():
            if neighbor_id is None:
                continue
            
            neighbor_piece = self.pieces_dict.get(neighbor_id)
            if neighbor_piece is None:
                continue
            
            # Compare edges visually
            score = self._compare_edges_visually(
                piece, direction,
                neighbor_piece, opposite_edge[direction]
            )
            scores.append(score)
        
        if not scores:
            return 0.0
        
        return sum(scores) / len(scores)
    
    def _compare_edges_visually(self, piece1, edge1_dir: str, 
                                 piece2, edge2_dir: str) -> float:
        """
        Compare two edges visually using the actual piece contour.
        
        Key insight: Pieces have white backgrounds. We must:
        1. Extract the piece mask (non-white pixels)
        2. Find the actual contour boundary
        3. Sample pixels along the real edge, not the image border
        
        Uses multiple matching strategies:
        - Shape similarity (Hausdorff distance on normalized contours)
        - Color similarity along contours (with proper reversal for matching edges)
        - Edge region histogram comparison
        - Gradient pattern correlation
        
        Args:
            piece1: First PuzzlePiece
            edge1_dir: Edge direction on piece1 ('top', 'right', 'bottom', 'left')
            piece2: Second PuzzlePiece
            edge2_dir: Edge direction on piece2
            
        Returns:
            Similarity score (0-1)
        """
        scores = []
        weights = []
        
        # Method 1: Shape similarity using Hausdorff distance on normalized contours
        # (Lower weight - puzzle piece edges tend to have similar shapes)
        shape_score = self._calculate_shape_similarity(piece1, edge1_dir, piece2, edge2_dir)
        if shape_score > 0:
            scores.append(shape_score)
            weights.append(0.10)
        
        # Method 2: Contour-based boundary pixel comparison (with proper reversal)
        # (Most important - colors along actual edge boundary)
        contour_score = self._compare_contour_boundaries(piece1, edge1_dir, piece2, edge2_dir)
        if contour_score > 0:
            scores.append(contour_score)
            weights.append(0.45)
        
        # Method 3: Edge region color comparison (inside the piece, near the edge)
        region_score = self._compare_edge_regions(piece1, edge1_dir, piece2, edge2_dir)
        if region_score > 0:
            scores.append(region_score)
            weights.append(0.30)
        
        # Method 4: Gradient pattern comparison along actual edge
        grad_score = self._compare_contour_gradients(piece1, edge1_dir, piece2, edge2_dir)
        if grad_score > 0:
            scores.append(grad_score)
            weights.append(0.15)
        
        if not scores:
            return 0.0
        
        # Weighted average
        total_weight = sum(weights)
        return sum(s * w for s, w in zip(scores, weights)) / total_weight
    
    def _get_piece_mask(self, piece) -> Optional[np.ndarray]:
        """
        Extract binary mask of the actual puzzle piece (excluding white background).
        
        Returns:
            Binary mask (255 for piece, 0 for background)
        """
        if piece is None or piece.image is None:
            return None
        
        # Cache mask on piece object
        if hasattr(piece, '_cached_mask') and piece._cached_mask is not None:
            return piece._cached_mask
        
        img = piece.image
        white = np.array([255, 255, 255])
        tolerance = 15  # Allow some variation in "white"
        
        # Create mask where pixels differ from white
        diff = np.abs(img.astype(np.int16) - white)
        mask = np.any(diff > tolerance, axis=2).astype(np.uint8) * 255
        
        # Clean up with morphological operations
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        piece._cached_mask = mask
        return mask
    
    def _get_piece_contour(self, piece) -> Optional[np.ndarray]:
        """
        Get the main contour of the puzzle piece.
        
        Returns:
            Contour points as numpy array
        """
        if piece is None:
            return None
        
        # Cache contour
        if hasattr(piece, '_cached_contour') and piece._cached_contour is not None:
            return piece._cached_contour
        
        mask = self._get_piece_mask(piece)
        if mask is None:
            return None
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        
        # Get largest contour (the piece)
        contour = max(contours, key=cv2.contourArea)
        piece._cached_contour = contour
        return contour
    
    def _get_edge_contour_points(self, piece, edge_dir: str) -> Optional[np.ndarray]:
        """
        Extract contour points for a specific edge of the piece.
        
        Args:
            piece: PuzzlePiece object
            edge_dir: 'top', 'right', 'bottom', 'left'
            
        Returns:
            Array of (x, y) contour points for that edge
        """
        contour = self._get_piece_contour(piece)
        if contour is None:
            return None
        
        h, w = piece.image.shape[:2]
        margin = w // 6  # Exclude corners
        
        points = contour.reshape(-1, 2)
        
        if edge_dir == 'top':
            edge_pts = np.array([(x, y) for x, y in points 
                                if margin < x < w - margin and y < h // 2])
        elif edge_dir == 'bottom':
            edge_pts = np.array([(x, y) for x, y in points 
                                if margin < x < w - margin and y > h // 2])
        elif edge_dir == 'left':
            edge_pts = np.array([(x, y) for x, y in points 
                                if margin < y < h - margin and x < w // 2])
        elif edge_dir == 'right':
            edge_pts = np.array([(x, y) for x, y in points 
                                if margin < y < h - margin and x > w // 2])
        else:
            return None
        
        if len(edge_pts) < 10:
            return None
        
        return edge_pts
    
    def _normalize_contour(self, contour: np.ndarray) -> np.ndarray:
        """
        Normalize contour points to standard position and scale.
        
        Centers the contour at origin and scales to unit variance.
        This allows shape comparison independent of position/scale.
        
        Args:
            contour: Array of (x, y) points
            
        Returns:
            Normalized contour points
        """
        if len(contour) == 0:
            return contour
        
        # Center at origin
        centroid = np.mean(contour, axis=0)
        centered = contour - centroid
        
        # Scale to unit variance
        std = np.std(centered)
        if std > 1e-6:
            normalized = centered / std
        else:
            normalized = centered
        
        return normalized
    
    def _calculate_shape_similarity(self, piece1, edge1_dir: str,
                                     piece2, edge2_dir: str) -> float:
        """
        Calculate shape similarity between two edges using Hausdorff distance.
        
        Matching edges should have complementary shapes (tab fits into indent).
        We normalize the contours to compare shapes regardless of position/scale.
        
        Args:
            piece1: First PuzzlePiece
            edge1_dir: Edge direction on piece1
            piece2: Second PuzzlePiece
            edge2_dir: Edge direction on piece2
            
        Returns:
            Similarity score (0-1), higher is more similar
        """
        # Get edge contour points
        pts1 = self._get_edge_contour_points(piece1, edge1_dir)
        pts2 = self._get_edge_contour_points(piece2, edge2_dir)
        
        if pts1 is None or pts2 is None:
            return 0.0
        
        if len(pts1) < 10 or len(pts2) < 10:
            return 0.0
        
        try:
            # Normalize contours
            norm1 = self._normalize_contour(pts1.astype(np.float32))
            norm2 = self._normalize_contour(pts2.astype(np.float32))
            
            # For matching edges, flip one contour (edges meet from opposite directions)
            # Flip the y-coordinates for horizontal edges, x for vertical
            if edge1_dir in ['top', 'bottom']:
                norm2_flipped = norm2.copy()
                norm2_flipped[:, 1] = -norm2_flipped[:, 1]  # Flip Y
            else:  # left, right
                norm2_flipped = norm2.copy()
                norm2_flipped[:, 0] = -norm2_flipped[:, 0]  # Flip X
            
            # Calculate Hausdorff distance (max of min distances)
            # For each point in norm1, find distance to closest point in norm2
            dist1 = np.min(np.linalg.norm(norm1[:, np.newaxis] - norm2_flipped, axis=2), axis=1)
            dist2 = np.min(np.linalg.norm(norm2_flipped[:, np.newaxis] - norm1, axis=2), axis=1)
            
            hausdorff = max(np.max(dist1), np.max(dist2))
            
            # Convert to similarity score using exponential decay
            # Lower distance = higher similarity
            score = np.exp(-hausdorff / 2.0)
            
            return float(score)
        except Exception as e:
            logger.debug(f"Shape similarity calculation failed: {e}")
            return 0.0

    def _sample_contour_edge_pixels(self, piece, edge_dir: str, 
                                     num_samples: int = 50) -> Optional[np.ndarray]:
        """
        Sample pixel colors along the piece contour for a specific edge.
        
        Args:
            piece: PuzzlePiece object
            edge_dir: 'top', 'right', 'bottom', 'left'
            num_samples: Number of pixels to sample
            
        Returns:
            Array of RGB values (num_samples, 3) or None
        """
        if piece is None or piece.image is None:
            return None
        
        contour = self._get_piece_contour(piece)
        if contour is None:
            return None
        
        h, w = piece.image.shape[:2]
        margin = w // 6  # Exclude corners
        
        # Filter contour points to get only the edge region
        points = contour.reshape(-1, 2)
        
        if edge_dir == 'top':
            # Top edge: y should be small, x in middle range
            edge_pts = [(x, y) for x, y in points 
                       if margin < x < w - margin and y < h // 2]
            if edge_pts:
                # Sort by x for consistent ordering
                edge_pts.sort(key=lambda p: p[0])
        elif edge_dir == 'bottom':
            edge_pts = [(x, y) for x, y in points 
                       if margin < x < w - margin and y > h // 2]
            if edge_pts:
                edge_pts.sort(key=lambda p: p[0])
        elif edge_dir == 'left':
            edge_pts = [(x, y) for x, y in points 
                       if margin < y < h - margin and x < w // 2]
            if edge_pts:
                edge_pts.sort(key=lambda p: p[1])
        elif edge_dir == 'right':
            edge_pts = [(x, y) for x, y in points 
                       if margin < y < h - margin and x > w // 2]
            if edge_pts:
                edge_pts.sort(key=lambda p: p[1])
        else:
            return None
        
        if len(edge_pts) < 5:
            return None
        
        # Sample evenly spaced points
        indices = np.linspace(0, len(edge_pts) - 1, num_samples).astype(int)
        sampled_pts = [edge_pts[i] for i in indices]
        
        # Get pixel colors at sampled points (sample slightly inside the piece)
        colors = []
        mask = self._get_piece_mask(piece)
        if mask is None:
            return None
        
        for x, y in sampled_pts:
            # Move slightly inside the piece to get actual piece color, not edge artifact
            if edge_dir == 'top':
                y_sample = min(y + 3, h - 1)
                x_sample = x
            elif edge_dir == 'bottom':
                y_sample = max(y - 3, 0)
                x_sample = x
            elif edge_dir == 'left':
                x_sample = min(x + 3, w - 1)
                y_sample = y
            elif edge_dir == 'right':
                x_sample = max(x - 3, 0)
                y_sample = y
            else:
                x_sample, y_sample = x, y
            
            # Ensure we're sampling from inside the piece
            if mask[y_sample, x_sample] > 0:
                colors.append(piece.image[y_sample, x_sample])
            else:
                colors.append(piece.image[y, x])
        
        return np.array(colors)
    
    def _compare_contour_boundaries(self, piece1, edge1_dir: str,
                                     piece2, edge2_dir: str) -> float:
        """
        Compare pixel colors along the actual piece contours that would touch.
        
        Key insight: When two edges meet, they are oriented in opposite directions.
        For example, piece1's right edge meets piece2's left edge, but they're
        sampled in opposite orders. We reverse one to align them properly.
        """
        colors1 = self._sample_contour_edge_pixels(piece1, edge1_dir)
        colors2 = self._sample_contour_edge_pixels(piece2, edge2_dir)
        
        if colors1 is None or colors2 is None:
            return 0.0
        
        if len(colors1) == 0 or len(colors2) == 0:
            return 0.0
        
        # Ensure same length
        min_len = min(len(colors1), len(colors2))
        colors1 = colors1[:min_len]
        colors2 = colors2[:min_len]
        
        # Reverse one edge to align the comparison properly
        # When edges meet, they go in opposite directions
        colors2_reversed = colors2[::-1]
        
        try:
            # Convert to LAB for perceptual comparison
            colors1_lab = cv2.cvtColor(colors1.reshape(1, -1, 3).astype(np.uint8), 
                                       cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
            colors2_lab = cv2.cvtColor(colors2.reshape(1, -1, 3).astype(np.uint8), 
                                       cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
            colors2_rev_lab = cv2.cvtColor(colors2_reversed.reshape(1, -1, 3).astype(np.uint8), 
                                           cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
            
            # Calculate per-pixel color difference for both orderings
            # and use the better one (since we're not sure of exact orientation)
            diff_normal = np.sqrt(np.sum((colors1_lab - colors2_lab) ** 2, axis=1))
            diff_reversed = np.sqrt(np.sum((colors1_lab - colors2_rev_lab) ** 2, axis=1))
            
            avg_normal = np.mean(diff_normal)
            avg_reversed = np.mean(diff_reversed)
            
            # Use the better alignment
            avg_diff = min(avg_normal, avg_reversed)
            
            # Convert to similarity (exponential decay)
            score = np.exp(-avg_diff / 25.0)
            
            return float(score)
        except Exception as e:
            logger.debug(f"Contour boundary comparison failed: {e}")
            return 0.0
    
    def _compare_edge_regions(self, piece1, edge1_dir: str,
                               piece2, edge2_dir: str) -> float:
        """
        Compare color distributions in regions near the edges.
        
        This captures the overall color character of each edge area.
        """
        region1 = self._extract_edge_region(piece1, edge1_dir)
        region2 = self._extract_edge_region(piece2, edge2_dir)
        
        if region1 is None or region2 is None:
            return 0.0
        
        # Compare using color histograms in LAB space
        try:
            lab1 = cv2.cvtColor(region1, cv2.COLOR_BGR2LAB)
            lab2 = cv2.cvtColor(region2, cv2.COLOR_BGR2LAB)
            
            scores = []
            for i in range(3):  # L, A, B channels
                h1 = cv2.calcHist([lab1], [i], None, [32], [0, 256])
                h2 = cv2.calcHist([lab2], [i], None, [32], [0, 256])
                
                h1 = cv2.normalize(h1, h1).flatten()
                h2 = cv2.normalize(h2, h2).flatten()
                
                corr = cv2.compareHist(h1.astype(np.float32), 
                                       h2.astype(np.float32), 
                                       cv2.HISTCMP_CORREL)
                scores.append((corr + 1) / 2)  # Normalize to 0-1
            
            return float(np.mean(scores))
        except Exception as e:
            logger.debug(f"Edge region comparison failed: {e}")
            return 0.0
    
    def _extract_edge_region(self, piece, edge_dir: str, 
                              depth: int = 40) -> Optional[np.ndarray]:
        """
        Extract a region of pixels near an edge, excluding white background.
        
        Args:
            piece: PuzzlePiece object
            edge_dir: 'top', 'right', 'bottom', 'left'
            depth: How deep into the piece to sample
            
        Returns:
            Image region with background pixels excluded (masked)
        """
        if piece is None or piece.image is None:
            return None
        
        mask = self._get_piece_mask(piece)
        if mask is None:
            return None
                
        # Define the region based on edge direction
        if edge_dir == 'top':
            region = piece.image[:depth, :, :]
            region_mask = mask[:depth, :]
        elif edge_dir == 'bottom':
            region = piece.image[-depth:, :, :]
            region_mask = mask[-depth:, :]
        elif edge_dir == 'left':
            region = piece.image[:, :depth, :]
            region_mask = mask[:, :depth]
        elif edge_dir == 'right':
            region = piece.image[:, -depth:, :]
            region_mask = mask[:, -depth:]
        else:
            return None
        
        # Extract only pixels that are part of the piece
        piece_pixels = region[region_mask > 0]
        
        if len(piece_pixels) < 100:
            return None
        
        # Return as a 2D array for histogram calculation
        # Reshape to approximate square for histogram
        side = int(np.sqrt(len(piece_pixels)))
        if side < 10:
            return None
        
        piece_pixels = piece_pixels[:side*side].reshape(side, side, 3)
        return piece_pixels
    
    def _compare_contour_gradients(self, piece1, edge1_dir: str,
                                    piece2, edge2_dir: str) -> float:
        """
        Compare gradient patterns along the actual piece edges.
        """
        grad1 = self._extract_contour_gradient(piece1, edge1_dir)
        grad2 = self._extract_contour_gradient(piece2, edge2_dir)
        
        if grad1 is None or grad2 is None:
            return 0.0
        
        # Ensure same length
        min_len = min(len(grad1), len(grad2))
        if min_len < 10:
            return 0.0
        
        grad1 = grad1[:min_len]
        grad2 = grad2[:min_len]
        
        # Compute correlation
        if np.std(grad1) < 1e-6 or np.std(grad2) < 1e-6:
            return 0.5  # Flat gradients
        
        try:
            correlation = np.corrcoef(grad1, grad2)[0, 1]
            if np.isnan(correlation):
                return 0.5
            return float((correlation + 1) / 2)
        except:
            return 0.0
    
    def _extract_contour_gradient(self, piece, edge_dir: str) -> Optional[np.ndarray]:
        """
        Extract gradient magnitudes along the piece contour edge.
        """
        if piece is None or piece.image is None:
            return None
        
        contour = self._get_piece_contour(piece)
        if contour is None:
            return None
        
        # Convert to grayscale and compute gradients
        gray = cv2.cvtColor(piece.image, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        h, w = piece.image.shape[:2]
        margin = w // 6
        
        # Get contour points for this edge
        points = contour.reshape(-1, 2)
        
        if edge_dir == 'top':
            edge_pts = [(x, y) for x, y in points 
                       if margin < x < w - margin and y < h // 2]
            edge_pts.sort(key=lambda p: p[0])
        elif edge_dir == 'bottom':
            edge_pts = [(x, y) for x, y in points 
                       if margin < x < w - margin and y > h // 2]
            edge_pts.sort(key=lambda p: p[0])
        elif edge_dir == 'left':
            edge_pts = [(x, y) for x, y in points 
                       if margin < y < h - margin and x < w // 2]
            edge_pts.sort(key=lambda p: p[1])
        elif edge_dir == 'right':
            edge_pts = [(x, y) for x, y in points 
                       if margin < y < h - margin and x > w // 2]
            edge_pts.sort(key=lambda p: p[1])
        else:
            return None
        
        if len(edge_pts) < 10:
            return None
        
        # Sample gradients at contour points
        gradients = [grad_mag[y, x] for x, y in edge_pts]
        
        return np.array(gradients)
    
    def _calculate_grid_score(self) -> float:
        """
        Calculate overall grid score based on all edge matches.
        
        Returns:
            Total score
        """
        total_score = 0.0
        num_edges = 0
        
        for row in range(self.rows):
            for col in range(self.cols):
                piece_id = self.grid.get_piece(row, col)
                if piece_id is None:
                    continue
                
                # Check right neighbor
                if col < self.cols - 1:
                    right_id = self.grid.get_piece(row, col + 1)
                    if right_id is not None:
                        piece = self.pieces_dict.get(piece_id)
                        right_piece = self.pieces_dict.get(right_id)
                        if piece and right_piece:
                            score = self._compare_edges_visually(piece, 'right', right_piece, 'left')
                            total_score += score
                            num_edges += 1
                
                # Check bottom neighbor
                if row < self.rows - 1:
                    bottom_id = self.grid.get_piece(row + 1, col)
                    if bottom_id is not None:
                        piece = self.pieces_dict.get(piece_id)
                        bottom_piece = self.pieces_dict.get(bottom_id)
                        if piece and bottom_piece:
                            score = self._compare_edges_visually(piece, 'bottom', bottom_piece, 'top')
                            total_score += score
                            num_edges += 1
        
        if num_edges == 0:
            return 0.0
        
        return total_score / num_edges
