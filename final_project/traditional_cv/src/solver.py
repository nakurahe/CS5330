"""
Puzzle Solver Module
Assembles puzzle pieces using greedy algorithm with backtracking.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import logging

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
    """Solves the puzzle using greedy assembly with constraints."""
    
    def __init__(self, rows: int, cols: int):
        """
        Initialize the solver.
        
        Args:
            rows: Number of rows in puzzle
            cols: Number of columns in puzzle
        """
        self.rows = rows
        self.cols = cols
        self.grid = PuzzleGrid(rows, cols)
        self.assembly_steps = []
    
    def solve(self, features_list: List, matches: List) -> Tuple[PuzzleGrid, List[AssemblyStep]]:
        """
        Solve the puzzle.
        
        Args:
            features_list: List of PieceFeatures
            matches: List of EdgeMatch objects
            
        Returns:
            Tuple of (PuzzleGrid, assembly steps)
        """
        logger.info("Starting puzzle solving...")
        
        # Create lookup dictionaries
        self.features_dict = {f.piece_id: f for f in features_list}
        self.matches_dict = self._build_matches_dict(matches)
        
        # Step 1: Place corner pieces
        self._place_corners(features_list)
        
        # Step 2: Place edge pieces
        self._place_edges(features_list)
        
        # Step 3: Fill interior
        self._fill_interior(features_list)
        
        logger.info(f"Puzzle solving complete. Placed {len(self.grid.placed_pieces)} pieces.")
        return self.grid, self.assembly_steps
    
    def _build_matches_dict(self, matches: List) -> Dict:
        """
        Build a dictionary for quick match lookup.
        
        Args:
            matches: List of EdgeMatch objects
            
        Returns:
            Dictionary mapping (piece_id, edge) -> list of matches
        """
        matches_dict = {}
        
        for match in matches:
            # Add for piece 1
            key1 = (match.piece1_id, match.edge1)
            if key1 not in matches_dict:
                matches_dict[key1] = []
            matches_dict[key1].append(match)
            
            # Add for piece 2 (reverse direction)
            key2 = (match.piece2_id, match.edge2)
            if key2 not in matches_dict:
                matches_dict[key2] = []
            matches_dict[key2].append(match)
        
        # Sort each list by score
        for key in matches_dict:
            matches_dict[key].sort(key=lambda m: m.score, reverse=True)
        
        return matches_dict
    
    def _place_corners(self, features_list: List):
        """Place corner pieces in the grid."""
        from src.feature_extractor import PieceType, EdgeType
        
        # Find corner pieces (2 flat edges)
        corners = [f for f in features_list if f.piece_type == PieceType.CORNER]
        
        logger.info(f"Found {len(corners)} corner pieces")
        
        # Define corner positions
        corner_positions = [
            (0, 0, ['top', 'left']),           # Top-left
            (0, self.cols - 1, ['top', 'right']),     # Top-right
            (self.rows - 1, 0, ['bottom', 'left']),   # Bottom-left
            (self.rows - 1, self.cols - 1, ['bottom', 'right'])  # Bottom-right
        ]
        
        placed_corners = 0
        for corner in corners[:4]:  # Try to place up to 4 corners
            if placed_corners >= len(corner_positions):
                break
            
            # Find which edges are flat
            flat_edges = [name for name, edge in corner.edges.items() 
                         if edge.edge_type == EdgeType.FLAT]
            
            # Try to match with a corner position
            for row, col, required_edges in corner_positions:
                if not self.grid.is_position_empty(row, col):
                    continue
                
                # Check if flat edges match required edges
                if set(flat_edges) == set(required_edges):
                    self.grid.place_piece(row, col, corner.piece_id)
                    step = AssemblyStep(corner.piece_id, row, col, 
                                      f"Corner piece with flat edges: {', '.join(flat_edges)}")
                    self.assembly_steps.append(step)
                    placed_corners += 1
                    break
        
        logger.info(f"Placed {placed_corners} corner pieces")
    
    def _place_edges(self, features_list: List):
        """Place edge pieces (1 flat edge) around the border."""
        from src.feature_extractor import PieceType, EdgeType
        
        # Find edge pieces
        edge_pieces = [f for f in features_list 
                      if f.piece_id not in self.grid.placed_pieces 
                      and f.piece_type == PieceType.EDGE]
        
        logger.info(f"Found {len(edge_pieces)} edge pieces")
        
        # Place top edge
        self._place_edge_row(edge_pieces, 0, 'top')
        
        # Place bottom edge
        self._place_edge_row(edge_pieces, self.rows - 1, 'bottom')
        
        # Place left edge
        self._place_edge_col(edge_pieces, 0, 'left')
        
        # Place right edge
        self._place_edge_col(edge_pieces, self.cols - 1, 'right')
    
    def _place_edge_row(self, edge_pieces: List, row: int, flat_edge: str):
        """Place pieces along a horizontal edge."""
        from src.feature_extractor import EdgeType
        
        for col in range(self.cols):
            if not self.grid.is_position_empty(row, col):
                continue
            
            # Find best matching piece
            best_piece = None
            best_score = 0.0
            
            for piece in edge_pieces:
                if piece.piece_id in self.grid.placed_pieces:
                    continue
                
                # Check if piece has flat edge in required position
                edge = piece.edges.get(flat_edge)
                if edge is None or edge.edge_type != EdgeType.FLAT:
                    continue
                
                # Calculate compatibility with neighbors
                score = self._calculate_position_score(piece.piece_id, row, col)
                
                if score > best_score:
                    best_score = score
                    best_piece = piece
            
            if best_piece and best_score > 0.1:
                self.grid.place_piece(row, col, best_piece.piece_id)
                step = AssemblyStep(best_piece.piece_id, row, col, 
                                  f"Edge piece ({flat_edge} flat)")
                self.assembly_steps.append(step)
    
    def _place_edge_col(self, edge_pieces: List, col: int, flat_edge: str):
        """Place pieces along a vertical edge."""
        from src.feature_extractor import EdgeType
        
        for row in range(self.rows):
            if not self.grid.is_position_empty(row, col):
                continue
            
            # Find best matching piece
            best_piece = None
            best_score = 0.0
            
            for piece in edge_pieces:
                if piece.piece_id in self.grid.placed_pieces:
                    continue
                
                # Check if piece has flat edge in required position
                edge = piece.edges.get(flat_edge)
                if edge is None or edge.edge_type != EdgeType.FLAT:
                    continue
                
                # Calculate compatibility with neighbors
                score = self._calculate_position_score(piece.piece_id, row, col)
                
                if score > best_score:
                    best_score = score
                    best_piece = piece
            
            if best_piece and best_score > 0.1:
                self.grid.place_piece(row, col, best_piece.piece_id)
                step = AssemblyStep(best_piece.piece_id, row, col, 
                                  f"Edge piece ({flat_edge} flat)")
                self.assembly_steps.append(step)
    
    def _fill_interior(self, features_list: List):
        """Fill interior positions using greedy matching."""
        from src.feature_extractor import PieceType
        
        # Get remaining pieces
        remaining = [f for f in features_list 
                    if f.piece_id not in self.grid.placed_pieces]
        
        logger.info(f"Filling interior with {len(remaining)} remaining pieces")
        
        # Iteratively place pieces
        max_iterations = len(remaining) * 2
        iteration = 0
        
        while remaining and iteration < max_iterations:
            iteration += 1
            placed_this_iteration = False
            
            # Get positions adjacent to placed pieces (prioritize)
            frontier_positions = self._get_frontier_positions()
            
            if not frontier_positions:
                # If no frontier, try all empty positions
                frontier_positions = self.grid.get_empty_positions()
            
            # Try to place a piece at each frontier position
            for row, col in frontier_positions:
                best_piece = None
                best_score = 0.0
                
                for piece in remaining:
                    score = self._calculate_position_score(piece.piece_id, row, col)
                    
                    if score > best_score:
                        best_score = score
                        best_piece = piece
                
                if best_piece and best_score > 0.2:  # Threshold
                    self.grid.place_piece(row, col, best_piece.piece_id)
                    step = AssemblyStep(best_piece.piece_id, row, col, 
                                      f"Interior piece (score: {best_score:.2f})")
                    self.assembly_steps.append(step)
                    remaining.remove(best_piece)
                    placed_this_iteration = True
                    break
            
            if not placed_this_iteration:
                # Lower threshold or pick best remaining position
                break
        
        logger.info(f"Interior filling complete. {len(remaining)} pieces remaining.")
    
    def _get_frontier_positions(self) -> List[Tuple[int, int]]:
        """Get empty positions adjacent to placed pieces."""
        frontier = set()
        
        for (row, col), piece_id in self.grid.grid.items():
            # Check all adjacent positions
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_row, new_col = row + dr, col + dc
                if (self.grid.is_valid_position(new_row, new_col) and 
                    self.grid.is_position_empty(new_row, new_col)):
                    frontier.add((new_row, new_col))
        
        return list(frontier)
    
    def _calculate_position_score(self, piece_id: int, row: int, col: int) -> float:
        """
        Calculate how well a piece fits at a position based on neighbors.
        
        Args:
            piece_id: ID of piece to place
            row: Row position
            col: Column position
            
        Returns:
            Compatibility score
        """
        neighbors = self.grid.get_neighbors(row, col)
        
        # Mapping of neighbor direction to piece edge
        edge_map = {
            'top': 'top',
            'right': 'right',
            'bottom': 'bottom',
            'left': 'left'
        }
        
        scores = []
        
        for direction, neighbor_id in neighbors.items():
            if neighbor_id is None:
                continue
            
            # Get the edge that should match
            piece_edge = edge_map[direction]
            
            # Find match score between this piece's edge and neighbor's opposite edge
            key = (piece_id, piece_edge)
            if key in self.matches_dict:
                matches = self.matches_dict[key]
                # Find match with the specific neighbor
                for match in matches:
                    if match.piece1_id == neighbor_id or match.piece2_id == neighbor_id:
                        scores.append(match.score)
                        break
        
        if not scores:
            return 0.0
        
        # Return average score
        return sum(scores) / len(scores)


def main():
    """Test puzzle solver."""
    pass


if __name__ == "__main__":
    main()
