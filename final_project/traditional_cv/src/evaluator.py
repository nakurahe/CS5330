"""
Puzzle Solver Evaluation Module
Evaluates puzzle solving results against ground truth annotations.
"""

from typing import List, Dict, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    # Direct placement accuracy
    direct_accuracy: float  # Percentage of pieces in correct positions
    pieces_correct: int
    pieces_total: int
    
    # Neighbor accuracy
    neighbor_accuracy: float  # Percentage of correct neighbor relationships
    neighbors_correct: int
    neighbors_total: int
    
    # By piece type
    corner_accuracy: float
    edge_accuracy: float
    interior_accuracy: float
    
    # Detailed results
    correct_placements: List[int]  # List of correctly placed piece IDs
    incorrect_placements: Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]]  # piece_id -> (predicted, actual)
    
    def __str__(self) -> str:
        lines = [
            "=" * 50,
            "EVALUATION RESULTS",
            "=" * 50,
            f"Direct Placement Accuracy: {self.direct_accuracy:.1%} ({self.pieces_correct}/{self.pieces_total})",
            f"Neighbor Accuracy: {self.neighbor_accuracy:.1%} ({self.neighbors_correct}/{self.neighbors_total})",
            "",
            "By Piece Type:",
            f"  Corners:  {self.corner_accuracy:.1%}",
            f"  Edges:    {self.edge_accuracy:.1%}",
            f"  Interior: {self.interior_accuracy:.1%}",
            "=" * 50,
        ]
        return "\n".join(lines)


class PuzzleEvaluator:
    """Evaluates puzzle solving results against ground truth."""
    
    def __init__(self, puzzle_info):
        """
        Initialize evaluator with puzzle info.
        
        Args:
            puzzle_info: PuzzleInfo object with ground truth annotations
        """
        self.puzzle_info = puzzle_info
        self.grid_size = puzzle_info.grid_size
        
        # Build ground truth mappings
        self.gt_grid = {}  # (row, col) -> piece_id
        self.gt_positions = {}  # piece_id -> (row, col)
        self.gt_types = {}  # piece_id -> piece_type
        
        for piece in puzzle_info.pieces:
            self.gt_grid[(piece.row, piece.col)] = piece.piece_id
            self.gt_positions[piece.piece_id] = (piece.row, piece.col)
            self.gt_types[piece.piece_id] = piece.piece_type
    
    def evaluate(self, predicted_grid) -> EvaluationMetrics:
        """
        Evaluate predicted grid against ground truth.
        
        Args:
            predicted_grid: PuzzleGrid object with predicted placements
            
        Returns:
            EvaluationMetrics with detailed results
        """
        logger.info("Evaluating puzzle solution...")
        
        # Direct placement accuracy
        direct_results = self._evaluate_direct_placement(predicted_grid)
        
        # Neighbor accuracy
        neighbor_results = self._evaluate_neighbors(predicted_grid)
        
        # By piece type accuracy
        type_results = self._evaluate_by_type(predicted_grid)
        
        metrics = EvaluationMetrics(
            direct_accuracy=direct_results['accuracy'],
            pieces_correct=direct_results['correct'],
            pieces_total=direct_results['total'],
            neighbor_accuracy=neighbor_results['accuracy'],
            neighbors_correct=neighbor_results['correct'],
            neighbors_total=neighbor_results['total'],
            corner_accuracy=type_results['corner'],
            edge_accuracy=type_results['edge'],
            interior_accuracy=type_results['interior'],
            correct_placements=direct_results['correct_pieces'],
            incorrect_placements=direct_results['incorrect_pieces']
        )
        
        logger.info(str(metrics))
        return metrics
    
    def _evaluate_direct_placement(self, predicted_grid) -> Dict:
        """Evaluate direct placement accuracy."""
        correct = 0
        total = 0
        correct_pieces = []
        incorrect_pieces = {}
        
        for (row, col), gt_piece_id in self.gt_grid.items():
            total += 1
            pred_piece_id = predicted_grid.get_piece(row, col)
            
            if pred_piece_id == gt_piece_id:
                correct += 1
                correct_pieces.append(gt_piece_id)
            else:
                # Find where the piece was actually placed
                pred_pos = None
                for (r, c), pid in predicted_grid.grid.items():
                    if pid == gt_piece_id:
                        pred_pos = (r, c)
                        break
                incorrect_pieces[gt_piece_id] = (pred_pos, (row, col))
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'correct_pieces': correct_pieces,
            'incorrect_pieces': incorrect_pieces
        }
    
    def _evaluate_neighbors(self, predicted_grid) -> Dict:
        """Evaluate neighbor relationship accuracy."""
        correct = 0
        total = 0
        
        # For each placed piece, check its neighbors
        for (row, col), piece_id in predicted_grid.grid.items():
            pred_neighbors = predicted_grid.get_neighbors(row, col)
            
            # Get ground truth position and neighbors for this piece
            if piece_id not in self.gt_positions:
                continue
            
            gt_row, gt_col = self.gt_positions[piece_id]
            
            # Compare neighbors in each direction
            directions = ['top', 'right', 'bottom', 'left']
            offsets = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            
            for direction, (dr, dc) in zip(directions, offsets):
                gt_neighbor_pos = (gt_row + dr, gt_col + dc)
                
                # Check if there should be a neighbor in this direction
                if gt_neighbor_pos in self.gt_grid:
                    total += 1
                    gt_neighbor_id = self.gt_grid[gt_neighbor_pos]
                    pred_neighbor_id = pred_neighbors.get(direction)
                    
                    if pred_neighbor_id == gt_neighbor_id:
                        correct += 1
        
        # Avoid double counting (each edge counted twice)
        # Actually, since we check from each piece's perspective, we should divide by 2
        # But for simplicity, we'll keep it as is since it's symmetric
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
    
    def _evaluate_by_type(self, predicted_grid) -> Dict:
        """Evaluate accuracy by piece type."""
        from src.feature_extractor import PieceType
        
        type_correct = {PieceType.CORNER: 0, PieceType.EDGE: 0, PieceType.INTERIOR: 0}
        type_total = {PieceType.CORNER: 0, PieceType.EDGE: 0, PieceType.INTERIOR: 0}
        
        for (row, col), gt_piece_id in self.gt_grid.items():
            piece_type = self.gt_types.get(gt_piece_id)
            if piece_type is None:
                continue
            
            type_total[piece_type] += 1
            pred_piece_id = predicted_grid.get_piece(row, col)
            
            if pred_piece_id == gt_piece_id:
                type_correct[piece_type] += 1
        
        return {
            'corner': type_correct[PieceType.CORNER] / type_total[PieceType.CORNER] if type_total[PieceType.CORNER] > 0 else 0.0,
            'edge': type_correct[PieceType.EDGE] / type_total[PieceType.EDGE] if type_total[PieceType.EDGE] > 0 else 0.0,
            'interior': type_correct[PieceType.INTERIOR] / type_total[PieceType.INTERIOR] if type_total[PieceType.INTERIOR] > 0 else 0.0
        }
    
    def print_grid_comparison(self, predicted_grid):
        """Print side-by-side comparison of predicted vs ground truth grid."""
        print("\nGrid Comparison (Predicted | Ground Truth):")
        print("-" * 50)
        
        for row in range(self.grid_size):
            pred_row = []
            gt_row = []
            for col in range(self.grid_size):
                pred_id = predicted_grid.get_piece(row, col)
                gt_id = self.gt_grid.get((row, col))
                
                pred_str = f"{pred_id:2d}" if pred_id is not None else " -"
                gt_str = f"{gt_id:2d}" if gt_id is not None else " -"
                
                # Mark correct with *
                marker = "*" if pred_id == gt_id else " "
                pred_row.append(f"{pred_str}{marker}")
                gt_row.append(f"{gt_str} ")
            
            print(f"  [{' '.join(pred_row)}]  |  [{' '.join(gt_row)}]")
        
        print("-" * 50)
        print("* indicates correct placement")


def evaluate_solution(predicted_grid, puzzle_info) -> EvaluationMetrics:
    """
    Convenience function to evaluate a solution.
    
    Args:
        predicted_grid: PuzzleGrid with predicted placements
        puzzle_info: PuzzleInfo with ground truth
        
    Returns:
        EvaluationMetrics
    """
    evaluator = PuzzleEvaluator(puzzle_info)
    return evaluator.evaluate(predicted_grid)
