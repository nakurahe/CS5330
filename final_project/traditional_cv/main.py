"""
Main Application - Jigsaw Puzzle Solver
Complete pipeline for solving real jigsaw puzzles from pre-separated pieces.

Input: --puzzle-dir puzzle_folder/ (directory with pieces/ and annotations/)
"""

import numpy as np
import os
import logging
from pathlib import Path
from typing import Tuple

from src.feature_extractor import PieceType, create_features_from_annotations
from src.solver import PuzzleSolver
from src.visualizer import PuzzleVisualizer
from src.puzzle_loader import PuzzleDataLoader
from src.evaluator import PuzzleEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PuzzleSolverPipeline:
    """Complete puzzle solving pipeline."""
    
    def __init__(self):
        """Initialize the pipeline."""
        self.rows = None
        self.cols = None
        self.puzzle_info = None  # Will be set when loading from puzzle directory
        
        # Initialize components
        self.visualizer = PuzzleVisualizer()
    
    def solve_from_directory(self, puzzle_dir: str, output_dir: str = "data/output") -> bool:
        """
        Solve puzzle from pre-separated pieces directory.
        
        Args:
            puzzle_dir: Path to puzzle directory with pieces/ and annotations/
            output_dir: Directory for output files
            
        Returns:
            True if successful
        """
        logger.info(f"Starting puzzle solver pipeline for directory: {puzzle_dir}")
        
        try:
            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Step 1: Load pieces and annotations
            logger.info("=" * 50)
            logger.info("STEP 1: Loading puzzle pieces and annotations")
            logger.info("=" * 50)
            
            loader = PuzzleDataLoader(puzzle_dir)
            pieces, self.puzzle_info = loader.load()
            
            if len(pieces) == 0:
                logger.error("No pieces loaded!")
                return False
            
            logger.info(f"Loaded {len(pieces)} pieces")
            logger.info(f"Grid size: {self.puzzle_info.grid_size}x{self.puzzle_info.grid_size}")
            
            # Set grid size from puzzle info
            self.rows = self.puzzle_info.grid_size
            self.cols = self.puzzle_info.grid_size
            
            # Create a combined image for visualization (optional)
            piece_size = self.puzzle_info.piece_size
            combined_image = self._create_combined_image(pieces, piece_size)
            
            # Step 2: Create features from ground truth annotations
            logger.info("=" * 50)
            logger.info("STEP 2: Creating piece features from annotations")
            logger.info("=" * 50)
            
            # Create features directly from ground truth piece types
            features_list = create_features_from_annotations(pieces)
            
            # Log piece types
            corners = sum(1 for f in features_list if f.piece_type == PieceType.CORNER)
            edges = sum(1 for f in features_list if f.piece_type == PieceType.EDGE)
            interior = sum(1 for f in features_list if f.piece_type == PieceType.INTERIOR)
            logger.info(f"Piece classification: {corners} corners, {edges} edges, {interior} interior")
            
            # Step 3: Solve puzzle
            logger.info("=" * 50)
            logger.info("STEP 3: Assembling puzzle")
            logger.info("=" * 50)
            
            try:
                # Solver uses piece-type-only mode with visual matching
                solver = PuzzleSolver(self.rows, self.cols)
                grid, assembly_steps = solver.solve(features_list, pieces)
            except Exception as e:
                logger.error(f"Puzzle solving failed: {e}")
                import traceback
                traceback.print_exc()
                return False
            
            logger.info(f"Placed {len(grid.placed_pieces)} out of {len(pieces)} pieces")
            logger.info(f"Assembly has {len(assembly_steps)} steps")
            
            # Step 4: Evaluate solution against ground truth
            logger.info("=" * 50)
            logger.info("STEP 4: Evaluating solution")
            logger.info("=" * 50)
            
            evaluator = PuzzleEvaluator(self.puzzle_info)
            metrics = evaluator.evaluate(grid)
            evaluator.print_grid_comparison(grid)
            
            # Step 5: Visualize solution
            logger.info("=" * 50)
            logger.info("STEP 5: Creating visualizations")
            logger.info("=" * 50)
            
            # Get base filename
            base_name = self.puzzle_info.puzzle_name
            output_prefix = os.path.join(output_dir, base_name)
            
            try:
                output_files = self.visualizer.create_solution_visualization(
                    combined_image, pieces, grid, assembly_steps, output_prefix
                )
            except Exception as e:
                logger.error(f"Visualization failed: {e}")
                import traceback
                traceback.print_exc()
                # Continue anyway, visualization is not critical
                output_files = []
            
            logger.info(f"Created {len(output_files)} output files")
            for file in output_files:
                logger.info(f"  - {file}")
            
            logger.info("=" * 50)
            logger.info("PUZZLE SOLVING COMPLETE!")
            logger.info(f"Direct Accuracy: {metrics.direct_accuracy:.1%}")
            logger.info(f"Neighbor Accuracy: {metrics.neighbor_accuracy:.1%}")
            logger.info("=" * 50)
            
            return True
            
        except Exception as e:
            logger.error(f"Unexpected error in pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_combined_image(self, pieces, piece_size: Tuple[int, int]) -> np.ndarray:
        """Create a combined image from all pieces for visualization."""
        # Arrange pieces in a grid for display
        n_pieces = len(pieces)
        grid_dim = int(np.ceil(np.sqrt(n_pieces)))
        
        # Get piece size from actual pieces if piece_size is not valid
        if piece_size[0] <= 0 or piece_size[1] <= 0:
            if pieces and pieces[0].image is not None:
                h, w = pieces[0].image.shape[:2]
            else:
                h, w = 300, 300  # Default fallback
        else:
            h, w = piece_size[1], piece_size[0]
        
        combined = np.ones((grid_dim * h, grid_dim * w, 3), dtype=np.uint8) * 128
        
        for i, piece in enumerate(pieces):
            row = i // grid_dim
            col = i % grid_dim
            
            # Handle size mismatch
            ph, pw = piece.image.shape[:2]
            combined[row * h:row * h + ph, col * w:col * w + pw] = piece.image
        
        return combined


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Jigsaw Puzzle Solver")
    parser.add_argument("--puzzle-dir", "-p", type=str, required=True,
                       help="Puzzle directory path with pieces/ and annotations/")
    parser.add_argument("--output", "-o", type=str, default="data/output",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Directory mode - pre-separated pieces with annotations
    if not os.path.isdir(args.puzzle_dir):
        print(f"Error: Puzzle directory not found: {args.puzzle_dir}")
        return
    
    pipeline = PuzzleSolverPipeline()
    success = pipeline.solve_from_directory(args.puzzle_dir, args.output)
    
    if success:
        print("\n" + "=" * 50)
        print("SUCCESS! Check the output directory for results:")
        print(f"  {os.path.abspath(args.output)}")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("FAILED: Could not solve puzzle. Check logs for details.")
        print("=" * 50)


if __name__ == "__main__":
    main()
