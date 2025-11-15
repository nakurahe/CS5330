"""
Main Application - Jigsaw Puzzle Solver
Complete pipeline for solving real jigsaw puzzles from photos.
"""

import cv2
import numpy as np
import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.piece_detector import PieceDetector
from src.feature_extractor import FeatureExtractor
from src.piece_matcher import PieceMatcher
from src.solver import PuzzleSolver
from src.visualizer import PuzzleVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PuzzleSolverPipeline:
    """Complete puzzle solving pipeline."""
    
    def __init__(self, rows: int = None, cols: int = None):
        """
        Initialize the pipeline.
        
        Args:
            rows: Number of puzzle rows (if known)
            cols: Number of puzzle columns (if known)
        """
        self.rows = rows
        self.cols = cols
        
        # Initialize components with adaptive detection
        self.detector = PieceDetector(adaptive=True)  # Use adaptive area thresholds
        self.feature_extractor = FeatureExtractor()
        self.matcher = PieceMatcher(shape_weight=0.3, color_weight=0.5, texture_weight=0.2)
        self.visualizer = PuzzleVisualizer()
    
    def solve_puzzle(self, image_path: str, output_dir: str = "data/output") -> bool:
        """
        Solve puzzle from image file.
        
        Args:
            image_path: Path to input image
            output_dir: Directory for output files
            
        Returns:
            True if successful
        """
        logger.info(f"Starting puzzle solver pipeline for: {image_path}")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return False
        
        logger.info(f"Image loaded: {image.shape}")
        
        # Step 1: Detect pieces
        logger.info("=" * 50)
        logger.info("STEP 1: Detecting puzzle pieces")
        logger.info("=" * 50)
        pieces = self.detector.detect_pieces(image)
        
        if len(pieces) == 0:
            logger.error("No pieces detected!")
            return False
        
        logger.info(f"Detected {len(pieces)} pieces")
        
        # Auto-detect grid size if not provided
        if self.rows is None or self.cols is None:
            self.rows, self.cols = self._estimate_grid_size(len(pieces))
            logger.info(f"Estimated grid size: {self.rows}x{self.cols}")
        
        # Step 2: Extract features
        logger.info("=" * 50)
        logger.info("STEP 2: Extracting features from pieces")
        logger.info("=" * 50)
        # Pass original image shape for boundary-based edge detection
        features_list = self.feature_extractor.extract_features(pieces, image_shape=image.shape[:2])
        
        # Log piece types
        from src.feature_extractor import PieceType
        corners = sum(1 for f in features_list if f.piece_type == PieceType.CORNER)
        edges = sum(1 for f in features_list if f.piece_type == PieceType.EDGE)
        interior = sum(1 for f in features_list if f.piece_type == PieceType.INTERIOR)
        logger.info(f"Piece classification: {corners} corners, {edges} edges, {interior} interior")
        
        # Step 3: Match pieces
        logger.info("=" * 50)
        logger.info("STEP 3: Finding piece matches")
        logger.info("=" * 50)
        matches = self.matcher.find_all_matches(features_list)
        logger.info(f"Found {len(matches)} potential matches")
        
        if len(matches) > 0:
            logger.info(f"Top 5 matches:")
            for match in matches[:5]:
                logger.info(f"  {match}")
        
        # Step 4: Solve puzzle
        logger.info("=" * 50)
        logger.info("STEP 4: Assembling puzzle")
        logger.info("=" * 50)
        solver = PuzzleSolver(self.rows, self.cols)
        grid, assembly_steps = solver.solve(features_list, matches)
        
        logger.info(f"Placed {len(grid.placed_pieces)} out of {len(pieces)} pieces")
        logger.info(f"Assembly has {len(assembly_steps)} steps")
        
        # Step 5: Visualize solution
        logger.info("=" * 50)
        logger.info("STEP 5: Creating visualizations")
        logger.info("=" * 50)
        
        # Get base filename
        base_name = Path(image_path).stem
        output_prefix = os.path.join(output_dir, base_name)
        
        output_files = self.visualizer.create_solution_visualization(
            image, pieces, grid, assembly_steps, output_prefix
        )
        
        logger.info(f"Created {len(output_files)} output files")
        for file in output_files:
            logger.info(f"  - {file}")
        
        logger.info("=" * 50)
        logger.info("PUZZLE SOLVING COMPLETE!")
        logger.info("=" * 50)
        
        return True
    
    def _estimate_grid_size(self, num_pieces: int) -> tuple:
        """
        Estimate grid dimensions from number of pieces.
        
        Args:
            num_pieces: Number of detected pieces
            
        Returns:
            Tuple of (rows, cols)
        """
        # Try to find factors close to square
        sqrt = int(np.sqrt(num_pieces))
        
        for rows in range(sqrt, 0, -1):
            if num_pieces % rows == 0:
                cols = num_pieces // rows
                return (rows, cols)
        
        # Fallback: closest to square
        rows = sqrt
        cols = int(np.ceil(num_pieces / rows))
        return (rows, cols)


def create_demo_puzzle(output_path: str = "data/input_images/demo_puzzle.jpg"):
    """
    Create a demo puzzle image for testing.
    
    Args:
        output_path: Path to save the demo image
    """
    logger.info("Creating demo puzzle image...")
    
    # Load a sample image or create one
    # For demo, create a simple colorful image
    width, height = 800, 600
    rows, cols = 3, 4
    piece_width = width // cols
    piece_height = height // rows
    
    # Create a colorful gradient image
    original = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i in range(rows):
        for j in range(cols):
            # Create unique color for each piece
            color = (
                int(255 * i / rows),
                int(255 * j / cols),
                int(255 * (i + j) / (rows + cols))
            )
            
            y1 = i * piece_height
            y2 = (i + 1) * piece_height
            x1 = j * piece_width
            x2 = (j + 1) * piece_width
            
            original[y1:y2, x1:x2] = color
            
            # Add some pattern
            cv2.circle(original, 
                      (x1 + piece_width // 2, y1 + piece_height // 2),
                      min(piece_width, piece_height) // 3,
                      (255, 255, 255), 2)
            
            # Add piece number
            cv2.putText(original, f"{i*cols + j}",
                       (x1 + piece_width // 3, y1 + 2 * piece_height // 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    # Create scattered pieces on white background
    puzzle_image = np.ones((height + 200, width + 200, 3), dtype=np.uint8) * 255
    
    # Cut and scatter pieces
    np.random.seed(42)
    margin = 100
    
    for i in range(rows):
        for j in range(cols):
            # Extract piece
            y1 = i * piece_height
            y2 = (i + 1) * piece_height
            x1 = j * piece_width
            x2 = (j + 1) * piece_width
            piece = original[y1:y2, x1:x2].copy()
            
            # Random position
            new_x = np.random.randint(margin, width + 200 - piece_width - margin)
            new_y = np.random.randint(margin, height + 200 - piece_height - margin)
            
            # Place piece
            puzzle_image[new_y:new_y + piece_height, new_x:new_x + piece_width] = piece
            
            # Add border to make piece distinct
            cv2.rectangle(puzzle_image, (new_x, new_y), 
                         (new_x + piece_width, new_y + piece_height),
                         (0, 0, 0), 2)
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, puzzle_image)
    logger.info(f"Demo puzzle saved to: {output_path}")
    
    return output_path


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Jigsaw Puzzle Solver")
    parser.add_argument("--input", "-i", type=str, help="Input image path")
    parser.add_argument("--output", "-o", type=str, default="data/output",
                       help="Output directory")
    parser.add_argument("--rows", "-r", type=int, help="Number of puzzle rows")
    parser.add_argument("--cols", "-c", type=int, help="Number of puzzle columns")
    parser.add_argument("--demo", action="store_true", help="Create and solve demo puzzle")
    
    args = parser.parse_args()
    
    # Create demo if requested
    if args.demo:
        logger.info("Running demo mode")
        demo_path = create_demo_puzzle()
        args.input = demo_path
        args.rows = 3
        args.cols = 4
    
    # Check input
    if not args.input:
        print("Error: Please provide an input image with --input or use --demo")
        parser.print_help()
        return
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    
    # Run pipeline
    pipeline = PuzzleSolverPipeline(rows=args.rows, cols=args.cols)
    success = pipeline.solve_puzzle(args.input, args.output)
    
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
