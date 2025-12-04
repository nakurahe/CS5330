"""
Visualization Module
Creates annotated images showing puzzle solution with assembly instructions.
"""

import cv2
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class PuzzleVisualizer:
    """Creates visualizations of puzzle solution."""
    
    def __init__(self):
        """Initialize the visualizer."""
        self.colors = [
            (0, 0, 255),    # Red
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Cyan
            (128, 0, 128),  # Purple
            (0, 165, 255),  # Orange
        ]
    
    def create_solution_visualization(self, original_image: np.ndarray, 
                                      pieces: List, grid, 
                                      assembly_steps: List,
                                      output_prefix: str = "output") -> List[str]:
        """
        Create all visualization outputs.
        
        Args:
            original_image: Original image with scattered pieces
            pieces: List of PuzzlePiece objects
            grid: PuzzleGrid with solution
            assembly_steps: List of AssemblyStep objects
            output_prefix: Prefix for output files
            
        Returns:
            List of output file paths
        """
        logger.info("Creating visualizations...")
        
        output_files = []
        
        # 1. Annotated input image (pieces numbered)
        annotated_input = self.annotate_input_image(original_image, pieces)
        input_path = f"{output_prefix}_1_input_labeled.jpg"
        cv2.imwrite(input_path, annotated_input)
        output_files.append(input_path)
        logger.info(f"Saved: {input_path}")
        
        # 2. Reconstructed puzzle
        reconstructed = self.reconstruct_puzzle(pieces, grid)
        if reconstructed is not None:
            recon_path = f"{output_prefix}_2_reconstructed.jpg"
            cv2.imwrite(recon_path, reconstructed)
            output_files.append(recon_path)
            logger.info(f"Saved: {recon_path}")
        
        # 3. Step-by-step assembly visualization
        step_images = self.create_assembly_steps(pieces, assembly_steps, grid)
        for i, step_img in enumerate(step_images[:10]):  # Save first 10 steps
            step_path = f"{output_prefix}_3_step_{i+1:02d}.jpg"
            cv2.imwrite(step_path, step_img)
            output_files.append(step_path)
        logger.info(f"Saved {len(step_images)} step images")
        
        # 4. Assembly instructions text
        instructions_path = f"{output_prefix}_4_instructions.txt"
        self.save_instructions(assembly_steps, instructions_path)
        output_files.append(instructions_path)
        logger.info(f"Saved: {instructions_path}")
        
        return output_files
    
    def annotate_input_image(self, image: np.ndarray, pieces: List) -> np.ndarray:
        """
        Annotate the input image with piece numbers and highlights.
        
        Args:
            image: Original image
            pieces: List of PuzzlePiece objects
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        for piece in pieces:
            # Get color for this piece
            color = self.colors[piece.id % len(self.colors)]
            
            # Draw contour
            cv2.drawContours(annotated, [piece.contour], 0, color, 3)
            
            # Draw filled circle at centroid
            cx, cy = piece.centroid
            cv2.circle(annotated, (int(cx), int(cy)), 20, color, -1)
            
            # Draw piece number
            text = f"{piece.id}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            
            # Get text size for centering
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = int(cx - text_width / 2)
            text_y = int(cy + text_height / 2)
            
            cv2.putText(annotated, text, (text_x, text_y), font, 
                       font_scale, (255, 255, 255), thickness)
        
        return annotated
    
    def reconstruct_puzzle(self, pieces: List, grid) -> Optional[np.ndarray]:
        """
        Reconstruct the solved puzzle from pieces.
        
        Args:
            pieces: List of PuzzlePiece objects
            grid: PuzzleGrid with solution
            
        Returns:
            Reconstructed puzzle image, or None if grid is empty
        """
        if len(grid.grid) == 0:
            logger.warning("Grid is empty, cannot reconstruct puzzle")
            return None
        
        # Create piece lookup
        piece_dict = {p.id: p for p in pieces}
        
        # Estimate piece size (use average)
        piece_sizes = [p.bbox[2:] for p in pieces]  # (width, height)
        avg_width = int(np.mean([s[0] for s in piece_sizes]))
        avg_height = int(np.mean([s[1] for s in piece_sizes]))
        
        # Create output image
        output_height = grid.rows * avg_height
        output_width = grid.cols * avg_width
        reconstructed = np.ones((output_height, output_width, 3), dtype=np.uint8) * 200
        
        # Place each piece
        for (row, col), piece_id in grid.grid.items():
            if piece_id not in piece_dict:
                continue
            
            piece = piece_dict[piece_id]
            
            # Calculate position in reconstructed image
            y_start = row * avg_height
            x_start = col * avg_width
            
            # Resize piece to fit
            piece_resized = cv2.resize(piece.image, (avg_width, avg_height))
            mask_resized = cv2.resize(piece.mask, (avg_width, avg_height))
            
            # Blend piece into reconstructed image
            y_end = min(y_start + avg_height, output_height)
            x_end = min(x_start + avg_width, output_width)
            
            piece_h = y_end - y_start
            piece_w = x_end - x_start
            
            piece_crop = piece_resized[:piece_h, :piece_w]
            mask_crop = mask_resized[:piece_h, :piece_w]
            
            # Use mask to blend
            for c in range(3):
                reconstructed[y_start:y_end, x_start:x_end, c] = \
                    np.where(mask_crop > 0, piece_crop[:, :, c], 
                            reconstructed[y_start:y_end, x_start:x_end, c])
            
            # Draw grid lines
            cv2.rectangle(reconstructed, (x_start, y_start), (x_end-1, y_end-1), 
                         (100, 100, 100), 1)
            
            # Draw piece number in corner
            cv2.putText(reconstructed, str(piece_id), (x_start + 5, y_start + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        return reconstructed
    
    def create_assembly_steps(self, pieces: List, assembly_steps: List, 
                              grid) -> List[np.ndarray]:
        """
        Create step-by-step assembly visualization.
        
        Shows partial reconstruction with the current piece highlighted.
        
        Args:
            pieces: List of PuzzlePiece objects
            assembly_steps: List of AssemblyStep objects
            grid: PuzzleGrid
            
        Returns:
            List of step images
        """
        step_images = []
        piece_dict = {p.id: p for p in pieces}
        
        # Estimate piece size
        piece_sizes = [p.bbox[2:] for p in pieces]
        avg_width = int(np.mean([s[0] for s in piece_sizes]))
        avg_height = int(np.mean([s[1] for s in piece_sizes]))
        
        # Output dimensions
        output_height = grid.rows * avg_height
        output_width = grid.cols * avg_width
        
        for step_idx, step in enumerate(assembly_steps):
            # Create reconstruction image
            canvas = np.ones((output_height, output_width, 3), dtype=np.uint8) * 200
            
            # Draw grid lines
            for i in range(grid.rows + 1):
                y = i * avg_height
                cv2.line(canvas, (0, y), (output_width, y), (150, 150, 150), 1)
            for j in range(grid.cols + 1):
                x = j * avg_width
                cv2.line(canvas, (x, 0), (x, output_height), (150, 150, 150), 1)
            
            # Place pieces up to current step
            for prev_idx, prev_step in enumerate(assembly_steps[:step_idx + 1]):
                row, col = prev_step.row, prev_step.col
                piece_id = prev_step.piece_id
                
                if piece_id not in piece_dict:
                    continue
                
                piece = piece_dict[piece_id]
                
                y_start = row * avg_height
                x_start = col * avg_width
                
                piece_resized = cv2.resize(piece.image, (avg_width, avg_height))
                mask_resized = cv2.resize(piece.mask, (avg_width, avg_height))
                
                y_end = min(y_start + avg_height, output_height)
                x_end = min(x_start + avg_width, output_width)
                
                piece_h = y_end - y_start
                piece_w = x_end - x_start
                
                piece_crop = piece_resized[:piece_h, :piece_w]
                mask_crop = mask_resized[:piece_h, :piece_w]
                
                for c in range(3):
                    canvas[y_start:y_end, x_start:x_end, c] = \
                        np.where(mask_crop > 0, piece_crop[:, :, c],
                                canvas[y_start:y_end, x_start:x_end, c])
                
                # Highlight the current piece with a colored border
                if prev_idx == step_idx:
                    cv2.rectangle(canvas, (x_start, y_start), (x_end - 1, y_end - 1),
                                 (0, 0, 255), 4)  # Red border for current piece
            
            # Add piece number label on current piece
            target_y = int((step.row + 0.5) * avg_height)
            target_x = int((step.col + 0.5) * avg_width)
            cv2.putText(canvas, f"#{step.piece_id}", (target_x - 20, target_y + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add text instruction at top
            instruction = f"Step {step_idx + 1}: Place piece #{step.piece_id} at ({step.row}, {step.col})"
            cv2.putText(canvas, instruction, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            step_images.append(canvas)
        
        return step_images
    
    def save_instructions(self, assembly_steps: List, output_path: str):
        """
        Save assembly instructions as text file.
        
        Args:
            assembly_steps: List of AssemblyStep objects
            output_path: Path to save instructions
        """
        with open(output_path, 'w') as f:
            f.write("PUZZLE ASSEMBLY INSTRUCTIONS\n")
            f.write("=" * 50 + "\n\n")
            
            for i, step in enumerate(assembly_steps):
                f.write(f"Step {i + 1}:\n")
                f.write(f"  - Take piece #{step.piece_id}\n")
                f.write(f"  - Place at position: Row {step.row}, Column {step.col}\n")
                if step.reason:
                    f.write(f"  - Reason: {step.reason}\n")
                f.write("\n")
            
            f.write("=" * 50 + "\n")
            f.write(f"Total steps: {len(assembly_steps)}\n")
