"""
Puzzle Piece Detection and Segmentation Module
Detects and extracts individual puzzle pieces from an input image.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PuzzlePiece:
    """Represents a single puzzle piece with its properties."""
    
    def __init__(self, piece_id: int, contour: np.ndarray, image: np.ndarray, 
                 mask: np.ndarray, bbox: Tuple[int, int, int, int]):
        self.id = piece_id
        self.contour = contour
        self.image = image  # Cropped piece image
        self.mask = mask    # Binary mask
        self.bbox = bbox    # (x, y, w, h)
        self.centroid = self._calculate_centroid()
        self.angle = 0
        
    def _calculate_centroid(self) -> Tuple[float, float]:
        """Calculate the centroid of the piece."""
        M = cv2.moments(self.contour)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx, cy = self.bbox[0] + self.bbox[2] // 2, self.bbox[1] + self.bbox[3] // 2
        return (cx, cy)


class PieceDetector:
    """Detects and segments puzzle pieces from an image."""
    
    def __init__(self, min_area: int = None, max_area: int = None, adaptive: bool = True):
        """
        Initialize the detector.
        
        Args:
            min_area: Minimum contour area to consider as a piece (None = auto-detect)
            max_area: Maximum contour area to consider as a piece (None = auto-detect)
            adaptive: Use adaptive area thresholds based on image size
        """
        self.min_area = min_area
        self.max_area = max_area
        self.adaptive = adaptive
        
    def detect_pieces(self, image: np.ndarray) -> List[PuzzlePiece]:
        """
        Detect and extract all puzzle pieces from the image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of PuzzlePiece objects
        """
        logger.info("Starting piece detection...")
        
        # Calculate adaptive thresholds if needed
        if self.adaptive and (self.min_area is None or self.max_area is None):
            img_area = image.shape[0] * image.shape[1]
            if self.min_area is None:
                # Min area: at least 0.5% of image for large images, or 500 pixels
                self.min_area = max(500, int(img_area * 0.005))
            if self.max_area is None:
                # Max area: up to 10% of image
                self.max_area = int(img_area * 0.1)
            logger.info(f"Using adaptive area thresholds: min={self.min_area}, max={self.max_area}")
        elif self.min_area is None:
            self.min_area = 500
        elif self.max_area is None:
            self.max_area = 50000
        
        # Preprocess image
        processed = self._preprocess_image(image)
        
        # Find contours
        contours = self._find_contours(processed)
        
        # Filter and extract pieces
        pieces = self._extract_pieces(image, contours)
        
        logger.info(f"Detected {len(pieces)} puzzle pieces")
        return pieces
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image for better segmentation.
        
        Args:
            image: Input BGR image
            
        Returns:
            Binary mask of potential pieces
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use adaptive thresholding for better separation
        # This works well even with varying lighting conditions
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Alternative: Otsu's thresholding if background is uniform
        # _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return binary
    
    def _find_contours(self, binary: np.ndarray) -> List[np.ndarray]:
        """
        Find contours in the binary image.
        
        Args:
            binary: Binary image
            
        Returns:
            List of contours
        """
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area < area < self.max_area:
                valid_contours.append(contour)
        
        logger.info(f"Found {len(valid_contours)} valid contours")
        return valid_contours
    
    def _extract_pieces(self, image: np.ndarray, contours: List[np.ndarray]) -> List[PuzzlePiece]:
        """
        Extract individual pieces from contours.
        
        Args:
            image: Original BGR image
            contours: List of piece contours
            
        Returns:
            List of PuzzlePiece objects
        """
        pieces = []
        
        for i, contour in enumerate(contours):
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Add padding
            padding = 5
            x_pad = max(0, x - padding)
            y_pad = max(0, y - padding)
            w_pad = min(image.shape[1] - x_pad, w + 2 * padding)
            h_pad = min(image.shape[0] - y_pad, h + 2 * padding)
            
            # Extract piece image
            piece_img = image[y_pad:y_pad + h_pad, x_pad:x_pad + w_pad].copy()
            
            # Create mask for the piece
            mask = np.zeros((h_pad, w_pad), dtype=np.uint8)
            
            # Adjust contour coordinates to the cropped region
            adjusted_contour = contour - [x_pad, y_pad]
            cv2.drawContours(mask, [adjusted_contour], 0, 255, -1)
            
            # Apply mask to piece image
            piece_img_masked = cv2.bitwise_and(piece_img, piece_img, mask=mask)
            
            # Create PuzzlePiece object
            piece = PuzzlePiece(i, contour, piece_img_masked, mask, (x_pad, y_pad, w_pad, h_pad))
            pieces.append(piece)
        
        return pieces
    
    def visualize_detection(self, image: np.ndarray, pieces: List[PuzzlePiece]) -> np.ndarray:
        """
        Visualize detected pieces with bounding boxes and IDs.
        
        Args:
            image: Original image
            pieces: List of detected pieces
            
        Returns:
            Annotated image
        """
        vis_image = image.copy()
        
        for piece in pieces:
            # Draw contour
            cv2.drawContours(vis_image, [piece.contour], 0, (0, 255, 0), 2)
            
            # Draw bounding box
            x, y, w, h = piece.bbox
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Draw centroid
            cx, cy = piece.centroid
            cv2.circle(vis_image, (int(cx), int(cy)), 5, (0, 0, 255), -1)
            
            # Draw ID
            cv2.putText(vis_image, f"#{piece.id}", (int(cx) - 15, int(cy) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_image


def main():
    """Test the piece detector."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python piece_detector.py <image_path>")
        return
    
    # Load image
    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Detect pieces
    detector = PieceDetector()
    pieces = detector.detect_pieces(image)
    
    # Visualize
    vis_image = detector.visualize_detection(image, pieces)
    
    # Save and show result
    output_path = "detected_pieces.jpg"
    cv2.imwrite(output_path, vis_image)
    print(f"Detected {len(pieces)} pieces. Result saved to {output_path}")
    
    cv2.imshow("Detected Pieces", vis_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
