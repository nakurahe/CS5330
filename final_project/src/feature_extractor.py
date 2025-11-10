"""
Feature Extraction Module
Extracts features from puzzle pieces for matching.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class EdgeType(Enum):
    """Types of puzzle piece edges."""
    FLAT = 0      # Straight edge (border piece)
    IN = 1        # Inward curve (socket)
    OUT = 2       # Outward curve (tab/knob)
    UNKNOWN = 3


class PieceType(Enum):
    """Types of puzzle pieces based on number of flat edges."""
    CORNER = 2      # 2 flat edges
    EDGE = 1        # 1 flat edge
    INTERIOR = 0    # 0 flat edges


class EdgeDescriptor:
    """Describes a single edge of a puzzle piece."""
    
    def __init__(self, edge_type: EdgeType, contour_points: np.ndarray, 
                 color_profile: np.ndarray, direction: str):
        self.edge_type = edge_type
        self.contour_points = contour_points  # Points along the edge
        self.color_profile = color_profile    # Color values along edge
        self.direction = direction            # 'top', 'right', 'bottom', 'left'
        self.descriptor = None                # Feature descriptor (SIFT/ORB)


class PieceFeatures:
    """Container for all features of a puzzle piece."""
    
    def __init__(self, piece_id: int):
        self.piece_id = piece_id
        self.piece_type = PieceType.INTERIOR
        self.edges = {'top': None, 'right': None, 'bottom': None, 'left': None}
        self.color_histogram = None
        self.dominant_colors = []
        self.keypoints = []
        self.descriptors = None


class FeatureExtractor:
    """Extracts features from puzzle pieces."""
    
    def __init__(self, flat_threshold_ratio: float = 0.5, adaptive: bool = True, use_boundary_detection: bool = False):
        """
        Initialize the feature extractor.
        
        Args:
            flat_threshold_ratio: Ratio of piece size to use as flatness threshold (default: 0.5)
                                 For grid-cut puzzles with photo content, higher values (0.4-0.6) work better
                                 For traditional jigsaw with smooth borders, lower values (0.1-0.2) work better
            adaptive: Use adaptive thresholds based on piece size
            use_boundary_detection: Use boundary-based detection (works for assembled puzzles in photo,
                                  not for scattered pieces)
        """
        self.sift = cv2.SIFT_create()
        self.flat_threshold_ratio = flat_threshold_ratio
        self.adaptive = adaptive
        self.use_boundary_detection = use_boundary_detection
        # Alternative: ORB (faster, free)
        # self.orb = cv2.ORB_create()
    
    def extract_features(self, pieces: List, image_shape: Tuple[int, int] = None) -> List[PieceFeatures]:
        """
        Extract features from all pieces.
        
        Args:
            pieces: List of PuzzlePiece objects
            image_shape: Optional (height, width) tuple of original image for boundary detection
            
        Returns:
            List of PieceFeatures
        """
        logger.info("Extracting features from pieces...")
        
        all_features = []
        for piece in pieces:
            features = self._extract_piece_features(piece, image_shape)
            all_features.append(features)
        
        logger.info(f"Extracted features from {len(all_features)} pieces")
        return all_features
    
    def _extract_piece_features(self, piece, image_shape: Tuple[int, int] = None) -> PieceFeatures:
        """
        Extract all features from a single piece.
        
        Args:
            piece: PuzzlePiece object
            image_shape: Optional (height, width) tuple of original image for boundary detection
            
        Returns:
            PieceFeatures object
        """
        features = PieceFeatures(piece.id)
        
        # Calculate adaptive threshold based on piece size
        if self.adaptive:
            piece_size = max(piece.bbox[2], piece.bbox[3])
            flat_threshold = piece_size * self.flat_threshold_ratio
            logger.debug(f"Piece #{piece.id}: size={piece_size}, threshold={flat_threshold:.1f}")
        else:
            flat_threshold = 5.0
        
        # Extract edge features with adaptive threshold and image shape for boundary detection
        features.edges = self._extract_edges(piece, flat_threshold, image_shape)
        
        # Classify piece type (corner, edge, interior)
        features.piece_type = self._classify_piece_type(features.edges)
        
        # Extract color histogram
        features.color_histogram = self._extract_color_histogram(piece)
        
        # Extract dominant colors
        features.dominant_colors = self._extract_dominant_colors(piece)
        
        # Extract keypoints and descriptors
        features.keypoints, features.descriptors = self._extract_keypoints(piece)
        
        return features
    
    def _extract_edges(self, piece, flat_threshold: float = 5.0, image_shape: Tuple[int, int] = None) -> Dict[str, EdgeDescriptor]:
        """
        Extract the four edges of a puzzle piece.
        
        Args:
            piece: PuzzlePiece object
            flat_threshold: Threshold for detecting flat edges
            image_shape: Optional (height, width) tuple of original image for boundary detection
            
        Returns:
            Dictionary of edge descriptors
        """
        # Get rotated rectangle to find piece orientation
        rect = cv2.minAreaRect(piece.contour)
        box = cv2.boxPoints(rect)
        box = box.astype(int)
        
        # Sort points to get consistent ordering (top-left, top-right, bottom-right, bottom-left)
        box = self._sort_box_points(box)
        
        edges = {}
        directions = ['top', 'right', 'bottom', 'left']
        
        for i, direction in enumerate(directions):
            # Get edge points
            start_idx = i
            end_idx = (i + 1) % 4
            
            # Extract contour segment for this edge
            edge_contour = self._extract_edge_contour(piece.contour, box[start_idx], box[end_idx])
            
            # Classify edge type (flat, in, out) with adaptive threshold and boundary check
            edge_type = self._classify_edge_type(piece, direction, edge_contour, flat_threshold, image_shape)
            
            # Extract color profile along edge
            color_profile = self._extract_edge_color_profile(piece, edge_contour)
            
            edges[direction] = EdgeDescriptor(edge_type, edge_contour, color_profile, direction)
        
        return edges
    
    def _sort_box_points(self, box: np.ndarray) -> np.ndarray:
        """
        Sort box points in consistent order: TL, TR, BR, BL.
        
        Args:
            box: Array of 4 corner points
            
        Returns:
            Sorted box points
        """
        # Sort by y-coordinate
        sorted_by_y = box[np.argsort(box[:, 1])]
        
        # Top two points
        top = sorted_by_y[:2]
        top = top[np.argsort(top[:, 0])]  # Sort by x: left, right
        
        # Bottom two points
        bottom = sorted_by_y[2:]
        bottom = bottom[np.argsort(bottom[:, 0])]  # Sort by x: left, right
        
        return np.vstack([top, bottom[::-1]])  # TL, TR, BR, BL
    
    def _extract_edge_contour(self, contour: np.ndarray, start_point: np.ndarray, 
                             end_point: np.ndarray) -> np.ndarray:
        """
        Extract the portion of contour between two points.
        
        Args:
            contour: Full piece contour
            start_point: Starting point
            end_point: Ending point
            
        Returns:
            Edge contour segment
        """
        # Simplified approach: use points within bounding region
        # For more accurate extraction, implement contour walking
        
        # Find closest contour points to start and end
        contour_flat = contour.reshape(-1, 2)
        
        start_dists = np.linalg.norm(contour_flat - start_point, axis=1)
        end_dists = np.linalg.norm(contour_flat - end_point, axis=1)
        
        start_idx = np.argmin(start_dists)
        end_idx = np.argmin(end_dists)
        
        # Extract segment (handle wrap-around)
        if start_idx <= end_idx:
            edge_points = contour_flat[start_idx:end_idx + 1]
        else:
            edge_points = np.vstack([contour_flat[start_idx:], contour_flat[:end_idx + 1]])
        
        return edge_points
    
    def _classify_edge_type(self, piece, direction: str, edge_contour: np.ndarray, 
                           flat_threshold: float = 5.0, image_shape: Tuple[int, int] = None) -> EdgeType:
        """
        Classify edge as FLAT, IN (socket), or OUT (tab) using hybrid approach.
        
        For grid puzzles, checks if edge is on image boundary first (primary method).
        For traditional jigsaw, uses contour analysis (secondary method).
        
        Args:
            piece: PuzzlePiece object
            direction: 'top', 'right', 'bottom', or 'left'
            edge_contour: Contour points of the edge
            flat_threshold: Threshold for detecting flat edges via contour analysis
            image_shape: Optional (height, width) tuple for boundary detection
            
        Returns:
            EdgeType
        """
        # Method 1: Boundary-based detection (for grid puzzles - only if enabled)
        if self.use_boundary_detection and image_shape is not None:
            if self._is_edge_on_boundary(piece, direction, image_shape):
                return EdgeType.FLAT
        
        # Method 2: Contour-based detection (for jigsaw puzzles or interior edges)
        if len(edge_contour) < 3:
            return EdgeType.UNKNOWN
        
        # Fit a line to the edge
        [vx, vy, x, y] = cv2.fitLine(edge_contour, cv2.DIST_L2, 0, 0.01, 0.01)
        
        # Calculate deviation from the line
        line_point = np.array([x[0], y[0]])
        line_direction = np.array([vx[0], vy[0]])
        
        deviations = []
        for point in edge_contour:
            # Calculate perpendicular distance
            v = point - line_point
            proj_length = np.dot(v, line_direction)
            proj_point = line_point + proj_length * line_direction
            deviation = np.linalg.norm(point - proj_point)
            
            # Determine sign (which side of the line)
            cross = np.cross(line_direction, v)
            deviation *= np.sign(cross)
            deviations.append(deviation)
        
        avg_deviation = np.mean(deviations)
        max_deviation = np.max(np.abs(deviations))
        
        # Use adaptive threshold
        if max_deviation < flat_threshold:
            return EdgeType.FLAT
        elif avg_deviation > 0:
            return EdgeType.OUT  # Tab/knob
        else:
            return EdgeType.IN   # Socket
    
    def _is_edge_on_boundary(self, piece, edge_direction: str, image_shape: Tuple[int, int], margin: int = 150) -> bool:
        """
        Check if piece edge is on image boundary (for grid puzzles).
        
        Args:
            piece: PuzzlePiece object
            edge_direction: 'top', 'right', 'bottom', or 'left'
            image_shape: (height, width) of original image
            margin: Pixel margin to consider as "on boundary" (default: 150 for scattered pieces)
            
        Returns:
            True if edge is on boundary (FLAT edge)
        """
        h, w = image_shape[:2]
        x, y, pw, ph = piece.bbox
        
        if edge_direction == 'top' and y < margin:
            return True  # Top edge touches top boundary
        elif edge_direction == 'bottom' and (y + ph) > (h - margin):
            return True  # Bottom edge touches bottom boundary
        elif edge_direction == 'left' and x < margin:
            return True  # Left edge touches left boundary
        elif edge_direction == 'right' and (x + pw) > (w - margin):
            return True  # Right edge touches right boundary
        
        return False  # Interior edge
    
    def _extract_edge_color_profile(self, piece, edge_contour: np.ndarray) -> np.ndarray:
        """
        Extract color values along the edge.
        
        Args:
            piece: PuzzlePiece object
            edge_contour: Edge contour points
            
        Returns:
            Array of color values
        """
        if len(edge_contour) == 0:
            return np.array([])
        
        color_profile = []
        
        x_offset, y_offset = piece.bbox[0], piece.bbox[1]
        
        for point in edge_contour:
            # Adjust coordinates to piece image space
            x = int(point[0] - x_offset)
            y = int(point[1] - y_offset)
            
            # Bounds checking
            if 0 <= y < piece.image.shape[0] and 0 <= x < piece.image.shape[1]:
                color = piece.image[y, x]
                color_profile.append(color)
        
        return np.array(color_profile)
    
    def _classify_piece_type(self, edges: Dict[str, EdgeDescriptor]) -> PieceType:
        """
        Classify piece as corner, edge, or interior based on flat edges.
        
        Args:
            edges: Dictionary of edge descriptors
            
        Returns:
            PieceType
        """
        flat_count = sum(1 for edge in edges.values() if edge.edge_type == EdgeType.FLAT)
        
        if flat_count >= 2:
            return PieceType.CORNER
        elif flat_count == 1:
            return PieceType.EDGE
        else:
            return PieceType.INTERIOR
    
    def _extract_color_histogram(self, piece) -> np.ndarray:
        """
        Extract color histogram from piece.
        
        Args:
            piece: PuzzlePiece object
            
        Returns:
            Color histogram
        """
        # Extract only the masked region
        masked_img = cv2.bitwise_and(piece.image, piece.image, mask=piece.mask)
        
        # Calculate histogram for each channel
        hist_b = cv2.calcHist([masked_img], [0], piece.mask, [32], [0, 256])
        hist_g = cv2.calcHist([masked_img], [1], piece.mask, [32], [0, 256])
        hist_r = cv2.calcHist([masked_img], [2], piece.mask, [32], [0, 256])
        
        # Normalize
        hist_b = cv2.normalize(hist_b, hist_b).flatten()
        hist_g = cv2.normalize(hist_g, hist_g).flatten()
        hist_r = cv2.normalize(hist_r, hist_r).flatten()
        
        # Concatenate
        histogram = np.concatenate([hist_b, hist_g, hist_r])
        
        return histogram
    
    def _extract_dominant_colors(self, piece, n_colors: int = 3) -> List[Tuple[int, int, int]]:
        """
        Extract dominant colors from piece using K-means.
        
        Args:
            piece: PuzzlePiece object
            n_colors: Number of dominant colors to extract
            
        Returns:
            List of RGB colors
        """
        # Get masked pixels
        masked_img = cv2.bitwise_and(piece.image, piece.image, mask=piece.mask)
        
        # Reshape to pixel list
        pixels = masked_img.reshape(-1, 3)
        
        # Remove black pixels (masked areas)
        pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]
        
        if len(pixels) < n_colors:
            return []
        
        # K-means clustering
        pixels = np.float32(pixels)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert to integers
        dominant_colors = [tuple(map(int, color)) for color in centers]
        
        return dominant_colors
    
    def _extract_keypoints(self, piece) -> Tuple[List, np.ndarray]:
        """
        Extract SIFT keypoints and descriptors.
        
        Args:
            piece: PuzzlePiece object
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(piece.image, cv2.COLOR_BGR2GRAY)
        
        # Detect keypoints only in masked region
        keypoints, descriptors = self.sift.detectAndCompute(gray, piece.mask)
        
        return keypoints, descriptors


def main():
    """Test feature extraction."""
    pass


if __name__ == "__main__":
    main()
