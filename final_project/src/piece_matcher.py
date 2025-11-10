"""
Piece Matching Module
Matches puzzle pieces based on edge compatibility.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.spatial.distance import cosine, euclidean
import logging

logger = logging.getLogger(__name__)


class EdgeMatch:
    """Represents a potential match between two piece edges."""
    
    def __init__(self, piece1_id: int, edge1: str, piece2_id: int, edge2: str, score: float):
        self.piece1_id = piece1_id
        self.edge1 = edge1  # 'top', 'right', 'bottom', 'left'
        self.piece2_id = piece2_id
        self.edge2 = edge2
        self.score = score  # Higher is better (0-1)
    
    def __repr__(self):
        return f"Match(P{self.piece1_id}:{self.edge1} <-> P{self.piece2_id}:{self.edge2}, score={self.score:.3f})"


class PieceMatcher:
    """Matches puzzle pieces based on edge features."""
    
    def __init__(self, shape_weight: float = 0.3, color_weight: float = 0.5, 
                 texture_weight: float = 0.2):
        """
        Initialize the matcher.
        
        Args:
            shape_weight: Weight for shape similarity
            color_weight: Weight for color similarity
            texture_weight: Weight for texture similarity
        """
        self.shape_weight = shape_weight
        self.color_weight = color_weight
        self.texture_weight = texture_weight
        
        # Normalize weights
        total = shape_weight + color_weight + texture_weight
        self.shape_weight /= total
        self.color_weight /= total
        self.texture_weight /= total
    
    def find_all_matches(self, features_list: List) -> List[EdgeMatch]:
        """
        Find all potential edge matches between pieces.
        
        Args:
            features_list: List of PieceFeatures objects
            
        Returns:
            List of EdgeMatch objects sorted by score
        """
        logger.info("Finding edge matches between pieces...")
        
        matches = []
        
        # Compare each piece with every other piece
        for i, features1 in enumerate(features_list):
            for j, features2 in enumerate(features_list):
                if i >= j:  # Avoid duplicate comparisons and self-comparison
                    continue
                
                # Compare all edge combinations
                piece_matches = self._compare_pieces(features1, features2)
                matches.extend(piece_matches)
        
        # Sort by score (descending)
        matches.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"Found {len(matches)} potential matches")
        return matches
    
    def _compare_pieces(self, features1, features2) -> List[EdgeMatch]:
        """
        Compare two pieces and find potential edge matches.
        
        Args:
            features1: PieceFeatures of first piece
            features2: PieceFeatures of second piece
            
        Returns:
            List of EdgeMatch objects for this pair
        """
        matches = []
        
        # Define compatible edge pairs
        # For pieces to connect: one must have OUT/IN, the other IN/OUT
        # Or use opposite edges (top-bottom, left-right)
        compatible_edges = [
            ('top', 'bottom'),
            ('bottom', 'top'),
            ('left', 'right'),
            ('right', 'left')
        ]
        
        for edge1_name, edge2_name in compatible_edges:
            edge1 = features1.edges.get(edge1_name)
            edge2 = features2.edges.get(edge2_name)
            
            if edge1 is None or edge2 is None:
                continue
            
            # Check edge type compatibility
            if not self._are_edges_compatible(edge1, edge2):
                continue
            
            # Calculate match score
            score = self._calculate_match_score(edge1, edge2)
            
            if score > 0.3:  # Threshold for considering a match
                match = EdgeMatch(features1.piece_id, edge1_name, 
                                features2.piece_id, edge2_name, score)
                matches.append(match)
        
        return matches
    
    def _are_edges_compatible(self, edge1, edge2) -> bool:
        """
        Check if two edges are compatible for matching.
        
        Args:
            edge1: EdgeDescriptor
            edge2: EdgeDescriptor
            
        Returns:
            True if edges are compatible
        """
        from src.feature_extractor import EdgeType
        
        # Flat edges can only match with flat edges
        if edge1.edge_type == EdgeType.FLAT:
            return edge2.edge_type == EdgeType.FLAT
        
        if edge2.edge_type == EdgeType.FLAT:
            return edge1.edge_type == EdgeType.FLAT
        
        # IN edge should match with OUT edge and vice versa
        if edge1.edge_type == EdgeType.IN:
            return edge2.edge_type == EdgeType.OUT
        
        if edge1.edge_type == EdgeType.OUT:
            return edge2.edge_type == EdgeType.IN
        
        return False
    
    def _calculate_match_score(self, edge1, edge2) -> float:
        """
        Calculate similarity score between two edges.
        
        Args:
            edge1: EdgeDescriptor
            edge2: EdgeDescriptor
            
        Returns:
            Match score (0-1, higher is better)
        """
        scores = []
        
        # Shape similarity
        shape_score = self._calculate_shape_similarity(edge1, edge2)
        scores.append(self.shape_weight * shape_score)
        
        # Color similarity
        color_score = self._calculate_color_similarity(edge1, edge2)
        scores.append(self.color_weight * color_score)
        
        # Texture similarity (using color profile variation)
        texture_score = self._calculate_texture_similarity(edge1, edge2)
        scores.append(self.texture_weight * texture_score)
        
        total_score = sum(scores)
        return total_score
    
    def _calculate_shape_similarity(self, edge1, edge2) -> float:
        """
        Calculate shape similarity between two edges.
        
        Args:
            edge1: EdgeDescriptor
            edge2: EdgeDescriptor
            
        Returns:
            Similarity score (0-1)
        """
        contour1 = edge1.contour_points
        contour2 = edge2.contour_points
        
        if len(contour1) < 2 or len(contour2) < 2:
            return 0.0
        
        # Normalize contours to same length
        n_points = min(len(contour1), len(contour2), 50)
        
        if len(contour1) > n_points:
            indices1 = np.linspace(0, len(contour1) - 1, n_points, dtype=int)
            contour1 = contour1[indices1]
        
        if len(contour2) > n_points:
            indices2 = np.linspace(0, len(contour2) - 1, n_points, dtype=int)
            contour2 = contour2[indices2]
        
        # For matching edges, reverse one contour (they should be mirror images)
        contour2_reversed = contour2[::-1]
        
        # Normalize position and scale
        contour1_norm = self._normalize_contour(contour1)
        contour2_norm = self._normalize_contour(contour2_reversed)
        
        # Calculate Hausdorff distance or mean distance
        try:
            distances = []
            for p1 in contour1_norm:
                min_dist = np.min(np.linalg.norm(contour2_norm - p1, axis=1))
                distances.append(min_dist)
            
            avg_distance = np.mean(distances)
            
            # Convert distance to similarity (0 distance = 1 similarity)
            # Use exponential decay
            similarity = np.exp(-avg_distance / 10.0)
            
            return float(similarity)
        except:
            return 0.0
    
    def _normalize_contour(self, contour: np.ndarray) -> np.ndarray:
        """
        Normalize contour to standard position and scale.
        
        Args:
            contour: Contour points
            
        Returns:
            Normalized contour
        """
        if len(contour) == 0:
            return contour
        
        # Center the contour
        centroid = np.mean(contour, axis=0)
        centered = contour - centroid
        
        # Scale to unit variance
        scale = np.std(centered)
        if scale > 0:
            normalized = centered / scale
        else:
            normalized = centered
        
        return normalized
    
    def _calculate_color_similarity(self, edge1, edge2) -> float:
        """
        Calculate color similarity between two edges.
        
        Args:
            edge1: EdgeDescriptor
            edge2: EdgeDescriptor
            
        Returns:
            Similarity score (0-1)
        """
        color1 = edge1.color_profile
        color2 = edge2.color_profile
        
        if len(color1) == 0 or len(color2) == 0:
            return 0.0
        
        # Sample colors evenly from both edges
        n_samples = min(len(color1), len(color2), 20)
        
        if len(color1) > n_samples:
            indices1 = np.linspace(0, len(color1) - 1, n_samples, dtype=int)
            color1 = color1[indices1]
        
        if len(color2) > n_samples:
            indices2 = np.linspace(0, len(color2) - 1, n_samples, dtype=int)
            color2 = color2[indices2]
        
        # Reverse one edge for matching
        color2 = color2[::-1]
        
        # Calculate color distance (in BGR space)
        try:
            color_diff = np.mean(np.linalg.norm(color1.astype(float) - color2.astype(float), axis=1))
            
            # Normalize (max possible difference is sqrt(255^2 * 3) ≈ 441)
            similarity = 1.0 - (color_diff / 441.0)
            similarity = max(0.0, min(1.0, similarity))
            
            return float(similarity)
        except:
            return 0.0
    
    def _calculate_texture_similarity(self, edge1, edge2) -> float:
        """
        Calculate texture similarity based on color variation patterns.
        
        Args:
            edge1: EdgeDescriptor
            edge2: EdgeDescriptor
            
        Returns:
            Similarity score (0-1)
        """
        color1 = edge1.color_profile
        color2 = edge2.color_profile
        
        if len(color1) < 3 or len(color2) < 3:
            return 0.0
        
        try:
            # Calculate gradients (color changes along edge)
            grad1 = np.diff(color1.astype(float), axis=0)
            grad2 = np.diff(color2.astype(float), axis=0)
            
            # Sample gradients
            n_samples = min(len(grad1), len(grad2), 15)
            
            if len(grad1) > n_samples:
                indices1 = np.linspace(0, len(grad1) - 1, n_samples, dtype=int)
                grad1 = grad1[indices1]
            
            if len(grad2) > n_samples:
                indices2 = np.linspace(0, len(grad2) - 1, n_samples, dtype=int)
                grad2 = grad2[indices2]
            
            # Reverse for matching
            grad2 = grad2[::-1]
            
            # Calculate correlation
            grad1_flat = grad1.flatten()
            grad2_flat = grad2.flatten()
            
            # Ensure same length
            min_len = min(len(grad1_flat), len(grad2_flat))
            grad1_flat = grad1_flat[:min_len]
            grad2_flat = grad2_flat[:min_len]
            
            # Normalize
            grad1_norm = grad1_flat - np.mean(grad1_flat)
            grad2_norm = grad2_flat - np.mean(grad2_flat)
            
            std1 = np.std(grad1_norm)
            std2 = np.std(grad2_norm)
            
            if std1 > 0 and std2 > 0:
                correlation = np.dot(grad1_norm, grad2_norm) / (len(grad1_norm) * std1 * std2)
                # Map correlation from [-1, 1] to [0, 1]
                similarity = (correlation + 1.0) / 2.0
                similarity = max(0.0, min(1.0, similarity))
            else:
                similarity = 0.5  # Neutral if no variation
            
            return float(similarity)
        except:
            return 0.0
    
    def get_best_match_for_edge(self, piece_id: int, edge_name: str, 
                                matches: List[EdgeMatch]) -> Optional[EdgeMatch]:
        """
        Get the best match for a specific piece edge.
        
        Args:
            piece_id: ID of the piece
            edge_name: Name of the edge ('top', 'right', 'bottom', 'left')
            matches: List of all matches
            
        Returns:
            Best EdgeMatch or None
        """
        relevant_matches = [
            m for m in matches 
            if (m.piece1_id == piece_id and m.edge1 == edge_name) or
               (m.piece2_id == piece_id and m.edge2 == edge_name)
        ]
        
        if not relevant_matches:
            return None
        
        return relevant_matches[0]  # Already sorted by score


def main():
    """Test piece matching."""
    pass


if __name__ == "__main__":
    main()
