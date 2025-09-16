import gradio as gr
import numpy as np
import cv2
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from skimage.metrics import structural_similarity as ssim
from typing import Tuple, List
import colorsys

class MosaicGenerator:
    """Interactive Image Mosaic Generator with adaptive gridding and multiple tile options."""
    
    def __init__(self):
        self.original_image = None
        self.mosaic_image = None
        self.tile_cache = {}
        self.pattern_tiles = self.generate_pattern_tiles()
        
    def generate_pattern_tiles(self, size: int = 32) -> dict:
        """Generate a set of pattern tiles for mosaic creation."""
        patterns = {}
        
        # Solid color (will be tinted based on average color)
        patterns['solid'] = np.ones((size, size, 3), dtype=np.uint8) * 255
        
        # Gradient patterns
        # Horizontal gradient: each row is the same, columns vary
        gradient_h_2d = np.tile(np.linspace(0, 255, size).reshape(1, -1), (size, 1))
        patterns['gradient_h'] = np.stack([gradient_h_2d] * 3, axis=-1).astype(np.uint8)
        
        # Vertical gradient: each column is the same, rows vary
        gradient_v_2d = np.tile(np.linspace(0, 255, size).reshape(-1, 1), (1, size))
        patterns['gradient_v'] = np.stack([gradient_v_2d] * 3, axis=-1).astype(np.uint8)
        
        # Diagonal gradient
        diag = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                diag[i, j] = (i + j) / (2 * size - 2) * 255
        patterns['gradient_diag'] = diag.astype(np.uint8)
        
        # Circle pattern
        center = size // 2
        circle = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                circle[i, j] = 255 if dist <= size//3 else 0
        patterns['circle'] = circle.astype(np.uint8)
        
        # Cross pattern
        cross = np.zeros((size, size))
        cross[size//2-2:size//2+2, :] = 255
        cross[:, size//2-2:size//2+2] = 255
        patterns['cross'] = cross.astype(np.uint8)
        
        # Convert single channel patterns to 3-channel
        for key in list(patterns.keys()):
            if len(patterns[key].shape) == 2:
                patterns[key] = np.stack([patterns[key]] * 3, axis=-1)
                
        return patterns
    
    def quantize_colors(self, image: np.ndarray, n_colors: int) -> np.ndarray:
        """Apply K-means color quantization to reduce color palette."""
        # Reshape image to be a list of pixels
        pixels = image.reshape(-1, 3)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Replace each pixel with its nearest cluster center
        new_colors = kmeans.cluster_centers_[kmeans.labels_]
        
        # Reshape back to original image shape
        return new_colors.reshape(image.shape).astype(np.uint8)
    
    def calculate_cell_variance(self, cell: np.ndarray) -> float:
        """Calculate the variance of a cell to determine if it needs subdivision."""
        # Convert to grayscale for variance calculation
        gray = cv2.cvtColor(cell, cv2.COLOR_RGB2GRAY)
        return np.std(gray)
    
    def adaptive_grid_segmentation(self, image: np.ndarray, base_size: int, 
                                 variance_threshold: float, max_depth: int = 3) -> List[dict]:
        """Perform adaptive grid segmentation based on image complexity."""
        h, w = image.shape[:2]
        segments = []
        
        def subdivide(x: int, y: int, size: int, depth: int):
            """Recursively subdivide cells based on variance threshold."""
            if depth >= max_depth or size < 8:
                # Add the segment
                segments.append({
                    'x': x, 'y': y, 'size': size,
                    'cell': image[y:min(y+size, h), x:min(x+size, w)]
                })
                return
            
            # Extract cell
            cell = image[y:min(y+size, h), x:min(x+size, w)]
            
            # Calculate variance
            if cell.size > 0:
                variance = self.calculate_cell_variance(cell)
                
                if variance > variance_threshold:
                    # Subdivide into 4 quadrants
                    half_size = size // 2
                    subdivide(x, y, half_size, depth + 1)
                    subdivide(x + half_size, y, half_size, depth + 1)
                    subdivide(x, y + half_size, half_size, depth + 1)
                    subdivide(x + half_size, y + half_size, half_size, depth + 1)
                else:
                    # Keep as single segment
                    segments.append({
                        'x': x, 'y': y, 'size': size,
                        'cell': cell
                    })
        
        # Start subdivision from base grid
        for y in range(0, h, base_size):
            for x in range(0, w, base_size):
                subdivide(x, y, base_size, 0)
        
        return segments
    
    def get_average_color(self, cell: np.ndarray) -> Tuple[int, int, int]:
        """Calculate the average color of a cell."""
        if cell.size == 0:
            return (128, 128, 128)
        return tuple(np.mean(cell.reshape(-1, 3), axis=0).astype(int))
    
    def create_color_tile(self, color: Tuple[int, int, int], size: int) -> np.ndarray:
        """Create a solid color tile."""
        tile = np.ones((size, size, 3), dtype=np.uint8)
        tile[:, :] = color
        return tile
    
    def create_pattern_tile(self, pattern_name: str, color: Tuple[int, int, int], 
                       size: int) -> np.ndarray:
        """Create a pattern tile tinted with the specified color using HSV for better results."""
        # Get base pattern
        if pattern_name not in self.pattern_tiles:
            pattern_name = 'solid'
        
        base_pattern = self.pattern_tiles[pattern_name]
        
        # Resize if necessary
        if base_pattern.shape[0] != size:
            base_pattern = cv2.resize(base_pattern, (size, size))
        
        # Convert to HSV for better color tinting
        hsv_color = colorsys.rgb_to_hsv(color[0]/255, color[1]/255, color[2]/255)
        
        # Convert pattern to HSV
        pattern_hsv = cv2.cvtColor(base_pattern, cv2.COLOR_RGB2HSV).astype(float)
        
        # Apply the hue and saturation from the target color, keep pattern's value (brightness)
        pattern_hsv[:, :, 0] = hsv_color[0] * 179  # OpenCV uses 0-179 for hue
        pattern_hsv[:, :, 1] = hsv_color[1] * 255 * (pattern_hsv[:, :, 2] / 255)  # Scale saturation by brightness
        
        # Convert back to RGB
        tinted = cv2.cvtColor(pattern_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        return tinted
    
    def create_mini_image_tile(self, cell: np.ndarray, size: int) -> np.ndarray:
        """Create a tile by downsampling the original cell."""
        # Downsample to create pixelated effect
        small_size = max(4, size // 8)
        small = cv2.resize(cell, (small_size, small_size), interpolation=cv2.INTER_AREA)
        # Resize back to tile size for blocky effect
        tile = cv2.resize(small, (size, size), interpolation=cv2.INTER_NEAREST)
        return tile
    
    def build_mosaic(self, image: np.ndarray, segments: List[dict], 
                    tile_type: str, pattern_type: str = 'solid') -> np.ndarray:
        """Build the mosaic image from segments."""
        h, w = image.shape[:2]
        mosaic = np.zeros_like(image)
        
        for segment in segments:
            x, y, size = segment['x'], segment['y'], segment['size']
            cell = segment['cell']
            
            if cell.size == 0:
                continue
            
            # Get average color
            avg_color = self.get_average_color(cell)
            
            # Create tile based on type
            if tile_type == 'solid':
                tile = self.create_color_tile(avg_color, size)
            elif tile_type == 'pattern':
                tile = self.create_pattern_tile(pattern_type, avg_color, size)
            else:  # mini_image
                tile = self.create_mini_image_tile(cell, size)
            
            # Place tile in mosaic
            actual_h = min(size, h - y)
            actual_w = min(size, w - x)
            mosaic[y:y+actual_h, x:x+actual_w] = tile[:actual_h, :actual_w]
        
        return mosaic
    
    def calculate_mse(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Mean Squared Error between two images."""
        return np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    
    def calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Structural Similarity Index between two images."""
        # Convert to grayscale for SSIM calculation
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        return ssim(gray1, gray2)
    
    
    def process_image(self, input_image, apply_quantization, n_colors, base_grid_size,
                     variance_threshold, tile_type, pattern_type, quality_priority):
        """Main processing function for Gradio interface."""
        
        if input_image is None:
            return None, None, None, "Please upload an image", 0.0, 0.0
        
        # Convert to numpy array
        image = np.array(input_image)
        
        # Store original
        self.original_image = image.copy()
        
        # Resize for consistent processing
        max_dim = 800 if quality_priority == "Quality" else 400
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Apply color quantization if requested
        if apply_quantization:
            image = self.quantize_colors(image, n_colors)
        
        # Perform adaptive grid segmentation
        segments = self.adaptive_grid_segmentation(image, base_grid_size, 
                                                  variance_threshold)
        
        # Build mosaic
        mosaic = self.build_mosaic(image, segments, tile_type, pattern_type)
    
        self.mosaic_image = mosaic
        
        # Calculate metrics
        mse = self.calculate_mse(image, mosaic)
        ssim_score = self.calculate_ssim(image, mosaic)
        
        # Create side-by-side comparison
        comparison = np.hstack([image, mosaic])        
        comparison_pil = Image.fromarray(comparison)
        
        return mosaic, comparison_pil, len(segments), mse, ssim_score

# Create global instance
mosaic_gen = MosaicGenerator()

# Gradio Interface
def create_interface():
    with gr.Blocks(title="Interactive Image Mosaic Generator") as demo:
        gr.Markdown("""
        # 🎨 Interactive Image Mosaic Generator
        
        Transform your images into beautiful mosaic art with adaptive gridding and various tile styles!
        """)
        
        with gr.Row(scale=2):
            # Input controls
            input_image = gr.Image(label="Upload Image", type="pil")

            # Output displays
            with gr.Tab("Mosaic Result"):
                mosaic_output = gr.Image(label="Mosaic Image")
            
            with gr.Tab("Side-by-Side Comparison"):
                comparison_output = gr.Image(label="Original vs Mosaic")

        with gr.Row():
            gr.Column(scale=1)  # Empty column for left spacing
            process_btn = gr.Button("Generate Mosaic", variant="primary")
            gr.Column(scale=1)  # Empty column for right spacing

                    
        with gr.Row(scale=3):
            segments_count = gr.Number(label="Number of Segments", precision=0)
            mse_metric = gr.Number(label="MSE (Lower is better)", precision=2)
            ssim_metric = gr.Number(label="SSIM (Higher is better)", precision=4)
    
        with gr.Row():
            # with gr.Column(scale=1):
            with gr.Accordion("Color Settings", open=True):
                apply_quantization = gr.Checkbox(label="Apply Color Quantization", 
                                                value=False)
                n_colors = gr.Slider(minimum=2, maximum=32, value=8, step=1,
                                    label="Number of Colors (if quantization enabled)")
            
            with gr.Accordion("Grid Settings", open=True):
                base_grid_size = gr.Slider(minimum=8, maximum=64, value=32, step=8,
                                            label="Base Grid Size")
                variance_threshold = gr.Slider(minimum=0, maximum=100, value=30, step=5,
                                                label="Subdivision Threshold (higher = more detail)")
            
            with gr.Accordion("Tile Settings", open=True):
                tile_type = gr.Radio(choices=["solid", "pattern", "mini_image"],
                                    value="solid", label="Tile Type")
                pattern_type = gr.Dropdown(choices=["solid", "gradient_h", "gradient_v", 
                                                    "gradient_diag", "circle", "cross"],
                                            value="gradient_diag", label="Pattern Type (if pattern)")
            
            with gr.Accordion("Performance Settings", open=True):
                quality_priority = gr.Radio(choices=["Speed", "Quality"], 
                                            value="Quality",
                                            label="Processing Priority")            
            
        # Example images
        gr.Examples(
            examples=[
                ["examples/sphynx_cat_0.png"],
                ["examples/sphynx_cat_1.jpg"],
                ["examples/cat_sakura_cut_female.png"],
                ["examples/cat_sakura_cut_male.png"]
            ],
            inputs=input_image,
            label="Sample Images"
        )
        
        # Event handlers
        process_btn.click(
            fn=mosaic_gen.process_image,
            inputs=[input_image, apply_quantization, n_colors, base_grid_size,
                   variance_threshold, tile_type, pattern_type, quality_priority],
            outputs=[mosaic_output, comparison_output, segments_count, mse_metric, ssim_metric]
        )
        
        with gr.Row():
            gr.Markdown("""
            ## 📝 Instructions
            
            1. **Upload an image** using the input panel
            2. **Adjust settings** to control the mosaic style:
                - **Color Quantization**: Reduce colors for a more stylized look
                - **Grid Size**: Larger = fewer, bigger tiles
                - **Subdivision Threshold**: Higher values create more detail in complex areas
                - **Tile Type**: Choose between solid colors, patterns, or mini-images
            3. **Click "Generate Mosaic"** to create your artwork
            4. **View results** in different tabs and check quality metrics
            """)
            
            gr.Markdown("""
            ## 🎯 Tips
            - Start with default settings and adjust gradually
            - Use color quantization for a more artistic effect
            - Lower subdivision threshold for abstract style
            - Try different tile types for unique effects
            """)
    
    return demo

# Launch the application
if __name__ == "__main__":
    # Create example images directory (you'll need to add actual images)
    import os
    if not os.path.exists("examples"):
        os.makedirs("examples")
        print("Please add sample images to the 'examples' directory")
    
    # Create and launch interface
    demo = create_interface()
    demo.launch(share=True, debug=True)
