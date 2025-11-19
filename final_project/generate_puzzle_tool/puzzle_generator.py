import os
import random
import json
import glob
import argparse
from PIL import Image, ImageDraw
import numpy as np

class PuzzleDatasetGenerator:
    def __init__(self, output_dir="output_dataset"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def assign_edge_types(self, grid_size):
        """
        Assign tab/blank/flat types to all edges in the puzzle grid.
        Ensures adjacent pieces have complementary edges.
        
        Returns:
            Dictionary mapping (row, col) to edge types
        """
        edge_types = {}
        
        # Initialize all pieces
        for i in range(grid_size):
            for j in range(grid_size):
                edge_types[(i, j)] = {
                    'top': 'flat',
                    'right': 'flat',
                    'bottom': 'flat',
                    'left': 'flat'
                }
        
        # Assign horizontal edges (between rows)
        for i in range(grid_size - 1):
            for j in range(grid_size):
                # Randomly choose tab or blank for the edge
                edge_type = random.choice(['tab', 'blank'])
                edge_types[(i, j)]['bottom'] = edge_type
                # Adjacent piece gets complementary edge
                edge_types[(i + 1, j)]['top'] = 'blank' if edge_type == 'tab' else 'tab'
        
        # Assign vertical edges (between columns)
        for i in range(grid_size):
            for j in range(grid_size - 1):
                # Randomly choose tab or blank for the edge
                edge_type = random.choice(['tab', 'blank'])
                edge_types[(i, j)]['right'] = edge_type
                # Adjacent piece gets complementary edge
                edge_types[(i, j + 1)]['left'] = 'blank' if edge_type == 'tab' else 'tab'
        
        return edge_types
    
    def generate_jigsaw_mask(self, piece_w, piece_h, top_type, right_type, bottom_type, left_type, tab_size=0.20):
        """
        Generate a mask for a jigsaw piece with realistic interlocking tabs and blanks.
        
        Args:
            piece_w, piece_h: Piece dimensions
            top_type, right_type, bottom_type, left_type: 'flat', 'tab', or 'blank'
            tab_size: Size of tab relative to piece dimension (default 0.20)
        
        Returns:
            PIL Image mask (L mode) and the expanded bounding box
        """
        # Calculate tab radius for more realistic proportions
        tab_radius = int(min(piece_w, piece_h) * tab_size)
        
        # Create expanded canvas to accommodate tabs
        expand = tab_radius
        mask_w = piece_w + 2 * expand
        mask_h = piece_h + 2 * expand
        
        # Create mask
        mask = Image.new('L', (mask_w, mask_h), 0)
        draw = ImageDraw.Draw(mask)
        
        # Start with base rectangle (offset by expand amount)
        base_left = expand
        base_top = expand
        base_right = expand + piece_w
        base_bottom = expand + piece_h
        
        # Build polygon points for the piece outline
        points = []
        
        # Top edge
        if top_type == 'flat':
            points.extend([(base_left, base_top), (base_right, base_top)])
        else:
            # Realistic jigsaw tab/blank on top edge
            cx = base_left + piece_w // 2
            cy = base_top
            
            # Left portion before tab
            points.append((base_left, base_top))
            points.append((cx - tab_radius, base_top))
            
            # Create curved tab/blank shape
            for angle in range(0, 181, 8):
                rad = np.radians(angle)
                if top_type == 'tab':
                    x = cx + tab_radius * np.cos(rad + np.pi)
                    y = cy - tab_radius * (0.7 + 0.3 * np.sin(rad))
                else:  # blank
                    x = cx + tab_radius * np.cos(rad + np.pi)
                    y = cy + tab_radius * (0.7 + 0.3 * np.sin(rad))
                points.append((int(x), int(y)))
            
            # Right portion after tab
            points.append((cx + tab_radius, base_top))
            points.append((base_right, base_top))
        
        # Right edge
        if right_type == 'flat':
            points.append((base_right, base_bottom))
        else:
            cx = base_right
            cy = base_top + piece_h // 2
            
            # Top portion before tab
            points.append((base_right, cy - tab_radius))
            
            # Create curved tab/blank shape
            for angle in range(-90, 91, 8):
                rad = np.radians(angle)
                if right_type == 'tab':
                    x = cx + tab_radius * (0.7 + 0.3 * np.cos(rad))
                    y = cy + tab_radius * np.sin(rad)
                else:  # blank
                    x = cx - tab_radius * (0.7 + 0.3 * np.cos(rad))
                    y = cy + tab_radius * np.sin(rad)
                points.append((int(x), int(y)))
            
            # Bottom portion after tab
            points.append((base_right, cy + tab_radius))
            points.append((base_right, base_bottom))
        
        # Bottom edge
        if bottom_type == 'flat':
            points.append((base_left, base_bottom))
        else:
            cx = base_left + piece_w // 2
            cy = base_bottom
            
            # Right portion before tab
            points.append((base_right, base_bottom))
            points.append((cx + tab_radius, base_bottom))
            
            # Create curved tab/blank shape
            for angle in range(180, -1, -8):
                rad = np.radians(angle)
                if bottom_type == 'tab':
                    x = cx + tab_radius * np.cos(rad + np.pi)
                    y = cy + tab_radius * (0.7 + 0.3 * np.sin(rad))
                else:  # blank
                    x = cx + tab_radius * np.cos(rad + np.pi)
                    y = cy - tab_radius * (0.7 + 0.3 * np.sin(rad))
                points.append((int(x), int(y)))
            
            # Left portion after tab
            points.append((cx - tab_radius, base_bottom))
            points.append((base_left, base_bottom))
        
        # Left edge
        if left_type == 'flat':
            points.append((base_left, base_top))
        else:
            cx = base_left
            cy = base_top + piece_h // 2
            
            # Bottom portion before tab
            points.append((base_left, cy + tab_radius))
            
            # Create curved tab/blank shape
            for angle in range(90, -91, -8):
                rad = np.radians(angle)
                if left_type == 'tab':
                    x = cx - tab_radius * (0.7 + 0.3 * np.cos(rad))
                    y = cy + tab_radius * np.sin(rad)
                else:  # blank
                    x = cx + tab_radius * (0.7 + 0.3 * np.cos(rad))
                    y = cy + tab_radius * np.sin(rad)
                points.append((int(x), int(y)))
            
            # Top portion after tab
            points.append((base_left, cy - tab_radius))
            points.append((base_left, base_top))
        
        # Draw filled polygon
        draw.polygon(points, fill=255)
        
        # Return mask and offset information
        return mask, expand
    
    def create_puzzle(self, image_path, puzzle_name, grid_size=3, jigsaw_style=False, tab_size=0.20, max_dimension=900):
        """
        Split image into puzzle pieces

        Args:
            image_path: Original image path
            puzzle_name: Puzzle folder name
            grid_size: Grid size (default 3x3)
            jigsaw_style: If True, create jigsaw pieces with tabs/blanks; if False, create square pieces
            tab_size: Size of tab relative to piece dimension (default 0.20)
            max_dimension: Maximum dimension for resized image (default 900)
        """
        # Create folder structure for this puzzle
        puzzle_dir = f"{self.output_dir}/{puzzle_name}"
        os.makedirs(puzzle_dir, exist_ok=True)
        os.makedirs(f"{puzzle_dir}/pieces", exist_ok=True)
        os.makedirs(f"{puzzle_dir}/annotations", exist_ok=True)

        img = Image.open(image_path).convert('RGB')

        # Resize image to standard size while maintaining aspect ratio
        img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

        # Save original image
        original_path = f"{puzzle_dir}/original.png"
        img.save(original_path)
        
        w, h = img.size
        piece_w, piece_h = w // grid_size, h // grid_size
        
        pieces_info = []
        edge_types = None
        
        # Generate edge types for jigsaw mode
        if jigsaw_style:
            edge_types = self.assign_edge_types(grid_size)
        
        # Split puzzle
        for i in range(grid_size):
            for j in range(grid_size):
                left = j * piece_w
                upper = i * piece_h
                right = left + piece_w
                lower = upper + piece_h
                piece_id = i * grid_size + j
                piece_path = f"{puzzle_dir}/pieces/piece_{piece_id}.png"

                if jigsaw_style:
                    # Get edge types for this piece
                    edges = edge_types[(i, j)]
                    
                    # Generate jigsaw mask
                    mask, expand = self.generate_jigsaw_mask(
                        piece_w, piece_h,
                        edges['top'], edges['right'], edges['bottom'], edges['left'],
                        tab_size=tab_size
                    )
                    
                    # Crop expanded area from original image
                    crop_left = max(0, left - expand)
                    crop_upper = max(0, upper - expand)
                    crop_right = min(w, right + expand)
                    crop_lower = min(h, lower + expand)
                    
                    piece_img = img.crop((crop_left, crop_upper, crop_right, crop_lower))
                    
                    # Adjust mask if piece is at border (crop was limited)
                    actual_expand_left = left - crop_left
                    actual_expand_top = upper - crop_upper
                    actual_expand_right = crop_right - right
                    actual_expand_bottom = crop_lower - lower
                    
                    # Crop mask to match actual piece size
                    mask_crop_left = expand - actual_expand_left
                    mask_crop_top = expand - actual_expand_top
                    mask_crop_right = expand + piece_w + actual_expand_right
                    mask_crop_bottom = expand + piece_h + actual_expand_bottom
                    
                    mask = mask.crop((mask_crop_left, mask_crop_top, mask_crop_right, mask_crop_bottom))
                    
                    # Create RGBA image with transparency
                    piece = Image.new('RGBA', piece_img.size)
                    piece.paste(piece_img, (0, 0))
                    piece.putalpha(mask)
                    piece.save(piece_path)
                    
                    pieces_info.append({
                        'piece_id': piece_id,
                        'row': i,
                        'col': j,
                        'path': piece_path,
                        'bbox': [left, upper, right, lower],
                        'edges': edges,
                        'jigsaw': True
                    })
                else:
                    # Original square piece logic
                    piece = img.crop((left, upper, right, lower))
                    piece.save(piece_path)

                    pieces_info.append({
                        'piece_id': piece_id,
                        'row': i,
                        'col': j,
                        'path': piece_path,
                        'bbox': [left, upper, right, lower],
                        'jigsaw': False
                    })

        # Generate annotation file
        annotation = {
            'puzzle_name': puzzle_name,
            'original_image': original_path,
            'grid_size': grid_size,
            'image_size': [w, h],
            'piece_size': [piece_w, piece_h],
            'pieces': pieces_info
        }

        with open(f"{puzzle_dir}/annotations/puzzle_info.json", 'w') as f:
            json.dump(annotation, f, indent=2)
        
        return pieces_info, img.size, puzzle_dir

    def create_scattered_image(self, puzzle_dir, pieces_info, img_size,
                               max_rotation=0, add_noise=False):
        """
        Create scattered puzzle image (simulating real photo scenario)

        Args:
            puzzle_dir: Puzzle directory path
            pieces_info: Puzzle pieces information
            img_size: Original image size
            max_rotation: Maximum rotation angle (default 0 for no rotation)
            add_noise: Whether to add noise
        """
        # Create larger canvas (leave space for pieces to move)
        canvas_size = (img_size[0] * 2, img_size[1] * 2)
        # Use RGBA for jigsaw pieces
        is_jigsaw = pieces_info[0].get('jigsaw', False)
        canvas = Image.new('RGBA' if is_jigsaw else 'RGB', canvas_size, color=(240, 240, 240, 255) if is_jigsaw else (240, 240, 240))
        
        scattered_positions = []
        
        # Randomly place each piece
        for piece_info in pieces_info:
            piece_img = Image.open(piece_info['path'])
            
            # Random rotation (only if max_rotation > 0)
            rotation = 0
            if max_rotation > 0:
                rotation = random.uniform(-max_rotation, max_rotation)
                if is_jigsaw:
                    piece_img = piece_img.rotate(rotation, expand=True, fillcolor=(240, 240, 240, 0))
                else:
                    piece_img = piece_img.rotate(rotation, expand=True, fillcolor=(240, 240, 240))
            
            # Random position (avoid overlap)
            max_attempts = 50
            for _ in range(max_attempts):
                x = random.randint(50, canvas_size[0] - piece_img.width - 50)
                y = random.randint(50, canvas_size[1] - piece_img.height - 50)
                
                # Simple overlap detection
                overlap = False
                for pos in scattered_positions:
                    if (abs(x - pos['x']) < piece_img.width and 
                        abs(y - pos['y']) < piece_img.height):
                        overlap = True
                        break
                
                if not overlap:
                    break
            
            # Paste with alpha channel support for jigsaw pieces
            if is_jigsaw and piece_img.mode == 'RGBA':
                canvas.paste(piece_img, (x, y), piece_img)
            else:
                canvas.paste(piece_img, (x, y))
            
            scattered_positions.append({
                'piece_id': piece_info['piece_id'],
                'x': x,
                'y': y,
                'rotation': rotation,
                'width': piece_img.width,
                'height': piece_img.height
            })
        
        # Add noise (optional)
        if add_noise:
            canvas_array = np.array(canvas)
            noise = np.random.normal(0, 5, canvas_array.shape)
            canvas_array = np.clip(canvas_array + noise, 0, 255).astype(np.uint8)
            canvas = Image.fromarray(canvas_array)
        
        # Save scattered image
        scattered_path = f"{puzzle_dir}/scattered.png"
        canvas.save(scattered_path)

        # Save scattered annotation
        scattered_annotation = {
            'scattered_image': scattered_path,
            'canvas_size': canvas_size,
            'pieces_positions': scattered_positions
        }

        with open(f"{puzzle_dir}/annotations/scattered_info.json", 'w') as f:
            json.dump(scattered_annotation, f, indent=2)
        
        return scattered_path

    def visualize_solution(self, puzzle_dir, pieces_info, img_size):
        """
        Create annotated solution visualization
        """
        solution = Image.new('RGB', img_size, color=(255, 255, 255))
        draw = ImageDraw.Draw(solution)
        
        for piece_info in pieces_info:
            piece_img = Image.open(piece_info['path'])
            bbox = piece_info['bbox']
            solution.paste(piece_img, (bbox[0], bbox[1]))
            
            # Draw border and labels
            draw.rectangle(bbox, outline=(255, 0, 0), width=2)
            
            # Add piece number
            text = str(piece_info['piece_id'])
            text_bbox = draw.textbbox((0, 0), text)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            text_pos = (bbox[0] + 10, bbox[1] + 10)
            draw.rectangle([text_pos[0]-2, text_pos[1]-2, 
                          text_pos[0]+text_w+2, text_pos[1]+text_h+2], 
                         fill=(255, 255, 0))
            draw.text(text_pos, text, fill=(0, 0, 0))

        solution_path = f"{puzzle_dir}/solution.png"
        solution.save(solution_path)
        
        return solution_path

def main():
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='Generate jigsaw puzzle datasets from images')
    parser.add_argument('--jigsaw', action='store_true', default=True,
                        help='Create jigsaw pieces with tabs/blanks (default: True)')
    parser.add_argument('--square', action='store_true',
                        help='Create square pieces instead of jigsaw (overrides --jigsaw)')
    parser.add_argument('--grid-size', type=int, default=3,
                        help='Grid size (e.g., 3 for 3x3 = 9 pieces, default: 3)')
    parser.add_argument('--rotation', type=float, default=0,
                        help='Maximum rotation angle in degrees (default: 0, no rotation)')
    parser.add_argument('--no-noise', action='store_true',
                        help='Disable noise on scattered image')
    parser.add_argument('--tab-size', type=float, default=0.20,
                        help='Tab size relative to piece dimension (default: 0.20)')
    parser.add_argument('--max-dimension', type=int, default=900,
                        help='Maximum image dimension in pixels (default: 900)')
    parser.add_argument('--output-dir', type=str, default='output_dataset',
                        help='Output directory (default: output_dataset)')
    parser.add_argument('--source-dir', type=str, default='source_images',
                        help='Source images directory (default: source_images)')
    
    args = parser.parse_args()
    
    # Determine puzzle style
    jigsaw_style = not args.square  # If --square is set, disable jigsaw
    add_noise = not args.no_noise
    
    # Initialize generator
    generator = PuzzleDatasetGenerator(output_dir=args.output_dir)

    # Automatically scan all images from source_images folder
    source_dir = args.source_dir
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.avif', '*.webp']
    image_paths = []

    for ext in image_extensions:
        image_paths.extend(glob.glob(f"{source_dir}/{ext}"))
        image_paths.extend(glob.glob(f"{source_dir}/{ext.upper()}"))

    if not image_paths:
        print(f"No images found in '{source_dir}' folder!")
        print(f"Supported formats: jpg, jpeg, png, bmp, gif, avif, webp")
        return
    
    print(f"\nFound {len(image_paths)} image(s) in '{source_dir}' folder")
    print(f"Configuration:")
    print(f"  - Style: {'Jigsaw' if jigsaw_style else 'Square'} pieces")
    print(f"  - Grid: {args.grid_size}x{args.grid_size} = {args.grid_size * args.grid_size} pieces")
    print(f"  - Rotation: ±{args.rotation}° {'(disabled)' if args.rotation == 0 else ''}")
    print(f"  - Noise: {'Enabled' if add_noise else 'Disabled'}")
    if jigsaw_style:
        print(f"  - Tab size: {args.tab_size}")
    print(f"  - Max dimension: {args.max_dimension}px")
    print("-" * 50)

    for idx, image_path in enumerate(image_paths):
        # Get image filename without extension as folder name
        image_filename = os.path.basename(image_path)
        puzzle_name = os.path.splitext(image_filename)[0]

        print(f"[{idx+1}/{len(image_paths)}] Processing: {image_filename}...")

        # 1. Split puzzle (pass tab_size and max_dimension through generator method)
        pieces_info, img_size, puzzle_dir = generator.create_puzzle(
            image_path, puzzle_name, 
            grid_size=args.grid_size, 
            jigsaw_style=jigsaw_style,
            tab_size=args.tab_size,
            max_dimension=args.max_dimension
        )

        # 2. Create annotated solution
        generator.visualize_solution(puzzle_dir, pieces_info, img_size)

        # 3. Create scattered image
        generator.create_scattered_image(puzzle_dir, pieces_info, img_size,
                                        max_rotation=args.rotation, 
                                        add_noise=add_noise)

        print(f"✓ {puzzle_name} completed!")
        print()

    print("=" * 50)
    print(f"All done! Dataset saved in '{args.output_dir}/' folder")
    print("\nOutput structure for each image:")
    print("  - original.png: Original complete puzzle")
    print("  - solution.png: Annotated solution with piece numbers")
    print("  - scattered.png: Scattered puzzle scene")
    print(f"  - pieces/: {args.grid_size * args.grid_size} individual {'jigsaw' if jigsaw_style else 'square'} puzzle pieces (piece_0.png to piece_{args.grid_size * args.grid_size - 1}.png)")
    print("  - annotations/: JSON files with detailed information")

if __name__ == "__main__":
    main()
