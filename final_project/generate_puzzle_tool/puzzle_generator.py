import os
import random
import json
import glob
from PIL import Image, ImageDraw
import numpy as np

class PuzzleDatasetGenerator:
    def __init__(self, output_dir="output_dataset"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_puzzle(self, image_path, puzzle_name, grid_size=3):
        """
        Split image into puzzle pieces

        Args:
            image_path: Original image path
            puzzle_name: Puzzle folder name
            grid_size: Grid size (default 3x3)
        """
        # Create folder structure for this puzzle
        puzzle_dir = f"{self.output_dir}/{puzzle_name}"
        os.makedirs(puzzle_dir, exist_ok=True)
        os.makedirs(f"{puzzle_dir}/pieces", exist_ok=True)
        os.makedirs(f"{puzzle_dir}/annotations", exist_ok=True)

        img = Image.open(image_path).convert('RGB')

        # Resize image to square (easier to split)
        size = min(img.size)
        img = img.crop((0, 0, size, size))
        img = img.resize((900, 900))  # Standardize size

        # Save original image
        original_path = f"{puzzle_dir}/original.png"
        img.save(original_path)
        
        w, h = img.size
        piece_w, piece_h = w // grid_size, h // grid_size
        
        pieces_info = []
        
        # Split puzzle
        for i in range(grid_size):
            for j in range(grid_size):
                left = j * piece_w
                upper = i * piece_h
                right = left + piece_w
                lower = upper + piece_h

                piece = img.crop((left, upper, right, lower))
                piece_id = i * grid_size + j
                piece_path = f"{puzzle_dir}/pieces/piece_{piece_id}.png"
                piece.save(piece_path)

                pieces_info.append({
                    'piece_id': piece_id,
                    'row': i,
                    'col': j,
                    'path': piece_path,
                    'bbox': [left, upper, right, lower]
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
                               max_rotation=30, add_noise=False):
        """
        Create scattered puzzle image (simulating real photo scenario)

        Args:
            puzzle_dir: Puzzle directory path
            pieces_info: Puzzle pieces information
            img_size: Original image size
            max_rotation: Maximum rotation angle
            add_noise: Whether to add noise
        """
        # Create larger canvas (leave space for pieces to move)
        canvas_size = (img_size[0] * 2, img_size[1] * 2)
        canvas = Image.new('RGB', canvas_size, color=(240, 240, 240))
        
        scattered_positions = []
        
        # Randomly place each piece
        for piece_info in pieces_info:
            piece_img = Image.open(piece_info['path'])
            
            # Random rotation
            rotation = random.uniform(-max_rotation, max_rotation)
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
    # Initialize generator
    generator = PuzzleDatasetGenerator(output_dir="output_dataset")

    # Automatically scan all images from source_images folder
    source_dir = "source_images"
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.avif', '*.webp']
    image_paths = []

    for ext in image_extensions:
        image_paths.extend(glob.glob(f"{source_dir}/{ext}"))
        image_paths.extend(glob.glob(f"{source_dir}/{ext.upper()}"))

    if not image_paths:
        print(f"No images found in '{source_dir}' folder!")
        print(f"Supported formats: jpg, jpeg, png, bmp, gif, avif, webp")
        return

    print(f"Found {len(image_paths)} image(s) in '{source_dir}' folder")
    print("-" * 50)

    for idx, image_path in enumerate(image_paths):
        # Get image filename without extension as folder name
        image_filename = os.path.basename(image_path)
        puzzle_name = os.path.splitext(image_filename)[0]

        print(f"[{idx+1}/{len(image_paths)}] Processing: {image_filename}...")

        # 1. Split puzzle
        pieces_info, img_size, puzzle_dir = generator.create_puzzle(image_path, puzzle_name, grid_size=3)

        # 2. Create annotated solution
        generator.visualize_solution(puzzle_dir, pieces_info, img_size)

        # 3. Create scattered image
        generator.create_scattered_image(puzzle_dir, pieces_info, img_size,
                                        max_rotation=15, add_noise=True)

        print(f"✓ {puzzle_name} completed!")
        print()

    print("=" * 50)
    print(f"All done! Dataset saved in '{generator.output_dir}/' folder")
    print("\nOutput structure for each image:")
    print("  - original.png: Original complete puzzle")
    print("  - solution.png: Annotated solution with piece numbers")
    print("  - scattered.png: Scattered puzzle scene")
    print("  - pieces/: 9 individual puzzle pieces (piece_0.png to piece_8.png)")
    print("  - annotations/: JSON files with detailed information")

if __name__ == "__main__":
    main()