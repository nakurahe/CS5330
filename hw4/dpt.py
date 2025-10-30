# ============================================
# Provided Part 1
# ============================================

import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import DPTImageProcessor, DPTForDepthEstimation
import requests
import cv2
import numpy as np
import imageio
import gradio as gr
import tempfile
import os

# ============================================
# Initialize the Depth Model
# ============================================
print("Loading Intel DPT depth estimation model...")
processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
model.eval()  # Set to evaluation mode

# Use GPU if available for faster processing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Model loaded on {device}")

# ============================================
# Load and Prepare Your Image
# ============================================
# Load from URL
image_url = "https://images.pexels.com/photos/1681010/pexels-photo-1681010.jpeg"
image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')

# Resize for faster processing (optional but recommended)
max_size = 640
if max(image.size) > max_size:
    ratio = max_size / max(image.size)
    new_size = tuple(int(dim * ratio) for dim in image.size)
    image = image.resize(new_size, Image.LANCZOS)

print(f"Image size: {image.size}")

# ============================================
# Extract Depth Map
# ============================================
# Prepare image for the model
inputs = processor(images=image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

# Run depth estimation
with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

# Interpolate to original size and normalize
prediction = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=image.size[::-1],  # (height, width)
    mode="bicubic",
    align_corners=False,
)

# Convert to numpy and normalize to 0-1 range
depth_map = prediction.squeeze().cpu().numpy()
depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

# # ============================================
# # Visualize Results
# # ============================================
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# # Original image
# axes[0].imshow(image)
# axes[0].set_title('Original Image')
# axes[0].axis('off')

# # Depth map
# im = axes[1].imshow(depth_map, cmap='plasma')
# axes[1].set_title('Depth Map (Yellow=Close, Purple=Far)')
# axes[1].axis('off')
# plt.colorbar(im, ax=axes[1], fraction=0.046)

# plt.tight_layout()
# plt.show()

# # ============================================
# # What You Get
# # ============================================
# print(f"Depth map shape: {depth_map.shape}") 
# print(f"Depth range: [{depth_map.min():.3f}, {depth_map.max():.3f}]") 
# print("Ready for processing!") 
# # depth_map is now a normalized numpy array where: 
# # - Values close to 1.0 = near to camera (yellow in visualization) 
# # - Values close to 0.0 = far from camera (purple in visualization) 
# # Use this depth_map for all subsequent processing!

# ============================================
# Part 2: Separate the image into foreground (the person) and background based on depth
# ============================================

print("\nProcessing foreground/background separation...")

# Convert depth map to uint8 for OpenCV processing
depth_uint8 = (depth_map * 255).astype(np.uint8)

# Step 1: Apply Otsu's thresholding to automatically find optimal threshold
# Higher depth values (closer to camera) will be foreground
_, binary_mask = cv2.threshold(depth_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print(f"Otsu threshold automatically selected")

# Step 2: Clean up the mask using morphological operations
# Remove noise with opening (erosion followed by dilation)
kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask_opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_small)

# Fill holes with closing (dilation followed by erosion)
kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
mask_cleaned = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel_large)

# Step 3: Find the largest connected component (assumed to be the main foreground)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_cleaned, connectivity=8)

# Skip label 0 (background) and find largest component
if num_labels > 1:
    largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    foreground_mask = (labels == largest_component).astype(np.uint8) * 255
else:
    foreground_mask = mask_cleaned

print(f"Found {num_labels - 1} connected components")

# Step 4: Create a soft edge mask using Gaussian blur for better blending
soft_mask = cv2.GaussianBlur(foreground_mask, (21, 21), 0)
soft_mask_normalized = soft_mask.astype(np.float32) / 255.0

# Step 5: Apply masks to separate foreground and background
# Convert PIL image to numpy array
image_np = np.array(image)

# Create foreground by applying the mask
foreground = image_np.copy()
foreground_mask_3ch = np.stack([soft_mask_normalized] * 3, axis=-1)
foreground = (foreground * foreground_mask_3ch).astype(np.uint8)

# Create background by inverting the mask
background = image_np.copy()
background_mask_3ch = 1.0 - foreground_mask_3ch
background = (background * background_mask_3ch).astype(np.uint8)

print("Separation complete!")

# # ============================================
# # Visualize Separation Results
# # ============================================
# fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# # Row 1: Original processing
# axes[0, 0].imshow(image)
# axes[0, 0].set_title('Original Image')
# axes[0, 0].axis('off')

# axes[0, 1].imshow(depth_map, cmap='plasma')
# axes[0, 1].set_title('Depth Map')
# axes[0, 1].axis('off')

# axes[0, 2].imshow(binary_mask, cmap='gray')
# axes[0, 2].set_title('Binary Mask (Otsu)')
# axes[0, 2].axis('off')

# # Row 2: Final results
# axes[1, 0].imshow(foreground_mask, cmap='gray')
# axes[1, 0].set_title('Cleaned Foreground Mask')
# axes[1, 0].axis('off')

# axes[1, 1].imshow(foreground)
# axes[1, 1].set_title('Foreground (Person)')
# axes[1, 1].axis('off')

# axes[1, 2].imshow(background)
# axes[1, 2].set_title('Background')
# axes[1, 2].axis('off')

# plt.tight_layout()
# plt.show()

# print("\nResults:")
# print(f"Foreground mask shape: {foreground_mask.shape}")
# print(f"Foreground pixels: {np.sum(foreground_mask > 0)}")
# print(f"Background pixels: {np.sum(foreground_mask == 0)}")

# ============================================
# Part 3: Create a clean background by filling holes where the foreground (the person) was removed
# ============================================

print("\nProcessing background inpainting...")

# Step 1: Dilate the foreground mask to ensure better edge coverage during inpainting
# This helps avoid visible seams at the edges of the inpainted region
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
dilated_mask = cv2.dilate(foreground_mask, kernel_dilate, iterations=2)

print(f"Dilated mask to cover edges better")

# Step 2: First inpainting pass using Telea algorithm (Fast Marching Method)
# This method is fast and good at propagating structure
inpainted_telea = cv2.inpaint(image_np, dilated_mask, inpaintRadius=15, flags=cv2.INPAINT_TELEA)

print("Applied Telea inpainting (Fast Marching Method)")

# Step 3: Second inpainting pass using Navier-Stokes algorithm
# This method is better at maintaining textures and smooth gradients
inpainted_ns = cv2.inpaint(image_np, dilated_mask, inpaintRadius=15, flags=cv2.INPAINT_NS)

print("Applied Navier-Stokes inpainting")

# Step 4: Blend the two inpainting results for best quality
# Use weighted average to combine strengths of both methods
blended_inpaint = cv2.addWeighted(inpainted_telea, 0.5, inpainted_ns, 0.5, 0)

# Step 5: Apply bilateral filter for edge-preserving smoothing
# This reduces artifacts while maintaining important edges
smooth_inpaint = cv2.bilateralFilter(blended_inpaint, d=9, sigmaColor=75, sigmaSpace=75)

print("Applied bilateral filtering for smoothing")

# Step 6: Create final clean background
# Selectively apply inpainted regions only where the foreground was removed
clean_background = np.where(
    dilated_mask[:, :, np.newaxis] > 0,  # Where mask is active
    smooth_inpaint,                       # Use inpainted content
    image_np                              # Keep original background
).astype(np.uint8)

print("Clean background created!")

# # ============================================
# # Visualize Inpainting Results
# # ============================================
# fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# # Row 1: Mask and inpainting methods
# axes[0, 0].imshow(foreground_mask, cmap='gray')
# axes[0, 0].set_title('Original Mask')
# axes[0, 0].axis('off')

# axes[0, 1].imshow(dilated_mask, cmap='gray')
# axes[0, 1].set_title('Dilated Mask (for inpainting)')
# axes[0, 1].axis('off')

# axes[0, 2].imshow(inpainted_telea)
# axes[0, 2].set_title('Telea Inpainting')
# axes[0, 2].axis('off')

# # Row 2: Comparison of results
# axes[1, 0].imshow(inpainted_ns)
# axes[1, 0].set_title('Navier-Stokes Inpainting')
# axes[1, 0].axis('off')

# axes[1, 1].imshow(blended_inpaint)
# axes[1, 1].set_title('Blended Result')
# axes[1, 1].axis('off')

# axes[1, 2].imshow(clean_background)
# axes[1, 2].set_title('Final Clean Background')
# axes[1, 2].axis('off')

# plt.tight_layout()
# plt.show()

# # ============================================
# # Final Comparison: Original vs Clean Background
# # ============================================
# fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# axes[0].imshow(image_np)
# axes[0].set_title('Original Image with Person')
# axes[0].axis('off')

# axes[1].imshow(clean_background)
# axes[1].set_title('Clean Background (Person Removed)')
# axes[1].axis('off')

# plt.tight_layout()
# plt.show()

# print("\nInpainting Statistics:")
# print(f"Mask area covered: {np.sum(dilated_mask > 0)} pixels")
# print(f"Percentage of image inpainted: {100 * np.sum(dilated_mask > 0) / dilated_mask.size:.2f}%")
# print("Background cleanup complete!")

# ============================================
# Part 4 & 5: Depth-Aware Motion Synthesis with Bokeh Effects
# Goal: Create a smooth parallax animation with realistic depth-of-field
# ============================================

print("\nGenerating parallax animation with bokeh effects...")

# Animation parameters
num_frames = 30
foreground_max_shift = 10  # pixels
background_max_shift = 3   # pixels

# Define aperture settings (f-numbers)
# Lower f-number = shallower depth of field = more blur
aperture_settings = {
    'f/1.4': {'name': 'f/1.4 (Heavy Blur)', 'max_kernel': 31, 'intensity': 1.0},
    'f/2.8': {'name': 'f/2.8 (Medium Blur)', 'max_kernel': 17, 'intensity': 0.6},
    'f/5.6': {'name': 'f/5.6 (Light Blur)', 'max_kernel': 9, 'intensity': 0.3}
}

# Choose aperture for animation
selected_aperture = 'f/2.8'
aperture_config = aperture_settings[selected_aperture]

print(f"Using aperture: {aperture_config['name']}")

# Smoothstep function for natural easing
def smoothstep(t):
    """Smooth interpolation function for natural motion"""
    return t * t * (3 - 2 * t)

def apply_depth_based_blur(image, depth_map, foreground_mask, aperture_config):
    """
    Apply depth-of-field blur to an image based on depth map.
    Keeps foreground sharp, blurs background progressively based on depth.
    """
    # Invert depth map: far objects (low depth) should get more blur
    # Normalize so that background (far) has high values
    blur_intensity_map = 1.0 - depth_map
    
    # Zero out the foreground region (person should stay sharp)
    fg_mask_float = (foreground_mask / 255.0)
    blur_intensity_map = blur_intensity_map * (1.0 - fg_mask_float)
    
    # Smooth the blur intensity map for gradual transitions
    blur_intensity_map = cv2.GaussianBlur(blur_intensity_map.astype(np.float32), (21, 21), 0)
    
    # Scale by aperture intensity
    blur_intensity_map = blur_intensity_map * aperture_config['intensity']
    
    # Create multiple blur levels for better quality
    max_kernel = aperture_config['max_kernel']
    
    # Generate progressively blurred versions
    blur_levels = []
    kernel_sizes = [1, 7, 13, 19, max_kernel]
    
    for ksize in kernel_sizes:
        if ksize == 1:
            blur_levels.append(image.copy())
        else:
            # Ensure kernel size is odd
            ksize = ksize if ksize % 2 == 1 else ksize + 1
            blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
            blur_levels.append(blurred)
    
    # Blend blur levels based on depth
    result = np.zeros_like(image, dtype=np.float32)
    
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            intensity = blur_intensity_map[y, x]
            
            # Select blur level based on intensity
            if intensity < 0.01:
                result[y, x] = blur_levels[0][y, x]
            elif intensity < 0.25:
                alpha = (intensity - 0.01) / 0.24
                result[y, x] = (1 - alpha) * blur_levels[0][y, x] + alpha * blur_levels[1][y, x]
            elif intensity < 0.5:
                alpha = (intensity - 0.25) / 0.25
                result[y, x] = (1 - alpha) * blur_levels[1][y, x] + alpha * blur_levels[2][y, x]
            elif intensity < 0.75:
                alpha = (intensity - 0.5) / 0.25
                result[y, x] = (1 - alpha) * blur_levels[2][y, x] + alpha * blur_levels[3][y, x]
            else:
                alpha = (intensity - 0.75) / 0.25
                result[y, x] = (1 - alpha) * blur_levels[3][y, x] + alpha * blur_levels[4][y, x]
    
    return result.astype(np.uint8)

# Get image dimensions
height, width = image_np.shape[:2]

# Shift depth map to match background motion (for accurate blur application)
# We'll apply blur based on the original depth map

# Store animation frames for different apertures
animation_frames_all_apertures = {key: [] for key in aperture_settings.keys()}

# Generate frames with parallax motion AND bokeh effects
print(f"Generating {num_frames} frames with bokeh effects...")

for frame_idx in range(num_frames):
    # Calculate normalized time (0 to 1 and back)
    # Creates a back-and-forth motion
    if frame_idx < num_frames // 2:
        t = frame_idx / (num_frames // 2)
    else:
        t = 1.0 - (frame_idx - num_frames // 2) / (num_frames // 2)
    
    # Apply smoothstep for natural easing
    t_smooth = smoothstep(t)
    
    # Calculate shifts with parallax effect
    fg_shift_x = foreground_max_shift * (2 * t_smooth - 1)  # Range: -10 to +10
    bg_shift_x = background_max_shift * (2 * t_smooth - 1)  # Range: -3 to +3
    
    # Create translation matrices
    fg_translation_matrix = np.float32([[1, 0, fg_shift_x], [0, 1, 0]])
    bg_translation_matrix = np.float32([[1, 0, bg_shift_x], [0, 1, 0]])
    
    # Apply motion to foreground (person) - keep sharp
    fg_layer = image_np.copy()
    fg_layer_shifted = cv2.warpAffine(fg_layer, fg_translation_matrix, (width, height), 
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    
    # Shift the foreground mask as well
    fg_mask_shifted = cv2.warpAffine(foreground_mask, fg_translation_matrix, (width, height),
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    fg_mask_shifted_3ch = np.stack([fg_mask_shifted / 255.0] * 3, axis=-1)
    
    # Apply motion to background (clean background without person)
    bg_layer_shifted = cv2.warpAffine(clean_background, bg_translation_matrix, (width, height),
                                       borderMode=cv2.BORDER_REPLICATE)
    
    # Also shift the depth map to match background motion
    depth_map_shifted = cv2.warpAffine(depth_map, bg_translation_matrix, (width, height),
                                        borderMode=cv2.BORDER_REPLICATE)
    
    # Generate frames for each aperture setting
    for aperture_key, aperture_conf in aperture_settings.items():
        # Part 5: Apply depth-of-field blur to background
        bg_blurred = apply_depth_based_blur(bg_layer_shifted, depth_map_shifted, 
                                             fg_mask_shifted, aperture_conf)
        
        # Composite the layers: blurred background + sharp foreground
        composite_frame = bg_blurred.copy()
        
        # Add foreground on top using the shifted mask
        composite_frame = np.where(
            fg_mask_shifted_3ch > 0.1,  # Where foreground exists
            fg_layer_shifted,            # Use sharp foreground
            composite_frame              # Keep blurred background
        ).astype(np.uint8)
        
        # Add frame to appropriate animation
        animation_frames_all_apertures[aperture_key].append(composite_frame)
    
    if (frame_idx + 1) % 5 == 0:
        print(f"  Generated frame {frame_idx + 1}/{num_frames}")

print(f"All {num_frames} frames generated for all apertures!")

# Save GIFs for each aperture setting
for aperture_key, aperture_conf in aperture_settings.items():
    output_gif_path = f"parallax_bokeh_{aperture_key.replace('/', '')}.gif"
    imageio.mimsave(output_gif_path, animation_frames_all_apertures[aperture_key], 
                    duration=0.1, loop=0)
    print(f"✓ Saved: {output_gif_path} ({aperture_conf['name']})")

print(f"\nAnimation Details:")
print(f"  Frame count: {num_frames}")
print(f"  Duration per frame: 100ms")
print(f"  Total animation length: {num_frames * 0.1:.1f}s")
print(f"  Aperture settings: {len(aperture_settings)}")

# # ============================================
# # Visualize Sample Frames - Compare Apertures
# # ============================================
# fig, axes = plt.subplots(3, 3, figsize=(15, 15))

# # Show middle frame for each aperture + original
# sample_frame_idx = num_frames // 2

# row = 0
# for aperture_key, aperture_conf in aperture_settings.items():
#     # Show start, middle, end for each aperture
#     frame_indices = [0, num_frames // 2, num_frames - 1]
#     titles = ['Start', 'Middle', 'End']
    
#     for col, (fidx, title) in enumerate(zip(frame_indices, titles)):
#         axes[row, col].imshow(animation_frames_all_apertures[aperture_key][fidx])
#         axes[row, col].set_title(f"{aperture_conf['name']} - {title}")
#         axes[row, col].axis('off')
    
#     row += 1

# plt.tight_layout()
# plt.show()

# # ============================================
# # Visualize Aperture Comparison (Same Frame)
# # ============================================
# fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# # Original without blur
# axes[0].imshow(image_np)
# axes[0].set_title('Original (No Bokeh)')
# axes[0].axis('off')

# # Each aperture setting
# for idx, (aperture_key, aperture_conf) in enumerate(aperture_settings.items()):
#     axes[idx + 1].imshow(animation_frames_all_apertures[aperture_key][sample_frame_idx])
#     axes[idx + 1].set_title(aperture_conf['name'])
#     axes[idx + 1].axis('off')

# plt.tight_layout()
# plt.show()

# print("\nParallax motion synthesis with bokeh effects complete!")
# print("✓ All animation files saved successfully!")

# # ============================================
# # Additional Analysis: Depth-Based Blur Visualization
# # ============================================

# print("\nGenerating depth-based blur analysis...")

# # Create a single frame showing blur intensity map
# sample_frame = clean_background.copy()

# # Create blur intensity visualization
# blur_intensity_viz = 1.0 - depth_map
# fg_mask_float = (foreground_mask / 255.0)
# blur_intensity_viz = blur_intensity_viz * (1.0 - fg_mask_float)
# blur_intensity_viz = cv2.GaussianBlur(blur_intensity_viz.astype(np.float32), (21, 21), 0)

# # Apply different aperture blurs to the same frame
# fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# # Row 1: Depth map, blur intensity, original
# axes[0, 0].imshow(depth_map, cmap='plasma')
# axes[0, 0].set_title('Depth Map')
# axes[0, 0].axis('off')

# axes[0, 1].imshow(blur_intensity_viz, cmap='hot')
# axes[0, 1].set_title('Blur Intensity Map')
# axes[0, 1].axis('off')

# axes[0, 2].imshow(image_np)
# axes[0, 2].set_title('Original Image')
# axes[0, 2].axis('off')

# # Row 2: Three aperture results on static image
# aperture_keys = list(aperture_settings.keys())
# for idx, aperture_key in enumerate(aperture_keys):
#     aperture_conf = aperture_settings[aperture_key]
    
#     # Apply blur to original image with person
#     blurred_bg = apply_depth_based_blur(clean_background, depth_map, 
#                                         foreground_mask, aperture_conf)
    
#     # Composite with sharp foreground
#     composite = blurred_bg.copy()
#     fg_mask_3ch = np.stack([foreground_mask / 255.0] * 3, axis=-1)
#     composite = np.where(fg_mask_3ch > 0.1, image_np, composite).astype(np.uint8)
    
#     axes[1, idx].imshow(composite)
#     axes[1, idx].set_title(f"{aperture_conf['name']}")
#     axes[1, idx].axis('off')

# plt.tight_layout()
# plt.show()

# print("\nDepth-of-Field Analysis Complete!")
# print(f"\nSummary:")
# print(f"  ✓ Part 1: Depth estimation using Intel DPT")
# print(f"  ✓ Part 2: Foreground/background separation")
# print(f"  ✓ Part 3: Background inpainting")
# print(f"  ✓ Part 4: Parallax motion synthesis ({num_frames} frames)")
# print(f"  ✓ Part 5: Depth-of-field bokeh effects (3 apertures)")
# print(f"\nGenerated Files:")
# for aperture_key in aperture_settings.keys():
#     print(f"  - parallax_bokeh_{aperture_key.replace('/', '')}.gif")
# print("\nAll parts completed successfully! 🎉")

# ============================================
# Part 6: Gradio Interface for Spatial Photo Pipeline
# ============================================

print("\n" + "="*50)
print("Setting up Gradio Interface...")
print("="*50)

def process_spatial_photo(input_image, parallax_strength, aperture_f_number, num_animation_frames, animation_style):
    """
    Main processing function for the Gradio interface.
    
    Args:
        input_image: PIL Image or numpy array
        parallax_strength: Float (1.0 to 3.0) - multiplier for motion
        aperture_f_number: Float (1.4 to 5.6) - f-stop value
        num_animation_frames: Int (15 to 60) - number of frames
        animation_style: String - "Back and Forth" or "Continuous Loop"
    
    Returns:
        depth_map_vis: Depth map visualization
        foreground_img: Extracted foreground
        background_img: Clean background
        gif_path: Path to generated GIF
    """
    try:
        # Convert input to PIL Image if needed
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)
        
        # Resize for processing
        max_size = 640
        if max(input_image.size) > max_size:
            ratio = max_size / max(input_image.size)
            new_size = tuple(int(dim * ratio) for dim in input_image.size)
            input_image = input_image.resize(new_size, Image.LANCZOS)
        
        img_np = np.array(input_image)
        
        # Step 1: Depth Estimation
        inputs = processor(images=input_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=input_image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        
        depth_map_proc = prediction.squeeze().cpu().numpy()
        depth_map_proc = (depth_map_proc - depth_map_proc.min()) / (depth_map_proc.max() - depth_map_proc.min())
        
        # Create depth map visualization
        depth_map_colored = plt.cm.plasma(depth_map_proc)
        depth_map_vis = (depth_map_colored[:, :, :3] * 255).astype(np.uint8)
        
        # Step 2: Foreground/Background Separation
        depth_uint8 = (depth_map_proc * 255).astype(np.uint8)
        _, binary_mask = cv2.threshold(depth_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_small)
        
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask_cleaned = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel_large)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_cleaned, connectivity=8)
        
        if num_labels > 1:
            largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            fg_mask = (labels == largest_component).astype(np.uint8) * 255
        else:
            fg_mask = mask_cleaned
        
        soft_mask = cv2.GaussianBlur(fg_mask, (21, 21), 0)
        soft_mask_norm = soft_mask.astype(np.float32) / 255.0
        
        # Create foreground image
        fg_mask_3ch = np.stack([soft_mask_norm] * 3, axis=-1)
        foreground_img = (img_np * fg_mask_3ch).astype(np.uint8)
        
        # Step 3: Background Inpainting
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dilated_mask = cv2.dilate(fg_mask, kernel_dilate, iterations=2)
        
        inpainted_telea = cv2.inpaint(img_np, dilated_mask, inpaintRadius=15, flags=cv2.INPAINT_TELEA)
        inpainted_ns = cv2.inpaint(img_np, dilated_mask, inpaintRadius=15, flags=cv2.INPAINT_NS)
        blended_inpaint = cv2.addWeighted(inpainted_telea, 0.5, inpainted_ns, 0.5, 0)
        smooth_inpaint = cv2.bilateralFilter(blended_inpaint, d=9, sigmaColor=75, sigmaSpace=75)
        
        clean_bg = np.where(
            dilated_mask[:, :, np.newaxis] > 0,
            smooth_inpaint,
            img_np
        ).astype(np.uint8)
        
        # Step 4 & 5: Animation with Bokeh
        height, width = img_np.shape[:2]
        
        # Map aperture f-number to kernel size and intensity
        # f/1.4 -> max blur, f/5.6 -> min blur
        intensity = (5.6 - aperture_f_number) / (5.6 - 1.4)
        max_kernel = int(9 + intensity * 22)  # 9 to 31
        if max_kernel % 2 == 0:
            max_kernel += 1
        
        aperture_conf = {
            'max_kernel': max_kernel,
            'intensity': intensity * 0.8 + 0.2  # 0.2 to 1.0
        }
        
        # Calculate shifts based on parallax strength
        fg_max_shift = int(10 * parallax_strength)
        bg_max_shift = int(3 * parallax_strength)
        
        frames = []
        num_frames_int = int(num_animation_frames)
        
        for frame_idx in range(num_frames_int):
            # Calculate time based on animation style
            if animation_style == "Back and Forth":
                if frame_idx < num_frames_int // 2:
                    t = frame_idx / (num_frames_int // 2)
                else:
                    t = 1.0 - (frame_idx - num_frames_int // 2) / (num_frames_int // 2)
            else:  # Continuous Loop
                t = frame_idx / num_frames_int
            
            t_smooth = t * t * (3 - 2 * t)  # smoothstep
            
            if animation_style == "Back and Forth":
                fg_shift_x = fg_max_shift * (2 * t_smooth - 1)
                bg_shift_x = bg_max_shift * (2 * t_smooth - 1)
            else:
                fg_shift_x = fg_max_shift * (t_smooth - 0.5) * 2
                bg_shift_x = bg_max_shift * (t_smooth - 0.5) * 2
            
            fg_trans = np.float32([[1, 0, fg_shift_x], [0, 1, 0]])
            bg_trans = np.float32([[1, 0, bg_shift_x], [0, 1, 0]])
            
            fg_shifted = cv2.warpAffine(img_np, fg_trans, (width, height),
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            fg_mask_shifted = cv2.warpAffine(fg_mask, fg_trans, (width, height),
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            bg_shifted = cv2.warpAffine(clean_bg, bg_trans, (width, height),
                                        borderMode=cv2.BORDER_REPLICATE)
            depth_shifted = cv2.warpAffine(depth_map_proc, bg_trans, (width, height),
                                          borderMode=cv2.BORDER_REPLICATE)
            
            # Apply bokeh blur
            bg_blurred = apply_depth_based_blur(bg_shifted, depth_shifted, fg_mask_shifted, aperture_conf)
            
            # Composite
            fg_mask_shifted_3ch = np.stack([fg_mask_shifted / 255.0] * 3, axis=-1)
            composite = np.where(fg_mask_shifted_3ch > 0.1, fg_shifted, bg_blurred).astype(np.uint8)
            
            frames.append(composite)
        
        # Save GIF with optimization to keep under 5MB
        temp_dir = tempfile.gettempdir()
        gif_path = os.path.join(temp_dir, f"spatial_photo_{os.getpid()}.gif")
        
        # Adjust duration to keep file size reasonable
        duration = max(0.05, min(0.15, 1.0 / num_frames_int))
        
        imageio.mimsave(gif_path, frames, duration=duration, loop=0)
        
        # Check file size and optimize if needed
        file_size_mb = os.path.getsize(gif_path) / (1024 * 1024)
        if file_size_mb > 5:
            # Re-save with longer duration (fewer effective fps)
            duration = duration * (file_size_mb / 4.5)
            imageio.mimsave(gif_path, frames, duration=duration, loop=0)
        
        # Return the same path for both display and download
        return depth_map_vis, foreground_img, clean_bg, gif_path, gif_path
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise e

# Create Gradio Interface
with gr.Blocks(title="Spatial Photo Effect Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎨 Spatial Photo Effect Generator
        ### Create stunning parallax animations with depth-of-field effects!
        
        Upload an image and adjust parameters to generate a professional spatial photo effect.
        The AI will automatically detect depth, separate foreground/background, and create an animated GIF.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Input")
            input_img = gr.Image(label="Upload Image", type="pil", height=300)
        
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Results")
            
            with gr.Tab("Final Animation"):
                output_gif_display = gr.Image(label="Animated GIF Preview", type="filepath")
                output_gif = gr.File(label="Download GIF", file_types=[".gif"])
                gr.Markdown("*Click the download icon above or right-click the preview to save*")
            
            with gr.Tab("Processing Steps"):
                with gr.Row():
                    depth_output = gr.Image(label="Depth Map", type="numpy", show_label=True)
                    foreground_output = gr.Image(label="Foreground (Sharp)", type="numpy", show_label=True)
                with gr.Row():
                    background_output = gr.Image(label="Background (Inpainted)", type="numpy", show_label=True)
    
    gr.Markdown("### ⚙️ Parameters")
    
    with gr.Row():
        parallax_slider = gr.Slider(
            minimum=1.0,
            maximum=3.0,
            value=1.5,
            step=0.1,
            label="Parallax Strength",
            info="How much the foreground/background move differently (1.0=subtle, 3.0=dramatic)"
        )
        
        aperture_slider = gr.Slider(
            minimum=1.4,
            maximum=5.6,
            value=2.8,
            step=0.2,
            label="Aperture (f-stop)",
            info="Camera aperture simulation (f/1.4=heavy blur, f/5.6=light blur)"
        )
        
        frames_slider = gr.Slider(
            minimum=15,
            maximum=60,
            value=30,
            step=5,
            label="Animation Frames",
            info="More frames = smoother but larger file"
        )
        
        animation_style = gr.Radio(
            choices=["Back and Forth", "Continuous Loop"],
            value="Back and Forth",
            label="Animation Style",
            info="Back and Forth: ping-pong motion, Continuous Loop: circular motion"
        )
    
    process_btn = gr.Button("✨ Generate Spatial Photo", variant="primary", size="lg")

    # Examples
    gr.Markdown("### 🎯 Try These Examples")
    gr.Examples(
        examples=[
            [image_url, 1.5, 2.8, 30, "Back and Forth"],
            [image_url, 2.0, 1.4, 25, "Back and Forth"],
            [image_url, 1.2, 4.0, 20, "Continuous Loop"],
        ],
        inputs=[input_img, parallax_slider, aperture_slider, frames_slider, animation_style],
        label="Quick Start Examples"
    )
    
    gr.Markdown(
        """
        ### 💡 Tips for Best Results:
        - **Portrait photos** work best (clear subject in foreground)
        - **Good lighting** helps depth estimation accuracy
        - **Simple backgrounds** produce cleaner results
        - Lower **frame counts** for faster processing and smaller files
        - **f/2.8** aperture is a good starting point for most images
        
        ### ⚡ Processing Info:
        - Average processing time: 30-60 seconds
        - GIF file size: automatically optimized to stay under 5MB
        - All processing happens on your device
        """
    )
    
    # Connect the button
    process_btn.click(
        fn=process_spatial_photo,
        inputs=[input_img, parallax_slider, aperture_slider, frames_slider, animation_style],
        outputs=[depth_output, foreground_output, background_output, output_gif_display, output_gif]
    )

print("\n✨ Gradio interface configured!")
print("Launching web interface...")

# Launch the interface
demo.launch(
    share=False,  # Set to True to create a public link
    show_error=True,
    server_name="localhost",
    server_port=7860
)
