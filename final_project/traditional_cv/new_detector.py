# import cv2
# import numpy as np

# def order_points(pts):
#     # Orders coordinates: top-left, top-right, bottom-right, bottom-left
#     rect = np.zeros((4, 2), dtype="float32")
#     s = pts.sum(axis=1)
#     rect[0] = pts[np.argmin(s)]
#     rect[2] = pts[np.argmax(s)]
#     diff = np.diff(pts, axis=1)
#     rect[1] = pts[np.argmin(diff)]
#     rect[3] = pts[np.argmax(diff)]
#     return rect

# def analyze_sides(contour, width, height):
#     # Get the rotated rectangle
#     rect = cv2.minAreaRect(contour)
#     box = cv2.boxPoints(rect)
#     box = box.astype(int)
#     box = order_points(box)
    
#     cnt_points = contour.reshape(-1, 2)
    
#     # Associate contour points with the 4 geometric corners
#     corner_indices = []
#     for b_point in box:
#         distances = np.linalg.norm(cnt_points - b_point, axis=1)
#         corner_indices.append(np.argmin(distances))
#     corner_indices.sort()
    
#     # Split contour into 4 segments
#     segments = []
#     for i in range(4):
#         p1 = corner_indices[i]
#         p2 = corner_indices[(i + 1) % 4]
#         if p1 < p2:
#             segment = cnt_points[p1:p2]
#         else:
#             segment = np.vstack((cnt_points[p1:], cnt_points[:p2]))
#         segments.append(segment)
        
#     flat_sides = 0
#     # Tolerance: How much wiggle room allowed for a line to be "flat"
#     # Adjusted to 6% of piece size for better accuracy
#     piece_size = max(width, height)
#     threshold = piece_size * 0.06 
    
#     for segment in segments:
#         if len(segment) < 10: continue 
        
#         # Fit line
#         [vx, vy, x, y] = cv2.fitLine(segment, cv2.DIST_L2, 0, 0.01, 0.01)
        
#         # Check maximum deviation from that line
#         max_deviation = 0
#         for point in segment:
#             px, py = point
#             dist = abs(vx * (py - y) - vy * (px - x))
#             if dist > max_deviation:
#                 max_deviation = dist
        
#         if max_deviation < threshold:
#             flat_sides += 1
            
#     return flat_sides

# def process_puzzle(image_path):
#     img = cv2.imread(image_path)
#     if img is None:
#         print("Error: Image not found.")
#         return

#     # Resize if image is massive (speeds up processing)
#     h, w = img.shape[:2]
#     if w > 2000:
#         scale = 2000 / w
#         img = cv2.resize(img, (int(w*scale), int(h*scale)))
    
#     output_img = img.copy()
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Gaussian Blur to smooth out noise/texture in the carpet
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
#     # Thresholding: 
#     # Using 230 to catch pieces even if lighting is slightly dim.
#     # THRESH_BINARY_INV means: Background (White) -> Black, Pieces (Darker) -> White
#     _, thresh = cv2.threshold(blurred, 230, 255, cv2.THRESH_BINARY_INV)
    
#     # Morphological Operations to remove small noise holes
#     kernel = np.ones((3,3), np.uint8)
#     thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
#     # Show the "Mask" so user can debug
#     cv2.imshow("Debug: Black & White Mask", thresh)
#     cv2.waitKey(1)
    
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     stats = {"Corner": 0, "Edge": 0, "Interior": 0}
    
#     total_img_area = img.shape[0] * img.shape[1]
    
#     print(f"Total Contours Found: {len(contours)}")
    
#     piece_count = 0
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
        
#         # FILTER 1: Ignore Tiny Noise (Dust)
#         if area < (total_img_area * 0.005): # Less than 0.5% of image
#             continue
            
#         # FILTER 2: Ignore The Whole Image Frame
#         if area > (total_img_area * 0.90): # More than 90% of image
#             continue
            
#         x, y, w, h = cv2.boundingRect(cnt)
        
#         flat_count = analyze_sides(cnt, w, h)
        
#         # Classification Logic
#         label = ""
#         color = (0, 0, 0)
        
#         if flat_count >= 2:
#             label = "Corner"
#             stats["Corner"] += 1
#             color = (0, 0, 255) # Red
#         elif flat_count == 1:
#             label = "Edge"
#             stats["Edge"] += 1
#             color = (0, 255, 0) # Green
#         else:
#             label = "Interior"
#             stats["Interior"] += 1
#             color = (255, 0, 0) # Blue
            
#         cv2.drawContours(output_img, [cnt], -1, color, 3)
#         cv2.putText(output_img, f"{label}", (x, y - 10), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
#         piece_count += 1
#         print(f"Piece {piece_count}: {label} ({flat_count} flat sides)")
        
#         # Check for 'q' key to quit
#         cv2.imshow("Puzzle Detection", output_img)
#         # Wait a bit longer (10ms) and check for 'q' or ESC
#         key = cv2.waitKey(10) & 0xFF
#         if key == ord('q') or key == 27:
#             print("Process interrupted by user.")
#             cv2.destroyAllWindows()
#             cv2.waitKey(1)
#             return

#     print("\n--- Final Counts ---")
#     for k, v in stats.items():
#         print(f"{k}: {v}")
        
#     cv2.imshow("Final Detection", output_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# # Run it
# process_puzzle('test.png')

import cv2
import numpy as np

def detect_flat_sides_robust(contour):
    """
    Robustly detects flat sides by approximating the contour as a polygon
    and counting how many segments are 'long' (approx the length of the piece).
    """
    # 1. Get accurate piece size using minAreaRect (handles rotation better)
    rect = cv2.minAreaRect(contour)
    w, h = rect[1]
    piece_length_max = max(w, h)

    # 2. Approximate the contour to a polygon
    # Lower epsilon (0.02) to preserve tab/hole shapes better. 
    # If too high, a side with a shallow hole becomes a straight line.
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    
    # 3. Analyze the segments of the polygon
    flat_sides = 0
    # Threshold: A segment must be significant to be a "side".
    # A flat side is roughly 100% of the grid length.
    # A shoulder of a tab is roughly 30-40%.
    # We set threshold to 55% of the max dimension.
    threshold = piece_length_max * 0.55
    
    points = approx[:, 0, :] # Flatten array
    num_points = len(points)
    
    for i in range(num_points):
        p1 = points[i]
        p2 = points[(i + 1) % num_points] # Wrap around to first point
        
        # Calculate distance between vertices
        length = np.linalg.norm(p1 - p2)
        
        # LOGIC:
        # A flat side is one continuous straight line.
        # A tab/hole side is broken into 3 smaller lines (shoulder, tab-side, shoulder).
        # Therefore, only flat sides will be long.
        if length > threshold:
            flat_sides += 1

    return flat_sides

def process_puzzle_v3(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return

    # Resize for consistency if image is huge
    h, w = img.shape[:2]
    if w > 2000:
        scale = 2000 / w
        img = cv2.resize(img, (int(w*scale), int(h*scale)))

    output_img = img.copy()
    
    # 1. Preprocessing (Standard)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 230, 255, cv2.THRESH_BINARY_INV)
    
    # Clean up noise
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    stats = {"Corner": 0, "Edge": 0, "Interior": 0}
    
    print(f"Total Contours Found: {len(contours)}\n")
    
    piece_id = 0
    total_area = img.shape[0] * img.shape[1]
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Filter Noise & Frame
        if area < (total_area * 0.005) or area > (total_area * 0.90):
            continue
            
        piece_id += 1
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Use the new robust detection
        flat_count = detect_flat_sides_robust(cnt)
        
        # Sanity cap (a piece can't have > 4 sides)
        if flat_count > 4: flat_count = 4
        
        # Classification
        label = ""
        color = (0,0,0)
        
        if flat_count >= 2:
            label = "Corner"
            stats["Corner"] += 1
            color = (0, 0, 255) # Red
        elif flat_count == 1:
            label = "Edge"
            stats["Edge"] += 1
            color = (0, 200, 0) # Green
        else:
            label = "Interior"
            stats["Interior"] += 1
            color = (255, 0, 0) # Blue
            
        # visual output
        cv2.drawContours(output_img, [cnt], -1, color, 3)
        # Draw the approximate polygon to show what the computer 'sees' (Cyan thin line)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        cv2.drawContours(output_img, [approx], -1, (255, 255, 0), 2)
        
        cv2.putText(output_img, f"{label} ({flat_count})", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        print(f"Piece {piece_id}: {label} ({flat_count} flat sides)")

        # Check for 'q' key to quit
        cv2.imshow("Puzzle Detection", output_img)
        # Wait a bit longer (10ms) and check for 'q' or ESC
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q') or key == 27:
            print("Process interrupted by user.")
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            return

    print("\n--- Final Counts ---")
    for k, v in stats.items():
        print(f"{k}: {v}")
        
    cv2.imshow("Robust Puzzle Detection", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run
process_puzzle_v3('test.png')
