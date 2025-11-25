import cv2
import numpy as np

def get_directional_flatness(contour, width, height):
    """
    Checks 4 sides (Top, Bottom, Left, Right) to see if they are flat.
    Returns a dict: {'T': True/False, 'B':..., 'L':..., 'R':...}
    """
    # 1. Approximate the contour to a polygon to simplify jagged edges
    # Use 0.02 epsilon (robust) instead of 0.04
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    
    # We look for long straight lines near the bounding box edges
    flat_flags = {'T': False, 'B': False, 'L': False, 'R': False}
    
    points = approx[:, 0, :]
    num_points = len(points)
    
    # Threshold: A line must be at least 50% of the piece dimension (robust)
    # and it must be close to the boundary.
    min_length = max(width, height) * 0.5
    tol = 15 # pixels tolerance from the edge
    
    for i in range(num_points):
        p1 = points[i]
        p2 = points[(i + 1) % num_points]
        
        length = np.linalg.norm(p1 - p2)
        
        if length < min_length:
            continue
        
        # Check if line is Horizontal (dy is small)
        if abs(p1[1] - p2[1]) < tol:
            # Check if it is at the Top (y near 0) or Bottom (y near height)
            center_y = (p1[1] + p2[1]) / 2
            if center_y < tol: flat_flags['T'] = True
            if center_y > height - tol: flat_flags['B'] = True
            
        # Check if line is Vertical (dx is small)
        if abs(p1[0] - p2[0]) < tol:
            # Check if it is Left (x near 0) or Right (x near width)
            center_x = (p1[0] + p2[0]) / 2
            if center_x < tol: flat_flags['L'] = True
            if center_x > width - tol: flat_flags['R'] = True
            
    return flat_flags

def get_grid_position(flags):
    """Maps flat flags to a (row, col) in a 3x3 grid"""
    row, col = -1, -1
    
    # Vertical Position
    if flags['T']: row = 0
    elif flags['B']: row = 2
    else: row = 1
    
    # Horizontal Position
    if flags['L']: col = 0
    elif flags['R']: col = 2
    else: col = 1
    
    return row, col

def assemble_puzzle(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to read image at {image_path}")
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Preprocessing
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 230, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Canvas for 3x3 grid
    # Assuming pieces are roughly same size, we take the max found size
    max_w, max_h = 0, 0
    pieces_data = []
    
    total_area = img.shape[0] * img.shape[1]

    # 1. Extraction Phase
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (total_area * 0.005) or area > (total_area * 0.90): continue
        
        x, y, w, h = cv2.boundingRect(cnt)
        max_w = max(max_w, w)
        max_h = max(max_h, h)
        
        # Extract the specific piece image
        # Note: We subtract x,y from contour to make it relative to the crop
        piece_cnt = cnt - [x, y]
        
        roi = original[y:y+h, x:x+w]
        
        # Create a mask for the piece using the contour
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [piece_cnt], -1, 255, -1) # Fill contour
        
        # Detect Orientation
        flags = get_directional_flatness(piece_cnt, w, h)
        row, col = get_grid_position(flags)
        
        pieces_data.append({
            "img": roi,
            "mask": mask,
            "row": row,
            "col": col,
            "w": w,
            "h": h,
            "flags": flags
        })

    if not pieces_data:
        print("No puzzle pieces detected. Check image or thresholds.")
        return

    # 2. Assembly Phase
    # We calculate the canvas size. 
    # Since pieces overlap (tabs/holes), simply adding widths is too wide.
    # We estimate overlap is roughly 15-20% of piece size.
    overlap_ratio = 0.2
    effective_w = int(max_w * (1 - overlap_ratio))
    effective_h = int(max_h * (1 - overlap_ratio))
    
    canvas_w = int(effective_w * 3 + max_w * overlap_ratio)
    canvas_h = int(effective_h * 3 + max_h * overlap_ratio)
    
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    
    print(f"Detected {len(pieces_data)} pieces. Assembling...")

    for p in pieces_data:
        r, c = p['row'], p['col']
        piece_img = p['img']
        mask = p['mask']
        ph, pw = piece_img.shape[:2]
        
        # Calculate coordinate to paste
        # This is an approximation. Real puzzle logic requires pixel matching.
        start_y = r * effective_h
        start_x = c * effective_w
        
        # Center the piece slightly in its 'slot' if it's smaller than max
        # to prevent gaps
        
        # Paste piece (Naive overlay)
        # We need to handle alpha blending or simple overwrite.
        
        # Handle boundary checks to prevent size mismatch errors
        end_y = min(start_y + ph, canvas.shape[0])
        end_x = min(start_x + pw, canvas.shape[1])
        h_paste, w_paste = end_y - start_y, end_x - start_x
        
        if h_paste <= 0 or w_paste <= 0: continue

        # Simple overwrite:
        target_area = canvas[start_y:end_y, start_x:end_x]
        piece_part = piece_img[0:h_paste, 0:w_paste]
        mask_part = mask[0:h_paste, 0:w_paste]
        
        # Use the precise mask we created earlier
        mask_inv = cv2.bitwise_not(mask_part)
        
        # Black-out area in canvas
        img1_bg = cv2.bitwise_and(target_area, target_area, mask=mask_inv)
        # Take only region of logo from logo image.
        img2_fg = cv2.bitwise_and(piece_part, piece_part, mask=mask_part)
        
        dst = cv2.add(img1_bg, img2_fg)
        canvas[start_y:end_y, start_x:end_x] = dst
        
        print(f"Placed Piece at Grid ({r},{c}) based on flags: {p['flags']}")

    cv2.imshow("Assembled Puzzle (Approximation)", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("assembled_result.jpg", canvas)

# Run
assemble_puzzle('test.png')
