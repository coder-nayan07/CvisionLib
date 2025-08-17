import cv2
import numpy as np
import os
import sys

def load_and_prepare_images(path_l: str, path_r: str, size: tuple = None) -> tuple:
    """Loads, resizes, and converts a stereo image pair to grayscale."""
    img_l = cv2.imread(path_l, cv2.IMREAD_GRAYSCALE)
    img_r = cv2.imread(path_r, cv2.IMREAD_GRAYSCALE)

    if img_l is None or img_r is None:
        sys.exit(f"Error: Could not load one or both images.")
    
    if size:
        img_l = cv2.resize(img_l, size)
        img_r = cv2.resize(img_r, size)
        
    return img_l, img_r

def manual_stereo_block_matching(img_l: np.ndarray, img_r: np.ndarray, block_size: int, search_range: int) -> np.ndarray:
    """
    Manually computes a disparity map using the Sum of Absolute Differences (SAD) method.
    """
    h, w = img_l.shape
    disparity_map = np.zeros((h, w), np.uint8)
    half_block = block_size // 2

    # Iterate over each pixel in the left image (skipping the border)
    for y in range(half_block, h - half_block):
        # Print progress to show it's working
        if y % 10 == 0:
            print(f"Processing row {y}/{h}...")
            
        for x in range(half_block, w - half_block):
            best_disparity = 0
            min_sad = float('inf')
            
            # Extract the reference block from the left image
            left_block = img_l[y - half_block : y + half_block + 1,
                               x - half_block : x + half_block + 1]

            # Search for the best match in the right image along the epipolar line
            for d in range(search_range):
                x_right = x - d
                
                # Ensure the block is within the bounds of the right image
                if x_right < half_block:
                    break

                right_block = img_r[y - half_block : y + half_block + 1,
                                    x_right - half_block : x_right + half_block + 1]
                
                # Calculate Sum of Absolute Differences (SAD)
                sad = np.sum(np.abs(np.subtract(left_block, right_block, dtype=np.float32)))
                
                # If this is a better match, store the disparity
                if sad < min_sad:
                    min_sad = sad
                    best_disparity = d
            
            disparity_map[y, x] = best_disparity
            
    return disparity_map

# --- Main Execution ---
if __name__ == "__main__":

    # --- Configuration ---
    IMAGE_PATHS = {
        'left': "images/stereo_img_left.png",
        'right': "images/stereo_img_right.png"
    }
    OUTPUT_DIR = "output"
    # Use a smaller size for faster processing with the manual method
    TARGET_SIZE = (480, 320)

    # Stereo Matching Parameters
    BLOCK_SIZE = 15          # Size of the matching window (must be odd)
    NUM_DISPARITIES = 64     # Maximum disparity to search for

    # --- Script ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    img_left, img_right = load_and_prepare_images(IMAGE_PATHS['left'], IMAGE_PATHS['right'], TARGET_SIZE)
    
    print("Starting manual stereo block matching. This will be very slow.")
    disparity = manual_stereo_block_matching(img_left, img_right, BLOCK_SIZE, NUM_DISPARITIES)
    print("Processing complete.")
    
    # Normalize the disparity map for visualization
    disparity_visual = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disparity_visual = cv2.convertScaleAbs(disparity_visual)

    # Save and display the result
    output_filename = os.path.join(OUTPUT_DIR, "disparity_map_manual.png")
    cv2.imwrite(output_filename, disparity_visual)
    print(f"Disparity map saved to: {output_filename}")
    
    cv2.imshow('Left Image', img_left)
    cv2.imshow('Right Image', img_right)
    cv2.imshow('Manual Disparity Map', disparity_visual)
    cv2.waitKey(0)
    cv2.destroyAllWindows()