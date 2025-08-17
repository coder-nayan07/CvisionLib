import cv2
import numpy as np
import os
import sys
import time


def load_images(image_path: str, template_path: str) -> tuple:
    """Loads the main image and template image."""
    image = cv2.imread(image_path)
    template = cv2.imread(template_path)

    if image is None:
        sys.exit(f"Error: Could not load image from '{image_path}'")
    if template is None:
        sys.exit(f"Error: Could not load template from '{template_path}'")
        
    return image, template

def manual_template_matching(image_gray: np.ndarray, template_gray: np.ndarray) -> tuple:
    """
    Performs template matching using a manual sliding window with SSD.
    Note: This is very slow and intended for educational purposes.
    """
    img_h, img_w = image_gray.shape
    temp_h, temp_w = template_gray.shape
    
    min_ssd = float('inf')
    best_loc = (0, 0)
    
    # Slide the template over the main image
    for y in range(img_h - temp_h + 1):
        for x in range(img_w - temp_w + 1):
            # Extract the region of interest (ROI)
            roi = image_gray[y:y+temp_h, x:x+temp_w]
            
            # Compute Sum of Squared Differences (SSD)
            ssd = np.sum((roi.astype(np.float32) - template_gray.astype(np.float32)) ** 2)
            
            if ssd < min_ssd:
                min_ssd = ssd
                best_loc = (x, y)
                
    return best_loc

def opencv_template_matching(image_gray: np.ndarray, template_gray: np.ndarray, method) -> tuple:
    """Performs template matching using OpenCV's optimized function."""
    result = cv2.matchTemplate(image_gray, template_gray, method)
    
    # For SSD-based methods, the best match is the minimum value.
    # For correlation-based methods, it's the maximum.
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        top_left = min_loc
    else:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        top_left = max_loc
        
    return top_left

def save_and_visualize_result(
    image_color: np.ndarray,
    template: np.ndarray,
    location: tuple,
    output_path: str,
    show_window: bool
):
    """Draws a bounding box, saves the result, and optionally displays it."""
    output_image = image_color.copy()
    top_left = location
    h, w = template.shape[:2]
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    # Draw the bounding box
    cv2.rectangle(output_image, top_left, bottom_right, (0, 255, 0), 2)
    
    # Save the result
    cv2.imwrite(output_path, output_image)
    print(f"Result saved to: {output_path}")

    # Display the result if requested
    if show_window:
        cv2.imshow("Detected Match", output_image)
        cv2.imshow("Template", template)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

## Main Execution

if __name__ == "__main__":

    # --- Configuration ---
    IMAGE_PATH = "images/template_matching.jpg"
    TEMPLATE_PATH = "images/template.png" # The template should be a separate file
    OUTPUT_DIR = "output"
    
    # --- Control Flags ---
    # Set to True to use the fast OpenCV function, False for the slow manual method
    USE_OPENCV_OPTIMIZED = True
    VISUALIZE = True
    
    # OpenCV's matching method (TM_SQDIFF_NORMED is often a good choice)
    MATCHING_METHOD = cv2.TM_SQDIFF_NORMED
    
    # --- Script ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image, template = load_images(IMAGE_PATH, TEMPLATE_PATH)
    
    # Use grayscale images for matching, as color adds complexity and computation
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gs = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    start_time = time.time()
    
    if USE_OPENCV_OPTIMIZED:
        print("Using OpenCV's optimized template matching...")
        best_location = opencv_template_matching(image_gs, template_gs, MATCHING_METHOD)
    else:
        print("Using manual sliding window matching. This will be very slow.")
        best_location = manual_template_matching(image_gs, template_gs)

    end_time = time.time()
    print(f"Matching complete. Found best match at: {best_location}")
    print(f"Time taken: {end_time - start_time:.4f} seconds.")

    # Prepare output name and visualize/save the result
    method_name = "opencv" if USE_OPENCV_OPTIMIZED else "manual"
    base_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    output_filename = os.path.join(OUTPUT_DIR, f"{base_name}_matched_{method_name}.png")

    save_and_visualize_result(
        image, template, best_location, output_filename, show_window=VISUALIZE
    )