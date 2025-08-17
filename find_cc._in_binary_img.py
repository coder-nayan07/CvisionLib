import cv2
import numpy as np
import os
import sys

def load_and_binarize_image(path: str, threshold: int = 127) -> np.ndarray:
    # Loads an image in grayscale and applies a binary threshold.
    gray_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray_image is None:
        sys.exit(f"Error: Could not read image from '{path}'")
    
    _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

def find_connected_components(binary_image: np.ndarray, connectivity: int = 8) -> tuple:
    # Finds all connected components in a binary image using DFS.
    height, width = binary_image.shape
    labeled_image = np.zeros_like(binary_image, dtype=np.int32)
    current_label = 0

    if connectivity == 4:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # 4-way
    else:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)] # 8-way

    for y in range(height):
        for x in range(width):
            # If the pixel is part of a component but not yet labeled
            if binary_image[y, x] == 255 and labeled_image[y, x] == 0:
                current_label += 1
                stack = [(y, x)]
                
                while stack:
                    cy, cx = stack.pop()
                    if labeled_image[cy, cx] == 0:
                        labeled_image[cy, cx] = current_label
                        
                        for dy, dx in directions:
                            ny, nx = cy + dy, cx + dx
                            
                            # Check bounds and if the neighbor is an unlabeled component pixel
                            if (0 <= ny < height and 0 <= nx < width and
                                    binary_image[ny, nx] == 255 and labeled_image[ny, nx] == 0):
                                stack.append((ny, nx))
                                
    return labeled_image, current_label

def visualize_and_save_components(
    labeled_image: np.ndarray,
    num_components: int,
    output_path: str,
    show_window: bool
):
    # Creates a colorized version of the labeled image for visualization.
    if num_components == 0:
        print("No components found to visualize.")
        return

    # Generate a unique color for each component label (background is black)
    colors = np.random.randint(60, 256, size=(num_components + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0] # Background color
    
    color_map = colors[labeled_image]
    
    cv2.imwrite(output_path, color_map)
    print(f"Result saved to: {output_path}")

    if show_window:
        cv2.imshow("Connected Components", color_map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# --- Main Execution ---
if __name__ == "__main__":

    # --- Configuration ---
    IMAGE_PATH = "images/bin2.jpg"
    OUTPUT_DIR = "output"
    VISUALIZE = True  # Set to False to disable the display window
    CONNECTIVITY = 8  # Use 4 or 8 for component analysis

    # --- Script ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load and prepare the image
    bin_image = load_and_binarize_image(IMAGE_PATH)

    # 2. Find connected components
    labeled_img, num_labels = find_connected_components(bin_image, CONNECTIVITY)
    print(f"Found {num_labels} connected component(s).")

    # 3. Visualize and save the result
    base_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    output_filename = os.path.join(OUTPUT_DIR, f"{base_name}_components.png")

    visualize_and_save_components(labeled_img, num_labels, output_filename, show_window=VISUALIZE)