import cv2
import numpy as np
import os
import sys
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.util import img_as_float

def load_and_preprocess_image(path: str, size: tuple) -> tuple:
    # Loads, resizes, and converts an image to grayscale.
    image_bgr = cv2.imread(path)
    if image_bgr is None:
        sys.exit(f"Error: Could not read image from '{path}'")
        
    image_bgr = cv2.resize(image_bgr, size)
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    return image_bgr, image_gray

def create_rectangular_contour(center: tuple, width: int, height: int, num_points: int) -> np.ndarray:
    # Creates an initial rectangular contour for the snake algorithm.
    x, y = center
    half_w, half_h = width // 2, height // 2
    
    top = np.array([np.linspace(x - half_w, x + half_w, num_points), np.full(num_points, y - half_h)]).T
    right = np.array([np.full(num_points, x + half_w), np.linspace(y - half_h, y + half_h, num_points)]).T
    bottom = np.array([np.linspace(x + half_w, x - half_w, num_points), np.full(num_points, y + half_h)]).T
    left = np.array([np.full(num_points, x - half_w), np.linspace(y + half_h, y - half_h, num_points)]).T
    
    contour = np.concatenate([top, right, bottom, left])
    # Convert (x, y) to scikit-image's expected (row, col) format
    return np.flip(contour, axis=1)

def find_snake_contour(image_gray: np.ndarray, initial_contour: np.ndarray, alpha: float, beta: float, gamma: float) -> np.ndarray:
    # Finds the optimal contour using the active contour model.
    image_float = img_as_float(image_gray)
    
    snake = active_contour(
        gaussian(image_float, sigma=3, preserve_range=False),
        initial_contour,
        alpha=alpha,
        beta=beta,
        gamma=gamma
    )
    return snake

def visualize_and_save_contour(
    image_bgr: np.ndarray, 
    snake_contour: np.ndarray, 
    output_path: str, 
    show_window: bool
):
    # Draws the final contour, saves the image, and optionally displays it.
    output_image = image_bgr.copy()
    
    # Convert contour points for OpenCV drawing
    points = snake_contour[:, [1, 0]].astype(np.int32)
    
    cv2.polylines(output_image, [points], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.imwrite(output_path, output_image)
    print(f"Result saved to: {output_path}")

    if show_window:
        cv2.imshow("Active Contour Result", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# --- Main Execution ---
if __name__ == "__main__":
    
    # --- Configuration ---
    IMAGE_PATH = "images/coin.png"
    OUTPUT_DIR = "output"
    VISUALIZE = True  # Set to False to disable display window

    TARGET_SIZE = (400, 400)
    
    # Initial contour parameters
    CONTOUR_CENTER = (200, 200)
    CONTOUR_SIZE = (300, 300)
    POINTS_PER_SIDE = 100

    # Active contour algorithm parameters
    ALPHA = 0.015  # Elasticity
    BETA = 10.0    # Smoothness
    GAMMA = 0.001  # Step size
    
    # --- Script ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Processing image: {IMAGE_PATH}")
    color_img, gray_img = load_and_preprocess_image(IMAGE_PATH, TARGET_SIZE)

    init_contour = create_rectangular_contour(CONTOUR_CENTER, *CONTOUR_SIZE, POINTS_PER_SIDE)

    final_contour = find_snake_contour(gray_img, init_contour, ALPHA, BETA, GAMMA)
    print("Contour optimization complete.")

    base_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    output_filename = os.path.join(OUTPUT_DIR, f"{base_name}_snake_contour.png")
    
    visualize_and_save_contour(color_img, final_contour, output_filename, show_window=VISUALIZE)