import cv2
import numpy as np
import os
import sys

# --- Image Loading and Saving ---

def load_and_prepare_image(path: str, size: tuple = None) -> tuple:
    """Loads an image, optionally resizes it, and prepares color and grayscale versions."""
    img_color = cv2.imread(path)
    if img_color is None:
        sys.exit(f"Error: Could not read image from '{path}'")
    
    if size:
        img_color = cv2.resize(img_color, size)
    
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    return img_color, img_gray

def save_image(img: np.ndarray, dir: str, name: str):
    """Saves an image to a specified directory."""
    cv2.imwrite(os.path.join(dir, name), img)

# --- Filter Functions ---

def apply_box_filter(image: np.ndarray) -> np.ndarray:
    """Applies a 5x5 box filter for smoothing."""
    kernel = np.ones((5, 5), np.float32) / 25
    return cv2.filter2D(image, -1, kernel)

def apply_gaussian_filter(image: np.ndarray) -> np.ndarray:
    """Applies a 5x5 Gaussian filter for smoothing."""
    return cv2.GaussianBlur(image, (5, 5), 0)

def apply_sobel_filter(image: np.ndarray) -> np.ndarray:
    """Applies a Sobel filter to detect edges."""
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return cv2.convertScaleAbs(magnitude)

def apply_laplacian_filter(image: np.ndarray) -> np.ndarray:
    """Applies a Laplacian filter to detect edges."""
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return cv2.convertScaleAbs(laplacian)

def apply_motion_blur(image: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """Applies a directional motion blur."""
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel /= kernel_size
    return cv2.filter2D(image, -1, kernel)

# --- Geometric Transformation Functions (using OpenCV's warpAffine) ---

def apply_translation(image: np.ndarray, tx: int, ty: int) -> np.ndarray:
    """Translates an image by (tx, ty)."""
    h, w = image.shape[:2]
    matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, matrix, (w, h))

def apply_rotation(image: np.ndarray, angle: float, scale: float = 1.0) -> np.ndarray:
    """Rotates an image around its center."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, matrix, (w, h))
    
def apply_affine_transform(image: np.ndarray) -> np.ndarray:
    """Applies a more complex affine transformation (shear + scale)."""
    h, w = image.shape[:2]
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    matrix = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(image, matrix, (w, h))

# --- Main Execution ---
if __name__ == "__main__":

    # --- Configuration ---
    IMAGE_PATH = "images/Lenna_test_image.png"
    OUTPUT_DIR = "output/filters_and_transforms"
    TARGET_SIZE = (400, 400)
    VISUALIZE = True # Set to False to only save files without showing a window

    # --- Script ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    img_color, img_gray = load_and_prepare_image(IMAGE_PATH, TARGET_SIZE)
    base_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]

    # 1. Define all operations to run
    # Grayscale operations
    filters_gray = {
        "sobel": apply_sobel_filter,
        "laplacian": apply_laplacian_filter,
    }
    # Color operations
    filters_color = {
        "box_filter": apply_box_filter,
        "gaussian": apply_gaussian_filter,
        "motion_blur": apply_motion_blur,
    }
    # Color transformations
    transformations = {
        "translation": lambda img: apply_translation(img, 50, 25),
        "rotation": lambda img: apply_rotation(img, 30),
        "affine": apply_affine_transform,
    }
    
    print("Applying filters and transformations...")
    
    # 2. Run all operations and save results
    for name, func in filters_gray.items():
        result = func(img_gray)
        save_image(result, OUTPUT_DIR, f"{base_name}_{name}.png")

    for name, func in filters_color.items():
        result = func(img_color)
        save_image(result, OUTPUT_DIR, f"{base_name}_{name}.png")
        
    for name, func in transformations.items():
        result = func(img_color)
        save_image(result, OUTPUT_DIR, f"{base_name}_{name}.png")
        
    print(f"All processed images have been saved to the '{OUTPUT_DIR}' directory.")

    # 3. (Optional) Visualize all results in a single window
    if VISUALIZE:
        # For simplicity, we'll just show the final affine transform
        final_result = apply_affine_transform(img_color)
        comparison = np.hstack([img_color, final_result])
        cv2.imshow("Original vs. Affine Transform", comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()