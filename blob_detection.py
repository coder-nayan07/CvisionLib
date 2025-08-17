import cv2
import numpy as np
import scipy.ndimage as ndimage
import sys
import os

# --- Core Blob Detection Functions ---

def create_log_kernel(size: int, sigma: float) -> np.ndarray:
    """Creates a scale-normalized Laplacian of Gaussian (LoG) kernel."""
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    
    # Gaussian part
    g = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    g /= np.sum(g)
    
    # Laplacian of Gaussian part
    log = ndimage.laplace(g)
    
    # Scale normalization
    return sigma * sigma * log

def build_scale_space(image: np.ndarray, sigmas: np.ndarray, kernel_size: int) -> np.ndarray:
    """Builds a scale space by filtering the image with LoG kernels of varying sigmas."""
    scale_space = []
    for sigma in sigmas:
        log_kernel = create_log_kernel(kernel_size, sigma)
        response = cv2.filter2D(image, cv2.CV_64F, log_kernel)
        scale_space.append(response)
    return np.array(scale_space)

def find_local_maxima(scale_space: np.ndarray, threshold: float) -> np.ndarray:
    """Finds local maxima in the 3D scale-space."""
    # Use absolute values for maxima detection to find both dark and bright blobs
    abs_scale_space = np.abs(scale_space)
    max_filtered = ndimage.maximum_filter(abs_scale_space, size=(3, 3, 3))
    
    # A point is a local maximum if it's equal to the maximum in its 3x3x3 neighborhood
    is_max = (abs_scale_space == max_filtered)
    is_above_threshold = (abs_scale_space > threshold)
    
    return np.argwhere(is_max & is_above_threshold)

def detect_blobs(image: np.ndarray, min_sigma: float, max_sigma: float, num_sigma: int, threshold: float, kernel_size: int) -> list:
    """
    Detects blobs in an image using the Laplacian of Gaussian method.

    Returns:
        A list of tuples, where each tuple is (x, y, radius) for a detected blob.
    """
    # Ensure image is grayscale for single-channel processing
    if len(image.shape) == 3:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = image
        
    gray_img = gray_img.astype(np.float64) / 255.0 # Normalize for consistent thresholding

    # 1. Define the range of scales (sigmas)
    sigmas = np.linspace(min_sigma, max_sigma, num_sigma)

    # 2. Build the scale-space representation
    scale_space = build_scale_space(gray_img, sigmas, kernel_size)

    # 3. Find local maxima in the scale-space
    blob_coords = find_local_maxima(scale_space, threshold)

    # 4. Extract blob details (coordinates and radius)
    blobs = []
    for sigma_idx, y, x in blob_coords:
        sigma_val = sigmas[sigma_idx]
        # Radius is proportional to sigma, sqrt(2) is a common factor for LoG
        radius = sigma_val * np.sqrt(2)
        blobs.append((x, y, radius))
        
    return blobs

# --- Visualization Function ---

def visualize_blobs(image: np.ndarray, blobs: list, window_name="Detected Blobs"):
    """Draws circles on the image for each detected blob and displays it."""
    output_image = image.copy()
    for x, y, radius in blobs:
        cv2.circle(output_image, (x, y), int(radius), (0, 0, 255), 2)
    
    save_image(output_image, "output", "blob_detected_lenna.png")
    cv2.imshow(window_name, output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_image(img, directory, filename):

    if not os.path.exists(directory):
        os.makedirs(directory)

    save_path = os.path.join(directory, filename)
    cv2.imwrite(save_path, img)
    print(f"Image saved to: {save_path}")



# --- Main Execution ---
if __name__ == "__main__":
    
    # --- Configuration ---
    IMAGE_PATH = "images\Lenna_test_image.png"
    VISUALIZE = True  # Set to False to run without displaying the result image

    # Detection Parameters
    MIN_SIGMA = 2
    MAX_SIGMA = 15
    NUM_SIGMA = 12
    THRESHOLD = 0.2 # Threshold for the normalized LoG response
    KERNEL_SIZE = 15 # Should be odd and large enough for the max_sigma

    # --- Script ---
    host_img = cv2.imread(IMAGE_PATH)
    if host_img is None:
        sys.exit(f"Error: Could not read the image from '{IMAGE_PATH}'")

    # 1. Detect the blobs by calling the main detection function
    blobs = detect_blobs(
        image=host_img,
        min_sigma=MIN_SIGMA,
        max_sigma=MAX_SIGMA,
        num_sigma=NUM_SIGMA,
        threshold=THRESHOLD,
        kernel_size=KERNEL_SIZE
    )
    print(f"Detected {len(blobs)} blobs.")

    # 2. Visualize the results if the flag is enabled
    if VISUALIZE:
        visualize_blobs(host_img, blobs)


