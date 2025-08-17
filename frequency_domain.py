import cv2
import numpy as np
import os
import sys
import time

def load_and_preprocess_image(path: str, size: tuple) -> np.ndarray:
    """Loads, resizes, and converts an image to grayscale."""
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        sys.exit(f"Error: Could not read image from '{path}'")
    
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

def compute_dft_manual(image: np.ndarray) -> np.ndarray:
    """Computes the 2D Discrete Fourier Transform using nested loops."""
    M, N = image.shape
    F = np.zeros((M, N), dtype=np.complex128)
    
    for u in range(M):
        for v in range(N):
            sum_val = 0.0
            for x in range(M):
                for y in range(N):
                    exponent = -2j * np.pi * ((u * x / M) + (v * y / N))
                    sum_val += image[x, y] * np.exp(exponent)
            F[u, v] = sum_val
            
    return F

def create_and_prepare_spectrum(dft_result: np.ndarray) -> np.ndarray:
    """Creates a viewable magnitude spectrum from a DFT result."""
    # Shift the zero-frequency component to the center
    shifted_dft = np.fft.fftshift(dft_result)
    
    # Calculate magnitude and apply log scale for better dynamic range
    magnitude_spectrum = np.log1p(np.abs(shifted_dft))
    
    # Normalize to 0-255 for display
    cv2.normalize(magnitude_spectrum, magnitude_spectrum, 0, 255, cv2.NORM_MINMAX)
    return magnitude_spectrum.astype(np.uint8)

def visualize_and_save(image: np.ndarray, output_path: str, show_window: bool):
    """Saves the image and optionally displays it in a window."""
    cv2.imwrite(output_path, image)
    print(f"Result saved to: {output_path}")

    if show_window:
        cv2.imshow("Magnitude Spectrum", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# --- Main Execution ---
if __name__ == "__main__":

    # --- Configuration ---
    IMAGE_PATH = "images/Lenna_test_image.png"
    OUTPUT_DIR = "output"
    VISUALIZE = True  # Set to False to disable the display window
    RESIZE_DIM = (32, 32) # Keep dimensions small for the manual DFT

    # --- Script ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load and prepare the image
    gray_image = load_and_preprocess_image(IMAGE_PATH, RESIZE_DIM)

    # 2. Compute the DFT
    print(f"Computing {RESIZE_DIM[0]}x{RESIZE_DIM[1]} DFT. This will take some time...")
    start_time = time.time()
    dft_complex = compute_dft_manual(gray_image)
    end_time = time.time()
    print(f"DFT computation finished in {end_time - start_time:.2f} seconds.")

    # 3. Create the spectrum for visualization
    spectrum = create_and_prepare_spectrum(dft_complex)

    # 4. Save and visualize the result
    base_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    output_filename = os.path.join(OUTPUT_DIR, f"{base_name}_spectrum.png")

    if VISUALIZE:
        cv2.imshow("host_image" ,gray_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    visualize_and_save(spectrum, output_filename, show_window=VISUALIZE)
