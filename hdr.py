import cv2
import numpy as np
import os
import sys

def load_images(image_paths: dict, size: tuple) -> dict:
    """Loads and resizes all images from the provided paths."""
    loaded_images = {}
    for name, path in image_paths.items():
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Could not load image '{name}' from '{path}'. Skipping.")
            continue
        loaded_images[name] = cv2.resize(img, size)
    
    if not loaded_images:
        sys.exit("Error: No images were successfully loaded.")
        
    return loaded_images

def create_hdr_mertens(images: list) -> np.ndarray:
    """Merges a list of exposures into a single HDR image using Mertens fusion."""
    merge_mertens = cv2.createMergeMertens()
    hdr_image = merge_mertens.process(images)
    
    # Convert from float [0, 1] to uint8 [0, 255] for saving and display
    return np.clip(hdr_image * 255, 0, 255).astype(np.uint8)

def create_comparison_view(image_dict: dict, size: tuple) -> np.ndarray:
    """Creates a 2x2 collage of images for comparison."""
    # Create a blank canvas for the 2x2 grid
    canvas = np.zeros((size[1] * 2, size[0] * 2, 3), dtype=np.uint8)
    
    # Define positions and add text labels to each image
    positions = {
        'underexposed': (0, 0),
        'normal': (size[0], 0),
        'overexposed': (0, size[1]),
        'hdr_result': (size[0], size[1])
    }
    
    labeled_images = {}
    for name, img in image_dict.items():
        labeled_img = img.copy()
        cv2.putText(labeled_img, name.replace('_', ' ').title(), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        labeled_images[name] = labeled_img

    # Place images onto the canvas
    for name, (x, y) in positions.items():
        if name in labeled_images:
            canvas[y:y + size[1], x:x + size[0]] = labeled_images[name]
            
    return canvas
    
# --- Main Execution ---
if __name__ == "__main__":

    # --- Configuration ---
    IMAGE_PATHS = {
        'underexposed': "images/ue.jpg",
        'normal': "images/ne.jpg",
        'overexposed': "images/oe.jpg"
    }
    OUTPUT_DIR = "output"
    VISUALIZE = True  # Set to False to disable the display window
    TARGET_SIZE = (400, 400)

    # --- Script ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load and prepare all input images
    images = load_images(IMAGE_PATHS, TARGET_SIZE)
    
    # 2. Create the HDR image
    hdr_result = create_hdr_mertens(list(images.values()))
    print("HDR image created successfully.")

    # 3. Save only the HDR result
    base_name = os.path.splitext(os.path.basename(IMAGE_PATHS['normal']))[0]
    output_filename = os.path.join(OUTPUT_DIR, f"{base_name}_hdr_result.png")
    cv2.imwrite(output_filename, hdr_result)
    print(f"Result saved to: {output_filename}")

    # 4. If visualizing, create and show the comparison view
    if VISUALIZE:
        images['hdr_result'] = hdr_result
        comparison_image = create_comparison_view(images, TARGET_SIZE)
        cv2.imshow("HDR Comparison", comparison_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()