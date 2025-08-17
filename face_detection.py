import cv2
import numpy as np
import os
import sys

def load_classifier(cascade_file: str) -> cv2.CascadeClassifier:
    """Loads a Haar Cascade classifier from the given file."""
    cascade_path = os.path.join(cv2.data.haarcascades, cascade_file)
    classifier = cv2.CascadeClassifier(cascade_path)
    if classifier.empty():
        sys.exit(f"Error: Could not load classifier from '{cascade_path}'")
    return classifier

def detect_faces(image: np.ndarray, classifier: cv2.CascadeClassifier, **params) -> list:
    """Detects faces in an image using the provided classifier."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(
        gray_image,
        scaleFactor=params.get('scaleFactor', 1.1),
        minNeighbors=params.get('minNeighbors', 5),
        minSize=params.get('minSize', (30, 30))
    )
    return faces

def draw_and_save_output(
    image: np.ndarray,
    faces: list,
    output_path: str,
    show_window: bool
):
    """Draws rectangles on faces, saves the image, and optionally displays it."""
    output_image = image.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imwrite(output_path, output_image)
    print(f"Result saved to: {output_path}")

    if show_window:
        cv2.imshow("Face Detection Result", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# --- Main Execution ---
if __name__ == "__main__":

    # --- Configuration ---
    IMAGE_PATH = "images\Lenna_test_image.png"
    OUTPUT_DIR = "output"
    CASCADE_XML = 'haarcascade_frontalface_default.xml'
    VISUALIZE = True  # Set to False to disable the display window

    # Detection parameters
    DETECT_PARAMS = {
        'scaleFactor': 1.1,
        'minNeighbors': 5,
        'minSize': (30, 30)
    }

    # --- Script ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load the classifier and the image
    face_classifier = load_classifier(CASCADE_XML)
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        sys.exit(f"Error: Could not read image from '{IMAGE_PATH}'")

    # 2. Detect faces
    detected_faces = detect_faces(image, face_classifier, **DETECT_PARAMS)
    print(f"Found {len(detected_faces)} face(s).")

    # 3. Draw results, save the output, and visualize
    base_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    output_filename = os.path.join(OUTPUT_DIR, f"{base_name}_faces_detected.png")

    draw_and_save_output(image, detected_faces, output_filename, show_window=VISUALIZE)