import cv2
import numpy as np
import os
import sys
import scipy.ndimage

def detect_and_match_features(img1: np.ndarray, img2: np.ndarray, lowe_ratio: float = 0.75) -> tuple:
    """
    Detects features, computes descriptors, and finds good matches using the ratio test.
    """
    # 1. Use SIFT to detect and compute features in both images
    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)

    if desc1 is None or desc2 is None:
        return [], [], [], []

    # 2. Use BFMatcher with knnMatch to find the 2 best matches for each descriptor
    bf = cv2.BFMatcher(cv2.NORM_L2)
    all_matches = bf.knnMatch(desc1, desc2, k=2)

    # 3. Apply Lowe's ratio test to find good matches
    good_matches = []
    for m, n in all_matches:
        if m.distance < lowe_ratio * n.distance:
            good_matches.append(m)
            
    return kp1, kp2, desc1, desc2, good_matches

def find_homography(matches: list, kp1: list, kp2: list, ransac_thresh: float) -> tuple:
    """Finds the homography matrix using RANSAC."""
    if len(matches) < 4:
        return None, None
        
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_thresh)
    return H, mask

def create_stitched_image(img1: np.ndarray, img2: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Warps and stitches two images together using a homography matrix."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    
    corners1_transformed = cv2.perspectiveTransform(corners1, H)
    all_corners = np.vstack((corners1_transformed, corners2))

    x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel())
    
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    output_size = (x_max - x_min, y_max - y_min)
    stitched_img = cv2.warpPerspective(img1, H_translation @ H, output_size)
    stitched_img[translation_dist[1]:h2 + translation_dist[1], translation_dist[0]:w2 + translation_dist[0]] = img2
    
    return stitched_img

# --- Visualization ---

def visualize_matches(img1, kp1, img2, kp2, matches, mask, title):
    """Draws and displays feature matches."""
    matchesMask = None
    if mask is not None:
        matchesMask = mask.ravel().tolist()
    
    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)
    cv2.imshow(title, match_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
# --- Main Execution ---
if __name__ == "__main__":

    # --- Configuration ---
    IMAGE_PATHS = {
        'img1': "images/image_l.jpg",
        'img2': "images/image_r.jpg"
    }
    OUTPUT_DIR = "output"
    VISUALIZE_STEPS = True  # Set to False to disable intermediate visualizations
    TARGET_SIZE = (800, 600)
    
    # Matching parameters
    LOWE_RATIO = 0.75
    RANSAC_THRESH = 4.0

    # --- Script ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load and preprocess images
    img1_color = cv2.imread(IMAGE_PATHS['img1'])
    img2_color = cv2.imread(IMAGE_PATHS['img2'])
    if img1_color is None or img2_color is None:
        sys.exit("Error: Could not load one or both images.")
        
    img1 = cv2.resize(cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY), TARGET_SIZE)
    img2 = cv2.resize(cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY), TARGET_SIZE)
    img1_color_resized = cv2.resize(img1_color, TARGET_SIZE)
    img2_color_resized = cv2.resize(img2_color, TARGET_SIZE)

    # 2. Detect features and find good matches
    kp1, kp2, desc1, desc2, good_matches = detect_and_match_features(img1, img2, LOWE_RATIO)
    print(f"Detected {len(kp1)} features in image 1 and {len(kp2)} features in image 2.")
    print(f"Found {len(good_matches)} good matches after ratio test.")
    
    # 3. Find homography with RANSAC
    H, mask = find_homography(good_matches, kp1, kp2, RANSAC_THRESH)
    if H is None:
        sys.exit("Error: Could not find a valid homography. Images may not overlap enough.")
    print(f"Found {np.sum(mask)} inlier matches after RANSAC.")
    
    # 4. Visualize RANSAC-filtered matches if enabled
    if VISUALIZE_STEPS:
        visualize_matches(img1_color_resized, kp1, img2_color_resized, kp2, good_matches, mask, "RANSAC Filtered Matches")

    # 5. Stitch images using the original full-resolution images for best quality
    # Re-calculate homography on original points for higher accuracy
    full_kp1, full_kp2, _, _, full_good_matches = detect_and_match_features(
        cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY), 
        cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY), 
        LOWE_RATIO
    )
    H_full, _ = find_homography(full_good_matches, full_kp1, full_kp2, RANSAC_THRESH)
    
    if H_full is not None:
        stitched_result = create_stitched_image(img1_color, img2_color, H_full)
    else:
        print("Stitching with resized images due to failure on full-res.")
        stitched_result = create_stitched_image(img1_color_resized, img2_color_resized, H)


    # 6. Save and display the final result
    output_filename = os.path.join(OUTPUT_DIR, "stitched_image_improved.png")
    cv2.imwrite(output_filename, stitched_result)
    print(f"Stitched image saved to: {output_filename}")
    
    cv2.imshow("Stitched Image", stitched_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()