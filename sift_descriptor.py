import cv2 as cv
import numpy as np
import scipy.ndimage as ndimage

def gaussian_kernel(size, sigma):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

def normalized_laplacian_of_gaussian(size, sigma):
    g_kernel = gaussian_kernel(size, sigma)
    log_kernel = ndimage.laplace(g_kernel)
    return sigma * sigma * log_kernel

def build_scale_space(image, sigmas, size):
    scale_space = []
    for sigma in sigmas:
        log_kernel = normalized_laplacian_of_gaussian(size, sigma)
        response = cv.filter2D(image, cv.CV_64F, log_kernel)
        scale_space.append(response)
    return np.array(scale_space)

def non_maximum_suppression(scale_space, threshold):
    max_filtered = ndimage.maximum_filter(scale_space, size=(3, 3, 3))
    blobs = (scale_space == max_filtered) & (scale_space > threshold)
    return blobs

def detect_blobs(image, sigmas, size, threshold):
    scale_space = build_scale_space(image, sigmas, size)
    blob_candidates = non_maximum_suppression(scale_space, threshold)
    blob_coords = np.argwhere(blob_candidates)
    
    blobs = []
    for sigma_index, y, x in blob_coords:
        sigma_val = sigmas[sigma_index]
        radius = sigma_val
        blobs.append((x, y, radius))
    
    return blobs

def draw_blobs(image, blobs):
    for x, y, radius in blobs:
        cv.circle(image, (x, y), int(radius), (0, 0, 255), 1)
    return image

# Read images
img1_path = r"D:\autonomous_driving_system\computer vision\images\room1.jpg"
img2_path = r"D:\autonomous_driving_system\computer vision\images\room2.jpg"

img1 = cv.imread(img1_path)
img2 = cv.imread(img2_path)

if img1 is None or img2 is None:
    print("Error: One or both image paths are incorrect.")
    exit()

# Convert to grayscale
img1_gs = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2_gs = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# Resize for consistency
img1_r = cv.resize(img1_gs, (500, 500))
img2_r = cv.resize(img2_gs, (500, 500))

# Blob Detection Parameters
min_sigma = 3
max_sigma = 50
num_sigma = 20
sigmas = np.linspace(min_sigma, max_sigma, num_sigma)

kernel_size = 9
threshold = 0.3 # Lowered threshold to detect more blobs

# Detect blobs in both images
blobs1 = detect_blobs(img1_r, sigmas, kernel_size, threshold)
blobs2 = detect_blobs(img2_r, sigmas, kernel_size, threshold)

# Convert blob coordinates to cv.KeyPoint objects
keypoints1 = [cv.KeyPoint(float(x), float(y), radius * 2) for (x, y, radius) in blobs1]
keypoints2 = [cv.KeyPoint(float(x), float(y), radius * 2) for (x, y, radius) in blobs2]

# Initialize SIFT detector
sift = cv.SIFT_create()

# Compute descriptors
if keypoints1 and keypoints2:
    keypoints1, descriptors1 = sift.compute(img1_r, keypoints1)
    keypoints2, descriptors2 = sift.compute(img2_r, keypoints2)
else:
    print("Error: No keypoints detected in one or both images.")
    exit()

if descriptors1 is None or descriptors2 is None:
    print("Error: No descriptors found.")
    exit()

# Initialize Brute-Force matcher
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

# Match descriptors
matches = bf.match(descriptors1, descriptors2)

# Sort matches by distance (best matches first)
matches = sorted(matches, key=lambda x: x.distance)

# Draw top N matches
N = min(20, len(matches))
N = len(matches)
matched_img = cv.drawMatches(img1_r, keypoints1, img2_r, keypoints2, matches[:N], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


# Display the matched image
cv.imshow('Matched Features', matched_img)
cv.waitKey(0)
cv.destroyAllWindows()


# storing matched points
pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Apply RANSAC to filter good matches
H, mask = cv.findHomography(pts1, pts2, cv.RANSAC, 5.0)

# Draw matches after RANSAC
matchesMask = mask.ravel().tolist()
draw_params = dict(matchColor=(0, 255, 0),  # Green matches are inliers
                   singlePointColor=None,
                   matchesMask=matchesMask,
                   flags=2)

matched_img = cv.drawMatches(img1_r, keypoints1, img2_r, keypoints2, matches, None, **draw_params)

# Show the result
cv.imshow("RANSAC Matches", matched_img)
cv.waitKey(0)
cv.destroyAllWindows()