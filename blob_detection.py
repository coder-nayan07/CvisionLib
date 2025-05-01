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
        # cv.imshow("responses", response)
        # cv.waitKey()
        # cv.destroyAllWindows()
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
        radius =  sigma_val
        blobs.append((x, y, radius))
    return blobs

def draw_blobs(image, blobs):
    for x, y, radius in blobs:
        cv.circle(image, (x, y), int(radius), (0, 0, 255), 1)
    return image

# Main
host_img = cv.imread("images\coin.png")
gray_img = cv.cvtColor(host_img, cv.COLOR_BGR2GRAY)

# Smaller range of sigmas and fewer steps
min_sigma = 3
max_sigma = 7
num_sigma = 10
sigmas = np.linspace(min_sigma, max_sigma, num_sigma)

kernel_size = 9
threshold = 10.0  # Start with a big threshold to avoid flooding

blobs = detect_blobs(gray_img, sigmas, kernel_size, threshold)
print(f"Detected {len(blobs)} blobs.")

result_img = draw_blobs(host_img.copy(), blobs)
cv.imshow("Detected Blobs", result_img)
cv.waitKey(0)
cv.destroyAllWindows()
