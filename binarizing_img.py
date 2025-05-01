import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def plot_grayscale_histogram(img):
    """
    Takes an uploaded image, converts it to grayscale if needed, and plots its histogram.
    
    Parameters:
    img (numpy.ndarray): Input image.
    """
 
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
 
    img = img.astype(np.float32) / 255.0
    
    # Compute histogram
    histogram, bin_edges = np.histogram(img, bins=256, range=(0.0, 1.0))
    
    # Plot histogram
    fig, ax = plt.subplots()
    ax.plot(bin_edges[:-1], histogram, color='black')
    ax.set_title("Grayscale Histogram")
    ax.set_xlabel("Grayscale Value")
    ax.set_ylabel("Pixels")
    ax.set_xlim(0, 1.0)
    plt.show()

# load images

shape1 = cv.imread(r"D:\autonomous_driving_system\computer vision\images\for_binarization.jpg")
shape2 = cv.imread(r"D:\autonomous_driving_system\computer vision\images\bin2.jpg")
# resizing
size = (400, 400)
shape1r = cv.resize(shape1, size)
shape2r = cv.resize(shape2, size)

cv.imshow("shape1", shape1r)
cv.imshow("shape2", shape2r)

# histogram plotting
plot_grayscale_histogram(shape1r)
plot_grayscale_histogram(shape2r)


# convert to grayscale
shape1gs = cv.cvtColor(shape1r, cv.COLOR_BGR2GRAY)
shape2gs = cv.cvtColor(shape2r, cv.COLOR_BGR2GRAY)
# apply thresholding
threshold = 220
shape1_bin = np.where(shape1gs > threshold, 255, 0).astype(np.uint8)
# cv.imshow("shape1bin", shape1_bin)
cv.imshow("shape1bin", shape1gs)

threshold = 100
shape2_bin = np.where(shape2gs > threshold, 255, 0).astype(np.uint8)
# cv.imshow("shape2bin", shape2_bin)
cv.imshow("shape2bin", shape2gs)

save_path = (r"D:\autonomous_driving_system\computer vision\images\shape1_binarized.jpg")
cv.imwrite(save_path, shape1_bin)

# otsu thresholding

cv.waitKey()
cv.destroyAllWindows()