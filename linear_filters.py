import cv2 as cv
import numpy as np
sh = cv.imshow
host_img = cv.imread(r"D:\autonomous_driving_system\computer vision\images\Lenna_test_image.png")

size = (400, 400)
host_imgr = cv.resize(host_img, size)

# linear filters

# box filter 
def box_filter (img):
    # A simple smoothing filter that replaces each pixel with the average of its neighboring pixels.
    kernel = np.ones((5, 5), dtype=np.float32) / 25
    filtered_img = cv.filter2D(img, -1, kernel)
    print(filtered_img.shape)
    cv.imshow("Original", img)
    sh("lena_box_filter",filtered_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

#  gaussian filter
def gaussian_filter(image):
    gaussian_kernel = np.array([[1,  4,  6,  4,  1],
                            [4, 16, 24, 16, 4],
                            [6, 24, 36, 24, 6],
                            [4, 16, 24, 16, 4],
                            [1,  4,  6,  4,  1]], dtype=np.float32)

    # Normalize the kernel
    gaussian_kernel /= gaussian_kernel.sum()

    # Apply the filter using cv2.filter2D
    filtered_image = cv.filter2D(image, -1, gaussian_kernel)

    # Show the images
    cv.imshow("Original", image)
    cv.imshow("Gaussian Blur (Custom)", filtered_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

# sobel filter edge detection

def sobel_filter_ed(image):
    # Define Sobel kernels
    sobel_x_kernel = np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=np.float32)

    sobel_y_kernel = np.array([[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]], dtype=np.float32)

    # Apply filters
    sobel_x = cv.filter2D(image, -1, sobel_x_kernel)
    sobel_y = cv.filter2D(image, -1, sobel_y_kernel)

    # Compute gradient magnitude
    sobel_combined = cv.magnitude(sobel_x.astype(np.float32), sobel_y.astype(np.float32))

    # Convert to uint8
    sobel_combined = np.uint8(sobel_combined)
    print(sobel_combined.shape)
    # Display results
    cv.imshow("Sobel X", sobel_x)
    cv.imshow("Sobel Y", sobel_y)
    cv.imshow("Sobel Combined", sobel_combined)
    cv.waitKey(0)
    cv.destroyAllWindows()

# edge detection using simple gradient
def edge_detection_using_gradients(image):
    # Define Sobel kernels
    grad_x_kernel = np.array([[-1, 1],
                                [-1, 1]], dtype=np.float32)

    grad_y_kernel = np.array([[-1, -1],
                                [1,  1]], dtype=np.float32)
 
    # Apply filters
    grad_x = cv.filter2D(image, -1, grad_x_kernel)
    grad_y = cv.filter2D(image, -1, grad_y_kernel)

    # Compute gradient magnitude
    sobel_combined = cv.magnitude(grad_x.astype(np.float32), grad_y.astype(np.float32))

    # Convert to uint8
    sobel_combined = np.uint8(sobel_combined)

    # Display results
    cv.imshow("grad X", grad_x)
    cv.imshow("grad Y", grad_y)
    cv.imshow("grad Combined", sobel_combined)
    cv.waitKey(0)
    cv.destroyAllWindows()


def laplacian_edge(image):
    laplacian_kernel = np.array([[ 0, -1,  0],
                                [-1,  4, -1],
                                [ 0, -1,  0]], dtype=np.float32)

    # Apply the custom Laplacian kernel
    laplacian_custom = cv.filter2D(image, -1, laplacian_kernel)

    # Convert to absolute values
    laplacian_custom = np.uint8(np.abs(laplacian_custom))

    # Display results
    cv.imshow("Custom Laplacian", laplacian_custom)
    cv.waitKey(0)
    cv.destroyAllWindows()

def motion_blur_filter(image):
    kernel_size = 15  # Adjust for more/less blur
    motion_blur_kernel = np.zeros((kernel_size, kernel_size))
    motion_blur_kernel[:, int((kernel_size - 1)/2)] = 1  # Fill the middle row
    print(motion_blur_kernel)
    motion_blur_kernel /= kernel_size  # Normalize the kernel

    # Apply the motion blur using convolution
    blurred_image = cv.filter2D(image, -1, motion_blur_kernel)

    # Show results
    cv.imshow("Original", image)
    cv.imshow("Motion Blur", blurred_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
motion_blur_filter(host_imgr)

