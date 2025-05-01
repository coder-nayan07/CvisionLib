import cv2 as cv
import numpy as np
from skimage.filters import gaussian
from skimage.segmentation import active_contour

img = cv.imread(r"images/coin.png")

size =(400,400)
image = cv.resize(img,size)
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow("coin",image)
cv.waitKey()
cv.destroyAllWindows()

sigma = 2
smoothed = gaussian(image, sigma=sigma)

# Compute gradient magnitude
grad_x = cv.Sobel(smoothed, cv.CV_64F, 1, 0, ksize=7)
grad_y = cv.Sobel(smoothed, cv.CV_64F, 0, 1, ksize=7)
gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

# Define rectangle parameters
x_c, y_c = 100, 100  # Center of the rectangle
w, h = 500, 500       
num_points_per_side = 50

# Generate rectangle points
top = np.linspace([x_c - w//2, y_c - h//2], [x_c + w//2, y_c - h//2], num_points_per_side)
right = np.linspace([x_c + w//2, y_c - h//2], [x_c + w//2, y_c + h//2], num_points_per_side)
bottom = np.linspace([x_c + w//2, y_c + h//2], [x_c - w//2, y_c + h//2], num_points_per_side)
left = np.linspace([x_c - w//2, y_c + h//2], [x_c - w//2, y_c - h//2], num_points_per_side)

# Combine all sides into a single contour array
rect_contour = np.vstack([top, right, bottom, left])

# Perform Active Contour Model optimization
snake = active_contour(gradient_magnitude, rect_contour, alpha=0.1, beta=0.3, gamma=0.1)

# Draw final contour on the image
contour_image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
for i in range(len(snake) - 1):
    cv.line(contour_image, (int(snake[i][1]), int(snake[i][0])), 
                             (int(snake[i+1][1]), int(snake[i+1][0])), 
                             (0, 255, 0), 2)

# Show results
cv.imshow("Active Contour", contour_image)
cv.waitKey(0)
cv.destroyAllWindows()
