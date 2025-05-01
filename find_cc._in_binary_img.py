import cv2 as cv
import numpy as np

img = cv.imread(r"D:\autonomous_driving_system\computer vision\images\shape1_binarized.jpg")

# find connected components

def find_connected_components(binary_image):

    height, width = binary_image.shape


    labeled_image = np.zeros((height, width), dtype=np.int32)


    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]


    def dfs(x, y, label):
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if labeled_image[cx, cy] == 0:  
                labeled_image[cx, cy] = label
                for dx, dy in directions:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < height and 0 <= ny < width and binary_image[nx, ny] == 255 and labeled_image[nx, ny] == 0:
                        stack.append((nx, ny))


    label = 0
    for i in range(height):
        for j in range(width):
            if binary_image[i, j] == 255 and labeled_image[i, j] == 0:  # Unvisited white pixel
                label += 1
                dfs(i, j, label)

    return labeled_image, label


labeled_image, num_components = find_connected_components(img)

print(f"Number of connected components: {num_components}")
cv.imshow("binary", img)

cv.waitKey(0)
cv.destroyAllWindows()