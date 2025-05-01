import cv2
import numpy as np

# Load the full image
image = cv2.imread(r"D:\autonomous_driving_system\computer vision\images\template_matching.jpg")

# Define the region to crop as the template (adjust coordinates as needed)
x, y, w, h = 400, 200, 100, 100  # Example coordinates
template = image[y:y+h, x:x+w]  # Crop a part of the image

# Save the template
cv2.imwrite("template.png", template)

def sliding_window_matching(image, template):

    img_h, img_w,_ = image.shape
    temp_h, temp_w,_ = template.shape
    
    min_value = float('inf')
    best_x, best_y = -1, -1
    
    # Sliding window approach
    for y in range(img_h - temp_h + 1):
        for x in range(img_w - temp_w + 1):
            # Extract region of interest (ROI)
            roi = image[y:y+temp_h, x:x+temp_w]
            
            # Compute Sum of Squared Differences (SSD)
            ssd = np.sum((roi - template) ** 2)
            
            # Update minimum value position
            if ssd < min_value:
                min_value = ssd
                best_x, best_y = x, y
    
    return best_x, best_y, min_value


def draw_bounding_box(image, best_x, best_y, temp_w, temp_h):
    
    cv2.rectangle(image, (best_x, best_y), (best_x + temp_w, best_y + temp_h), (0, 255, 0), 2)
    
    # Show the image
    cv2.imshow("Detected Match", image)
    cv2.imshow("Template", template)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

bst_x, bst_y, min_v = sliding_window_matching(image,template)
print(bst_x)
print(bst_y)
print(min_v)
draw_bounding_box(image,bst_x,bst_y,w,h)