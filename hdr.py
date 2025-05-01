import numpy as np
import cv2 as cv

# Load images
ue = cv.imread(r"D:\autonomous_driving_system\computer vision\images\ue.jpg")
ne = cv.imread(r"D:\autonomous_driving_system\computer vision\images\ne.jpg")
oe = cv.imread(r"D:\autonomous_driving_system\computer vision\images\oe.jpg")

# Ensure images are loaded
if ue is None or ne is None or oe is None:
    print("Error: One or more images not found.")
    exit()

# Resize images to the same dimensions
size = (400, 400)
uer = cv.resize(ue, size)
ner = cv.resize(ne, size)
oer = cv.resize(oe, size)

# Display individual images
cv.imshow('Underexposed', uer)
cv.imshow('Normal Exposure', ner)
cv.imshow('Overexposed', oer)

# Convert images to float32 for HDR merging
def merge_hdr_mertens(images):
    """
    Merges multiple exposure images using Mertens fusion and returns the HDR result.
    
    :param images: List of images (should be numpy arrays of the same size)
    :return: HDR image after Mertens exposure fusion (8-bit)
    """
    # Convert images to float32 for processing
    images_float32 = [img.astype(np.float32) for img in images]

    # Merge images using Mertens fusion
    merge_mertens = cv.createMergeMertens()
    hdr_mertens = merge_mertens.process(images_float32)

    # Convert back to 8-bit for display
    hdr_mertens = (hdr_mertens * 255).astype(np.uint8)
    
    return hdr_mertens

hdr_result = merge_hdr_mertens([ner, oer])

cv.imshow("hdr",hdr_result)
cv.waitKey(0)
cv.destroyAllWindows()