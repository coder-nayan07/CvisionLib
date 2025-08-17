import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import sys

def load_image(path):
    """Loads an image from a relative path and handles errors."""
    img = cv.imread(path)
    if img is None:
        print(f"Error: Could not load image from path: {path}")
        print("Please ensure the file exists in the correct subfolder.")
        sys.exit() 
    return img

def plot_grayscale_histogram(img, title="Grayscale Histogram"):
    """Plots the histogram of a grayscale image."""
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    histogram = cv.calcHist([img], [0], None, [256], [0, 256])
    
    plt.figure()
    plt.title(title)
    plt.xlabel("Grayscale Value (0-255)")
    plt.ylabel("Pixel Count")
    plt.plot(histogram, color='black')
    plt.xlim([0, 256])
    plt.show()

def binarize_image(img_gs, threshold_value):
    """Applies a fixed binary threshold to a grayscale image."""
    _, binarized_img = cv.threshold(img_gs, threshold_value, 255, cv.THRESH_BINARY)
    return binarized_img

def save_image(img, directory, filename):
    """Saves an image to a specified directory, creating it if needed."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    save_path = os.path.join(directory, filename)
    cv.imwrite(save_path, img)
    print(f"Image saved to: {save_path}")

def main(vis=False, size=None):
    """Main function to run the image processing workflow."""
    # Load images using relative paths
    shape1 = load_image("images/for_binarization.jpg")
    shape2 = load_image("images/bin2.jpg")

    if size != None:
        shape1 = cv.resize(shape1, size)
        shape2 = cv.resize(shape2, size)


    # Display originals and their histograms
    if vis:
        cv.imshow("Original Shape 1", shape1)
        cv.imshow("Original Shape 2", shape2)
        plot_grayscale_histogram(shape1, "Histogram of Shape 1")
        plot_grayscale_histogram(shape2, "Histogram of Shape 2")

    # Convert to grayscale
    shape1_gs = cv.cvtColor(shape1, cv.COLOR_BGR2GRAY)
    shape2_gs = cv.cvtColor(shape2, cv.COLOR_BGR2GRAY)

    # Apply binarization with different thresholds
    shape1_bin = binarize_image(shape1_gs, threshold_value=220)
    shape2_bin = binarize_image(shape2_gs, threshold_value=100)

    # Display binarized results
    if vis:
        cv.imshow("Binarized Shape 1", shape1_bin)
        cv.imshow("Binarized Shape 2", shape2_bin)

    # Save one of the processed images
    save_image(shape1_bin, "output", "shape1_binarized.jpg")

    # Wait for user input to close windows
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    # pass the flag to see the visualisations
    vis = True
    size = (400, 400)
    main(vis, size)