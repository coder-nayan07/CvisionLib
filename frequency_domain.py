import cv2 as cv
import numpy as np

img = cv.imread(r"D:\autonomous_driving_system\computer vision\images\Lenna_test_image.png")

size =(50,50)
img_r = cv.resize(img,size)
# if len(img.shape) == 3:
img_gs = cv.cvtColor(img_r, cv.COLOR_BGR2GRAY)

cv.imshow("lenna",img_gs)
cv.waitKey()
cv.destroyAllWindows()
    
def dft2d(image):
    M, N= image.shape  
    F = np.zeros((M, N), dtype=complex)  
    
    for u in range(M):  
        for v in range(N): 
            sum_value = 0 
            for x in range(M): 
                for y in range(N): 
                    exponent = -2j * np.pi * ((u * x / M) + (v * y / N))
                    sum_value += image[x, y] * np.exp(exponent)
            F[u, v] = sum_value 
            
    return F

img_ft = dft2d(img_gs)

magnitude_spectrum = np.abs(img_ft)
magnitude_spectrum = np.log(1 + magnitude_spectrum)  # Log scaling for better visibility
magnitude_spectrum = cv.normalize(magnitude_spectrum, None, 0, 255, cv.NORM_MINMAX)
magnitude_spectrum = magnitude_spectrum.astype(np.uint8)

cv.imshow("Magnitude Spectrum", magnitude_spectrum)
cv.waitKey(0)
cv.destroyAllWindows()

