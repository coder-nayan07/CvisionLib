import cv2 as cv

img_l = cv.imread(r"D:\codes\autonomous_driving_system\computer vision\images\stereo_img_left.png")
img_r = cv.imread(r"D:\codes\autonomous_driving_system\computer vision\images\stereo_img_right.png")

img_l_gs = cv.cvtColor(img_l, cv.COLOR_BGR2GRAY)
img_r_gs = cv.cvtColor(img_r, cv.COLOR_BGR2GRAY)

stereo = cv.StereoBM_create(numDisparities=64, blockSize=15)

disparity = stereo.compute(img_l_gs, img_r_gs)


disparity_normalized = cv.normalize(disparity, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
disparity_normalized = cv.convertScaleAbs(disparity_normalized)
cv.imshow("left", img_l)
cv.imshow("right", img_r)
cv.imshow('Disparity Map', disparity_normalized)
cv.waitKey(0)
cv.destroyAllWindows()
