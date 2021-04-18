import cv2 as cv
import numpy as np

path = r"C:\Users\Administrator\Pictures\Saved Pictures\01.jpg"
img = cv.imread(path)
img = cv.resize(img, (256, 256))
img1 = cv.flip(img, 1)
img2 = cv.flip(img, 0)
img3 = cv.flip(img, -1)
# cv.imshow("picture", img1)
imgs = [img, img1, img2, img3]

imgs = np.hstack(imgs)
cv.imshow("sadasd", imgs)
cv.waitKey(0)