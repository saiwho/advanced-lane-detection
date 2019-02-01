import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('screen.png')

imgWidth  = img.shape[1]
imgHeight  = img.shape[0]

# srcImg = np.float32([[25, 225], [630, 220], [500, 125], [150, 125]])
# desImg = np.float32([[0, 470], [640, 470], [640, 0], [0, 0]])
#
# M = cv2.getPerspectiveTransform(srcImg, desImg)
# transformedImg = cv2.warpPerspective(img,M,(640,470))

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()