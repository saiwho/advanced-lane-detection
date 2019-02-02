import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt


# def fitBySlidingWindows(binPerspectiveImg, lineL, lineR,  ):

def binPerspectiveTransform(img):

    imgHeight,imgWidth = img.shape[0:2]

    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    binImg = np.zeros(shape = (imgHeight,imgWidth), dtype ="uint8")

    #These are for colour based thresholding
    yellowLow = np.array([20, 70, 180], dtype="uint8")
    yellowHigh = np.array([40,200,200], dtype="uint8")
    whiteLow = np.array([0, 0, 185], dtype="uint8")
    whiteHigh = np.array([40, 10, 235], dtype="uint8")

    yellowMask = cv2.inRange(hsvImg, yellowLow, yellowHigh)
    binImg = cv2.bitwise_or(binImg,yellowMask)
    whiteMask = cv2.inRange(hsvImg, whiteLow, whiteHigh)
    binImg = cv2.bitwise_or(binImg,whiteMask)

    #Now for detecting the edges and Thresholding
    sobelX = cv2.Sobel(binImg, cv2.CV_64F, 1, 0, ksize = 5)
    sobelY = cv2.Sobel(binImg, cv2.CV_64F, 0, 1, ksize = 5)
    sobelMag = np.sqrt(sobelX**2 + sobelY**2)
    sobelMag = np.uint8(sobelMag / np.max(sobelMag) * 255)
    _, sobelMag = cv2.threshold(sobelMag, 50, 255, cv2.THRESH_BINARY)
    binImg = cv2.bitwise_or(sobelMag, binImg)

    #To fill small gaps, light morphology is applied
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(binImg, cv2.MORPH_CLOSE, kernel)

    #Perspective Transformation
    srcImg = np.float32([[0, 410], [570, 410], [490, 340], [150, 340]])
    desImg = np.float32([[0, imgHeight], [imgWidth, imgHeight], [imgWidth, 0], [0, 0]])
    M = cv2.getPerspectiveTransform(srcImg, desImg)
    Minv = cv2.getPerspectiveTransform(desImg, srcImg)
    transformedImg = cv2.warpPerspective(closing, M, (640, 480), flags=cv2.INTER_LINEAR)

    histogram = np.sum(transformedImg[transformedImg.shape[0] // 2:, :], axis=0)
    plt.plot(histogram)
    plt.show()
    exit()

    return transformedImg


if __name__=='__main__':
    imgCollection = []
    # for img in glob.glob("./images/screen*.png"):
    #     n = cv2.imread(img)
    #     imgCollection.append(n)
    # for img in imgCollection:
    #     gotImg = perspectiveTransform(img)
    #     cv2.imshow('img', gotImg)
    #     cv2.waitKey(500)

    img = cv2.imread('./images/screen17.png')
    binPerspectiveImg = binPerspectiveTransform(img)

