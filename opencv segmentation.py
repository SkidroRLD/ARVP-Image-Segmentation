import cv2 as cv
from math import atan2, cos, sin, sqrt, pi
import numpy as np


def drawAxis(img, p_, q_, color, scale):
    p = list(p_)
    q = list(q_)

    angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) +
                      (p[0] - q[0]) * (p[0] - q[0]))

    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv.line(img, (int(p[0]), int(p[1])),
            (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)

    # creating the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv.line(img, (int(p[0]), int(p[1])),
            (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)

    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv.line(img, (int(p[0]), int(p[1])),
            (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)


def getOrientation(pts, img): # Using PCA analysis for orientation analysis
    
    size = len(pts)
    data_pts = np.empty((size, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]

    
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)

    
    center = (int(mean[0, 0]), int(mean[0, 1])) # find the center of the object


    # This part is adapted a bit from the OpenCV docs
    # Draw the principal components
    cv.circle(img, center, 3, (255, 0, 255), 2)
    p1 = (center[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0],
          center[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
    p2 = (center[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0],
          center[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])
    drawAxis(img, center, p1, (255, 255, 0), 1)
    drawAxis(img, center, p2, (0, 0, 255), 5)

    # orientation in radians
    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])

    # Label with the rotation angle
    label = "  Rotation Angle: " + \
        str(-int(np.rad2deg(angle)) - 90) + " degrees"
    textbox = cv.rectangle(
        img, (center[0], center[1]-25), (center[0] + 250, center[1] + 10), (255, 255, 255), -1)
    cv.putText(img, label, (center[0], center[1]),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

    return angle


def main():
    img = cv.imread("in3.jpg")
    # vid = cv.VideoCapture(0)
    # while(1):
    # confirm, img = vid.read()
    # img = cv.resize(img, (640,400), interpolation= cv.INTER_LINEAR)
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Convert image to grayscale
    
    #thresholding for purple
    colour = np.uint8([[[255, 192, 203]]]) #here insert the bgr values which you want to convert to hsv
    hsvColour = cv.cvtColor(colour, cv.COLOR_BGR2HSV)

    lowerLimit = hsvColour[0][0][0] - 10, 100, 100
    upperLimit = hsvColour[0][0][0] + 10, 255, 255

    lowerLimit = tuple([int(i) for i in lowerLimit])
    upperLimit = tuple([int(i) for i in upperLimit])

    mask = cv.inRange(hsv_img, lowerLimit, upperLimit)
    result = cv.bitwise_and(img, img, mask = mask)
    cv.imshow('test',result)
    cv.waitKey()
    result = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
    _, bw = cv.threshold(result, 0, 225, cv.THRESH_BINARY) # Convert image to binary

    contours, _ = cv.findContours(bw, cv.RETR_FLOODFILL, cv.CHAIN_APPROX_NONE) # Find all the contours in the thresholded image

    for i, c in enumerate(contours):

        area = cv.contourArea(c)

        if area < 3700 or 100000 < area:
            continue

        cv.drawContours(img, contours, i, (0, 0, 255), 2)

        getOrientation(c, img)

    cv.imshow('Output Image', img)
    cv.waitKey(1)
    # cv.imwrite("output_img.jpg", img)


if __name__ == "__main__":
    main()
