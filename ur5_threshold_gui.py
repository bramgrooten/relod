# Preview: https://raw.githubusercontent.com/FedericoPonzi/LegoLab/master/media/hsv-colour.png
import time
import cv2
import sys

import numpy as np

def nothing(x):
    pass

def main():
    # Create a window
    cv2.namedWindow('image')

    # create trackbars for color change
    cv2.createTrackbar('bMin','image',0,255,nothing) # Hue is from 0-179 for Opencv, not 255
    cv2.createTrackbar('gMin','image',0,255,nothing)
    cv2.createTrackbar('rMin','image',0,255,nothing)
    cv2.createTrackbar('BMax','image',0,255,nothing)
    cv2.createTrackbar('GMax','image',0,255,nothing)
    cv2.createTrackbar('RMax','image',0,255,nothing)

    # Set default value for MAX HSV trackbars.
    cv2.setTrackbarPos('BMax', 'image', 255)
    cv2.setTrackbarPos('GMax', 'image', 255)
    cv2.setTrackbarPos('RMax', 'image', 255)

    # Initialize to check if HSV min/max value changes
    rMin = gMin = bMin = rMax = gMax = bMax = 0
    prMin = pgMin = pbMin = pRMax = pGMax = pBMax = 0

    # Output Image to display
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FPS, 25)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 90)
    waitTime = 330

    while(1):
        ret, img = cap.read()

        # get current positions of all trackbars
        rMin = cv2.getTrackbarPos('rMin','image')
        gMin = cv2.getTrackbarPos('gMin','image')
        bMin = cv2.getTrackbarPos('bMin','image')

        RMax = cv2.getTrackbarPos('RMax','image')
        GMax = cv2.getTrackbarPos('GMax','image')
        BMax = cv2.getTrackbarPos('BMax','image')

        # Set minimum and max HSV values to display
        lower = np.array([bMin, gMin, rMin])
        upper = np.array([BMax, GMax, RMax])

        mask = cv2.inRange(img, lower, upper)


        # img_th = cv2.adaptiveThreshold(
        #     mask,
        #     255,
        #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #     cv2.THRESH_BINARY,
        #     11,
        #     1
        # )

        # Display output image
        new_img = np.concatenate((img, 
                    cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)), 
                    # cv2.cvtColor(img_th, cv2.COLOR_GRAY2BGR)), 
                    axis=1)
        cv2.imshow('image', new_img)

        # Wait longer to prevent freeze for videos.
        if cv2.waitKey(waitTime) & 0xFF == ord('q'):
            break


    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
