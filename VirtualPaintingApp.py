import cv2
import numpy as np
import time
import os
import HandTracker as ht

imgPath = "PaintingAppMenuHeaders"
imgList = os.listdir(imgPath)
utilsList = []
print(f'Listed overlay images : {imgList}')

for img in imgList:
    image = cv2.imread(f'{imgPath}/{img}')
    utilsList.append(image)

print(f'Directory size : {len(utilsList)}')

# app setup
header = utilsList[-1]
drawColor = (255, 0, 255)
brushThickness = 15
eraserThickness = 100

x_prev, y_prev = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = ht.HandTracker(detectionConfidence=0.8)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # find hand landmarks
    img = detector.findHands(img)
    landmark_list = detector.findPosition(img, draw=False)

    if len(landmark_list) > 0:
        # print(landmark_list)

        # get the tip of the index and middle fingers
        x1, y1 = landmark_list[8][1:]
        x2, y2 = landmark_list[12][1:]

        # get fingers
        fingers = detector.fingersUp()

        # menu selection mode (both index and middle fingers up)
        if fingers[1] and fingers[2]:
            x_prev, y_prev = 0, 0
            print("Selection mode")

            """ this part will need some big fixing """
            # fingers within header
            if y1 < 125:
                # red
                if 180 < x1 < 250:
                    header = utilsList[6]
                    drawColor = (0, 0, 255)

                # orange
                elif 310 < x1 < 380:
                    header = utilsList[7]
                    drawColor = (0, 127, 255)

                # yellow
                elif 440 < x1 < 510:
                    header = utilsList[2]
                    drawColor = (0, 255, 255)

                # green
                elif 575 < x1 < 640:
                    header = utilsList[3]
                    drawColor = (0, 255, 0)

                # light blue
                elif 700 < x1 < 770:
                    header = utilsList[5]
                    drawColor = (255, 255, 0)

                # blue
                elif 830 < x1 < 900:
                    header = utilsList[4]
                    drawColor = (255, 0, 0)

                # purple
                elif 960 < x1 < 1030:
                    header = utilsList[0]
                    drawColor = (255, 0, 127)

                # eraser
                elif 1095 < x1 < 1200:
                    header = utilsList[1]
                    drawColor = (0, 0, 0)

            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # drawing mode
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing mode")

            if x_prev == 0 and y_prev == 0:
                x_prev, y_prev = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (x_prev, y_prev), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (x_prev, y_prev), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (x_prev, y_prev), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (x_prev, y_prev), (x1, y1), drawColor, brushThickness)

            x_prev, y_prev = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInverse = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInverse = cv2.cvtColor(imgInverse, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInverse)
    img = cv2.bitwise_or(img, imgCanvas)

    # Header UI
    img[0:125, 0:1280] = header
    img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    cv2.imshow("Inverse", imgInverse)
    cv2.waitKey(1)