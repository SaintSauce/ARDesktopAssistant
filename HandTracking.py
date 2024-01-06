import cv2
import mediapipe as mp
import time

# webcam (# depends on the machine)
cap = cv2.VideoCapture(0)

# create Hands() object
mpHands = mp.solutions.hands
hands = mpHands.Hands()

# drawing utilities
mpDraw = mp.solutions.drawing_utils

# calculate framerate
prev_time = 0
curr_time = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)       # convert BGR to RGB
    results = hands.process(imgRGB)                     # process the images
    # print(results.multi_hand_landmarks)

    # can extract information for each hand
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            for id, landmark in enumerate(hand.landmark):
                # print(id, landmark)
                height, width, channel = img.shape
                center_x, center_y = int(landmark.x * width), int(landmark.y * height)
                print(id, center_x, center_y)

            # draw hand landmarks and hand connections
            mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)

    # calculate framerate
    curr_time = time.time()
    fps = 1/(curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(img, str(int(fps)),
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)