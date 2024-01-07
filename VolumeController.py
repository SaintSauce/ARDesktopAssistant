import cv2
import time
import numpy as np
import HandTracker as ht
import math
import os

import osascript

# from comtypes import CLSCTX_ALL
# from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# the program uses different libraries for volume control depending on OS
current_os = ""
os_name = os.name

if os_name == 'nt':
    current_os = "Windows"
elif os_name == 'posix':
    current_os = "macOS/Linux"
else:
    current_os = "Other"

print(f'You are currently running this program on {current_os}')

camera_width, camera_height = 1920, 1080

cap = cv2.VideoCapture(0)
cap.set(3, camera_width)
cap.set(4, camera_height)
prev_time = 0

detector = ht.HandTracker(detectionConfidence=0.8)

""" FOR WINDOWS """
# if os_name == "Windows":
#     devices = AudioUtilities.GetSpeakers()
#     interface = devices.Activate(
#         IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
#     volume = interface.QueryInterface(IAudioEndpointVolume)
#     # volume.GetMute()
#     # volume.GetMasterVolumeLevel()
#
#     volume_range = volume.GetVolumeRange()
#     MIN_VOLUME = volume_range[0]
#     MAX_VOLUME = volume_range[1]
# else:
MIN_VOLUME = 0
MAX_VOLUME = 100
""" FOR MAC """


while True:
    success, img = cap.read()
    detector.findHands(img)
    landmark_list = detector.findPosition(img, draw=False)

    if len(landmark_list) > 0:
        # print(landmark_list[4], landmark_list[8])

        x1, y1 = landmark_list[4][1], landmark_list[4][2]
        x2, y2 = landmark_list[8][1], landmark_list[8][2]
        center_x, center_y = (x1 + x2)//2, (y1 + y2)//2

        cv2.circle(img, (x1, y1), 15, (255, 0, 0), cv2.FONT_HERSHEY_SIMPLEX)
        cv2.circle(img, (x2, y2), 15, (255, 0, 0), cv2.FONT_HERSHEY_SIMPLEX)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.circle(img, (center_x, center_y), 15, (255, 0, 0), cv2.FONT_HERSHEY_SIMPLEX)

        fingers_distance = math.hypot(x2-x1, y2-y1)

        # fingers_distance range is about 50 - 300
        # volume_range is -65 to 0
        volume = np.interp(fingers_distance, [50, 300], [MIN_VOLUME, MAX_VOLUME])

        if current_os == "Windows":
            volume.SetMasterVolumeLevel(volume, None)

        osascript.osascript("set volume output volume {}".format(volume))

        if fingers_distance < 50:
            cv2.circle(img, (center_x, center_y), 15, (0, 0, 255), cv2.FILLED)

    curr_time = time.time()
    fps = 1/(curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(img, f'FPS: {str(int(fps))}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)