import cv2
import mediapipe as mp
import time

class HandTracker():
    def __init__(self, mode=False,
                 maxNumHands=2,
                 modelComplexity=1,
                 detectionConfidence=0.5,
                 trackingConfidence=0.5):
        self.mode = mode
        self.maxNumHands = maxNumHands
        self.modelComplexity = modelComplexity
        self.detectionConfidence = detectionConfidence
        self.trackingConfidence = trackingConfidence

        # create Hands() object
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,
                                        self.maxNumHands,
                                        self.modelComplexity,
                                        self.detectionConfidence,
                                        self.trackingConfidence)

        # drawing utilities
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
        self.results = self.hands.process(imgRGB)  # process the images
        # print(results.multi_hand_landmarks)

        # can extract information for each hand
        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                if draw:
                    # draw hand landmarks and hand connections
                    self.mpDraw.draw_landmarks(img, hand, self.mpHands.HAND_CONNECTIONS)

        # just spend an hour trying to fix a bug
        # where the image is empty and just realized
        # it was because the return was inside the if statement
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.landmark_list = []

        if self.results.multi_hand_landmarks:
            newHand = self.results.multi_hand_landmarks[handNo]
            for id, landmark in enumerate(newHand.landmark):
                # print(id, landmark)
                height, width, channel = img.shape
                center_x, center_y = int(landmark.x * width), int(landmark.y * height)
                # print(id, center_x, center_y)
                self.landmark_list.append([id, center_x, center_y])

                if draw:
                    cv2.circle(img, (center_x, center_y), 15,
                               (0, 0, 255), cv2.FILLED)

        return self.landmark_list

    def fingersUp(self):
        fingers = []

        # thumb
        if self.landmark_list[self.tipIds[0]][1] < self.landmark_list[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 fingers
        for id in range(1, 5):
            if self.landmark_list[self.tipIds[id]][2] < self.landmark_list[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

def main():
    # calculate framerate
    prev_time = 0
    curr_time = 0

    # webcam (# depends on the machine)
    cap = cv2.VideoCapture(0)

    # initialize hand tracker instance
    detector = HandTracker()

    while True:
        success, img = cap.read()

        if not success:
            print("Image not found.")
            break

        img = detector.findHands(img)

        if img is None:
            print("Empty frame.")
            break

        landmark_list = detector.findPosition(img)

        if len(landmark_list) != 0:
            print(landmark_list[4])

        # calculate framerate
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(img, str(int(fps)),
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()