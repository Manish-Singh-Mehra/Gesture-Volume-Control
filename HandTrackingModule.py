import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, model_complexity=1 ,detectionConfi=0.5, trackConfi=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.Complexity = model_complexity
        self.detectionConfi = detectionConfi
        self.trackConfi = trackConfi

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.Complexity,
                                        self.detectionConfi, self.trackConfi)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)
        if self.results.multi_hand_landmarks:
            for handLandmark in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLandmark, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:  # Added "self." before results
            myHand = self.results.multi_hand_landmarks[handNo]  # Removed indexing from HAND_CONNECTIONS
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 123, 241), cv2.FILLED)
        return lmList

def main():
    PreTime = 0
    CurrTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])
        CurrTime = time.time()
        fps = 1 / (CurrTime - PreTime)
        PreTime = CurrTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (245, 235, 240), 2)

        cv2.imshow('image', img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
