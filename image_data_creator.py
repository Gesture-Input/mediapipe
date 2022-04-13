import cv2
import mediapipe as mp
import time
import csv
import os

datasheet = open("data.csv",'a',newline='')
writer = csv.writer(datasheet)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    idx = 0
    for image_file in os.listdir("./imgs/paper/"):
        image = cv2.imread('./imgs/paper/' + image_file,1)
        results = hands.process(image)


        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            tmp = ['paper']
            for hand_landmarks in results.multi_hand_world_landmarks:
                for id, lm in enumerate(hand_landmarks.landmark):
                    if(id % 4 == 0 and id != 0):
                        tmp.append(lm.x)
                        tmp.append(lm.y)
            writer.writerow(tmp)
        print("paper file :" + str(idx) + " completed")
        idx += 1

    idx = 0
    for image_file in os.listdir("./imgs/rock/"):
        image = cv2.imread('./imgs/rock/' + image_file,1)
        results = hands.process(image)


        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            tmp = ['paper']
            for hand_landmarks in results.multi_hand_world_landmarks:
                for id, lm in enumerate(hand_landmarks.landmark):
                    if(id % 4 == 0 and id != 0):
                        tmp.append(lm.x)
                        tmp.append(lm.y)
            writer.writerow(tmp)
        print("rock image file :" + str(idx) + " completed")
        idx += 1


datasheet.close()
