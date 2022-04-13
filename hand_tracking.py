import cv2
import mediapipe as mp
import time

flag = "rock"

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(1)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      print(flag)
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
      for hand_landmarks in results.multi_hand_world_landmarks:
        for id, lm in enumerate(hand_landmarks.landmark):
          sum = 0
          if(id % 4 == 0 and id != 0):
            h, w, c = image.shape
            cx, cy = (lm.x * w), (lm.y * h)
            sum += round(cx*cx + cy*cy,2)
          if(sum <= 350 and flag != "rock" and sum != 0):
            flag = "rock"
          elif(sum >= 800 and flag != "paper" and sum != 0):
            flag = "paper" 
            
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()