import cv2
import numpy as np
import imutils

cap = cv2.VideoCapture('/Users/vinhpham/Desktop/AI Programming/Source/pingpong.mp4')

scoreA = 0
scoreB = 0

checkA = False
checkB = False

while(True):
    _, frame = cap.read()

    blur = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    lower = np.array([15, 130, 30])
    upper = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    ball_cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ball_cnts = imutils.grab_contours(ball_cnts)

    if (len(ball_cnts) > 0):
        c = max (ball_cnts, key = cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)

    if (x < 900 and checkA == False):
        scoreB += 1
        checkA = True
        checkB = False
        
    if (x > 900 and checkB == False):
        scoreA += 1
        checkB = True
        checkA = False

    cv2.putText(img = frame, text = "Player A: " + str(scoreA), org = (50, 50), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (255, 0, 0), thickness = 2)
    cv2.putText(img = frame, text = "Player B: " + str(scoreB), org = (50, 100), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (255, 0, 0), thickness = 2)
    cv2.line(img = frame, pt1 = (900, 300), pt2 = (900, 700), color = (0, 255, 255), thickness = 4)

    cv2.imshow("frame", frame)

    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

