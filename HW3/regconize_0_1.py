import cv2
import numpy as np
from collections import deque

theta = np.loadtxt('/Users/vinhpham/Desktop/AI Programming/HW3/theta.txt')
print("Kích thước theta:", theta.shape)

def sigmoid(s):
    s = np.clip(s, -500, 500)
    return 1 / (1 + np.exp(-s))

cap = cv2.VideoCapture(0)
pred_buffer = deque(maxlen=5)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    thresh = cv2.dilate(thresh, (3, 3))
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        if w < 20 or h < 20:
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        length = int(max(w, h) * 1.6)
        pt1 = max(int(y + h // 2 - length // 2), 0)
        pt2 = max(int(x + w // 2 - length // 2), 0)
        roi = thresh[pt1:pt1 + length, pt2:pt2 + length]

        try:
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        except:
            continue

        roi = roi.reshape(1, 28 * 28)
        roi = roi / 255.0 

        x_input = np.concatenate((roi, np.ones((1, 1))), axis=1)

        prob = sigmoid(np.dot(x_input, theta.T))
        pred = int(prob > 0.5)  

        cv2.putText(frame, str(pred), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (0, 255, 0), 3)

    cv2.imshow("Handwritten Digit Detection", frame)
    cv2.imshow("Threshold (White Text, Black Background)", thresh)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
