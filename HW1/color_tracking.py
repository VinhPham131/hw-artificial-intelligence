import cv2
import numpy as np

draw_color = (0, 0, 255)

def main():
    cap = cv2.VideoCapture(0)
    canvas = None
    prev_center = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if canvas is None:
            canvas = np.zeros_like(frame)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red = np.array([0, 150, 150])
        upper_red = np.array([10, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        mask = mask1

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 500:
                (x, y, w, h) = cv2.boundingRect(c)
                center = (x + w//2, y + h//2)
                cv2.circle(frame, center, 5, (255, 0, 0), -1)
                if prev_center is not None:
                    cv2.line(canvas, prev_center, center, draw_color, 5)
                prev_center = center
            else:
                prev_center = None
        else:
            prev_center = None

        output = cv2.add(frame, canvas)

        cv2.imshow("Mask", mask)
        cv2.imshow("Air Drawing", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
