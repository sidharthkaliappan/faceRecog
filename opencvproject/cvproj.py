import cv2

cv2.namedWindow("test window",cv2.WINDOW_AUTOSIZE)
webcam = cv2.VideoCapture(0)

while(True):
    _,frame = webcam.read()
    cv2.imshow("test window", frame)

    if cv2.waitKey(20) & 0xFF == 27:
        break

webcam.release()
cv2.destroyAllWindows()