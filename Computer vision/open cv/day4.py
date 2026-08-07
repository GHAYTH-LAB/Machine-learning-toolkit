#MANPULATING A WEBCAM
import os
import cv2
webcam=cv2.VideoCapture(0)
while True:
    ret,frame=webcam.read()
    cv2.imshow("Webcam",frame)
    if(cv2.waitKey(50) & 0XFF==ord("q")):
        break
webcam.release()
cv2.destroyAllWindows()

