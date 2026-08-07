#TALEL Training
import cv2
import os
vid=cv2.VideoCapture(r"C:\Users\abidli\Desktop\Machine learning toolkit\assets\1672952571967.jpeg")
fps=vid.get(cv2.CAP_PROP_FPS)
ret=True
while ret:
    ret,frame=vid.read()
    if ret:
        cv2.imshow("Talel GRIND",frame)
        cv2.waitKey(int(1000/fps))