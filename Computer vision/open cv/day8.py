import os
import cv2
image=cv2.imread(r"C:\Users\abidli\Desktop\Machine learning toolkit\assets\Board.jpeg")
cv2.imshow("Window",image)
cv2.waitKey(0)
video=cv2.VideoCapture(r"C:\Users\abidli\Desktop\Machine learning toolkit\assets\Ronaldo Bicycle Kick + Player Reaction🥶🤯.mp4")
fps=video.get(cv2.CAP_PROP_FPS)
ret=True
while ret:
    ret,frame=video.read()
    if ret:
        cv2.imshow("ronaldo",frame)
        cv2.waitKey(int(1000/fps))
webcam=cv2.VideoCapture(0)
while True:
    ret,frame=webcam.read()
    cv2.imshow("Webcam",frame)
    if(cv2.waitKey(50) & 0XFF==ord("q")):
        break
image_changed=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image_cropped=image[230:255,23:25]
cv2.line(image,(255,29),(201,340),(0,255,0),3)
cv2.imshow("Image",image)
cv2.waitKey(0)
cv2.line(image,(33,34),(331,226),3)
cv2.imshow("Image + line",image)
cv2.waitKey(0)
image=cv2.imread(r"C:\Users\abidli\Desktop\Machine learning toolkit\assets\Profile_-_SpongeBob_SquarePants.webp")
cv2.rectangle(image,(35,29),(244,341),(0,0,255),3)
cv2.imshow("Spongbob",image)
cv2.waitKey(0)
cv2.circle(image,(117,150),50,(255,0,0),3)
cv2.imshow("Circle",image)
cv2.waitKey(0)
cv2.putText(image,"Hey MF",(6,15),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,255),2,cv2.LINE_AA)
cv2.imshow("Image + text",image)
cv2.waitKey(0)