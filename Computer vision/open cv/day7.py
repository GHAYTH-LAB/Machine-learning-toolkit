import cv2
import os
image=cv2.imread(r"C:\Users\abidli\Desktop\Machine learning toolkit\assets\Cat.jpg")
folder_path=r"C:\Users\abidli\Desktop\Machine learning toolkit\assets"
file_name="Image.jpg"
full_path=os.path.join(folder_path,file_name)
cv2.imwrite(full_path,image)
cv2.imshow("cat",image)
cv2.waitKey(0)
vid=cv2.VideoCapture(r"C:\Users\abidli\Desktop\Machine learning toolkit\assets\Ronaldo Bicycle Kick + Player Reaction🥶🤯.mp4")
fps=vid.get(cv2.CAP_PROP_FPS)
ret=True
while ret:
    ret,frame=vid.read()
    if ret:
        cv2.imshow("Ronaldo",frame)
        cv2.waitKey(int(1000/fps))
webcam=cv2.VideoCapture(0)
while True:
    ret,frame=webcam.read()
    cv2.imshow("webcam",frame)
    if (cv2.waitKey(1) & 0xFF) == ord("q"):
        break
image2=cv2.imread(r"C:\Users\abidli\Desktop\Machine learning toolkit\assets\ieee_insat_student_branch_logo.jpeg")
image_converted=cv2.cvtColor(image2,cv2.COLOR_RGB2BGR)
image_blurred1=cv2.blur(image2,(9,9))
image_blurred2=cv2.medianBlur(image,9)
image_blurred3=cv2.GaussianBlur(image,(11,11),0)
cv2.imshow("Image blurred1",image_blurred1)
cv2.waitKey(0)
cv2.imshow("Image blurred2",image_blurred2)
cv2.waitKey(0)
cv2.imshow("Image blurred3",image_blurred3)
cv2.waitKey(0)