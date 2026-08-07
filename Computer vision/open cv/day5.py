import cv2
import os
image=cv2.imread(r"C:\Users\abidli\Desktop\Machine learning toolkit\assets\LinkedIn_icon.svg.webp")
folder_path=r"C:\Users\abidli\Desktop\Machine learning toolkit\assets"
file_name="Linkedin updated.jpeg"
full_path=os.path.join(folder_path,file_name)
cv2.imwrite(full_path,image)
cv2.imshow("Linkedin photo",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
vid=cv2.VideoCapture(r"C:\Users\abidli\Desktop\Machine learning toolkit\assets\Ronaldo Bicycle Kick + Player Reaction🥶🤯.mp4")
fps = vid.get(cv2.CAP_PROP_FPS)
ret=True
while ret:
    ret,frame=vid.read()
    if ret:
        cv2.imshow("Suiii",frame)
        cv2.waitKey(int(1000/fps))
vid.release()
cv2.destroyAllWindows()
webcam=cv2.VideoCapture(0)
while True:
    ret,frame=webcam.read()
    cv2.imshow("Webcam",frame)
    if(cv2.waitKey(1) & 0XFF==ord("q")):
        break
cv2.destroyAllWindows()
img=cv2.imread(r"C:\Users\abidli\Desktop\Machine learning toolkit\assets\Messi image.jpg")
print(img.shape)
image_resized=cv2.resize(img,(306,204))
folder_path=r"C:\Users\abidli\Desktop\Machine learning toolkit\assets"
file_name="Image_resized.jpg"
full_path=os.path.join(folder_path,file_name)
cv2.imwrite(full_path,image_resized)
image_cropped=img[30:378,30:582]
cv2.imshow("Window",image_cropped)
cv2.waitKey(0)