import cv2
import os
image=cv2.imread(r"C:\Users\abidli\Desktop\Machine learning toolkit\assets\4656c67b-46a9-4c91-af5e-48f0063adcfb.jpeg")
print(image.shape)
folder_path=r"C:\Users\abidli\Desktop\Machine learning toolkit\assets"
file_name="Cs_chapter.jpg"
full_path=os.path.join(folder_path,file_name)
cv2.imwrite(full_path,image)
cv2.imshow("Cs chapter",image)
vid=cv2.VideoCapture(r"C:\Users\abidli\Desktop\Machine learning toolkit\assets\Kitty meowing to attract cats.mp4")
fps=vid.get(cv2.CAP_PROP_FPS)
ret=True
while ret:
    ret,frame=vid.read()
    if ret:
        cv2.imshow("Video display",frame)
        cv2.waitKey(int(1000/fps))
webcam=cv2.VideoCapture(0)
while True:
    ret,frame=webcam.read()
    cv2.imshow("Webcam",frame)
    if(cv2.waitKey(50) & 0XFF==ord("q")):
        break
image_resized=cv2.resize(image,(640,640))
image_cropped=image[30:1265,100:1948]
cv2.imshow("Image resized",image_resized)
cv2.waitKey(0)
cv2.imshow("Image cropped",image_cropped)
cv2.waitKey(0)
webcam.release()
cv2.destroyAllWindows()
img_converted=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
cv2.imshow("changed the colorspace",img_converted)
cv2.waitKey(0)

img2_converted=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("GRAYSCALE",img2_converted)
cv2.waitKey(0)