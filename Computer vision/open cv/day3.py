import cv2
import os
image=cv2.imread(r"C:\Users\abidli\Desktop\Machine learning toolkit\assets\image-178.webp")
folder_path=r"C:\Users\abidli\Desktop\Machine learning toolkit\assets"
file_name="opencv image updated.jpg"
full_path=os.path.join(folder_path,file_name)
cv2.imwrite(full_path,image)
cv2.imshow("WINDOW",image)
cv2.waitKey(0)
vid=cv2.VideoCapture(r"C:\Users\abidli\Desktop\Machine learning toolkit\assets\Ronaldo Bicycle Kick + Player Reaction🥶🤯.mp4")
ret=True
fps = vid.get(cv2.CAP_PROP_FPS)
while ret:
    ret,frame=vid.read()
    if ret:
        cv2.imshow("Suiiiiiiiiii",frame)
        cv2.waitKey(int(1000/fps))
vid.release()
cv2.destroyAllWindows()