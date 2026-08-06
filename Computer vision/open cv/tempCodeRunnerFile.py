import os
import cv2
image=cv2.imread(r"C:\Users\abidli\Desktop\Machine learning toolkit\assets\Messi image.jpg")
print(image.shape)
folder_name=r"C:\Users\abidli\Desktop\Machine learning toolkit\assets"
file_name="output.jpg"
full_path=os.path.join(folder_name,file_name)
cv2.imwrite(full_path,image)