# Reminder of python lists
List=[1,2,3,4]
print(List)
# we acces elements by index(first element index is 0)
print(List[0])
#Numpy=Numeric python
#ndarray=n-dimensional array()
import numpy as np
array=np.array([0,1,2,3,4,5,6,7,8,9])
print(array)
print(array.shape)
array2=np.arange(10)
print(array2)
array3=np.arange(4,11,2)
print(array3)
New_array=np.zeros(10)
print(New_array)
New_array1=np.zeros((3,4))
print(New_array1)
New_array2=np.full((10),6)
print(New_array2)
New_array3=np.full((5,4),8)
print(New_array3)
print(array3[3])