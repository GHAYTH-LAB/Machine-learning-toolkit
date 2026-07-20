import numpy as np
arr=np.arange(1,9,2)
arr1=np.array([1,3,7,9,-1])
arr1=arr1[::-1]
arr1=np.abs(arr1)
#to iterate 
for x in np.nditer(arr1):
    print(x,end="/")
arr2=np.full((4,3),1)
print("\n")
for x in np.nditer(arr2):
    print( x,end="/")
arr3=np.array([[1,3,5],[2,4,6]])
print("\n")
print(arr3)
for x in np.nditer(arr3):
    print(x,end="/")
