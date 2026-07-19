#Revision for day1  Numpy
import pandas as pd
import numpy as np
array=np.array([1,3,5,7,9])
print(array)
array=np.arange(10,0,-1)
print(array)
array=np.zeros(10)
print(array)
array=np.full(10,6)
print(array)
#Execlusive for day2 (arrays slicing)
array=np.array([1,5,7,8,4,3])
#[5,7,8,4]
print(array[1:5])
#[8,4,3]
print(array[3:])
#[8,4]
print(array[-3:-1])
#get the last 3 elements
print(array[-3:])
#[::step] to skip elemnts based on step
print(array[::2])
#reverse the array
print(array[::-1])
#2d arrays 
import numpy as np

arr2d = np.array([
    [1,  2,  3,  4],   # row 0
    [5,  6,  7,  8],   # row 1
    [9, 10, 11, 12]    # row 2
])
print(arr2d[0])
print(arr2d[:, 1])
print(arr2d[0:2, 0:2])
