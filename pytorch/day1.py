"""import torch
import numpy as np
array=np.array([[1,2,7,9],[11,3,9,6]])
tensor=torch.tensor(array)
print(tensor)
"""
"""
import torch
import numpy as np
tensor=torch.tensor([[1,2,3],[9,8,7],[6,4,5]])
print(tensor)"""
import torch 
import pandas as pd
import numpy as np
df=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\numerical_sample_dataset.csv")
tensor=torch.tensor(df.values,dtype=torch.float32)
print(tensor)
print(tensor.dtype)
#for initializing weights
zeros=torch.ones(2,3)
ones=torch.ones(2,3)
random_tensor=torch.randint(0,10,(2,3))
print(zeros)
print(ones)
print(random_tensor)
#Need a New tensor as the old one(shape,type)
New_tensor=torch.randint_like(random_tensor,low=5,high=11,dtype=torch.float32) #the high param is exlusive it goes just to high-1
print(New_tensor) 