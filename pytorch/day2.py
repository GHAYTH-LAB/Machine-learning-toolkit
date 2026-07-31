"""import pandas as pd
import numpy as np
import torch
tensor=torch.tensor([[1,3,5],[7,9,11],[11,19,6]])
index_extract=torch.tensor([[2],[0],[1]])
Selected_values=torch.gather(tensor,dim=1,index=index_extract)
print(Selected_values)
"""
import torch
import pandas as pd
import numpy as np
nd_array=np.array([[1,2,4],[3,5,7],[7,1,6]])
df=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\numerical_sample_dataset.csv")
tensor=torch.tensor(df.values,requires_grad=True,dtype=torch.float32)
print(tensor.device)
print(tensor.shape)
print(tensor.dtype)
print(tensor)
print(tensor.mean(dim=1))
print(tensor.mean(dim=1))
tensor1=torch.randint_like(tensor,low=0,high=8,requires_grad=True,dtype=torch.float32)
print(torch.cuda.is_available())
print(tensor1.mean(dim=1))
positions_to_extract=torch.tensor([[1],[0],[2]])
print(torch.gather(tensor1,dim=1,index=positions_to_extract))
