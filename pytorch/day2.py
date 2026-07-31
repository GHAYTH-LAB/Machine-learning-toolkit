import pandas as pd
import numpy as np
import torch
tensor=torch.tensor([[1,3,5],[7,9,11],[11,19,6]])
index_extract=torch.tensor([[2],[0],[1]])
Selected_values=torch.gather(tensor,dim=1,index=index_extract)
print(Selected_values)
