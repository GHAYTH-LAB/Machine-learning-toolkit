"""import pandas as pd
import numpy as np
import torch
print(torch.cuda.is_available())
print(torch.device)
if torch.cuda.is_available():
    torch.device="cuda"
print(torch.device)
print(torch.cuda.get_device_name(0))
if torch.cuda.is_available():
    torch.device="cuda"
tensor=torch.tensor([[1,3,5],[7,9,11],[6,8,10]],requires_grad=True)
tensor=tensor.to("cuda")
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import dataloader,TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X,y=load_breast_cancer( return_X_y=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
