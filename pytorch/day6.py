import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
df=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\numerical_dataset.csv")
y=df["target"]
X=df.drop(columns="target")
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
X_train=torch.from_numpy(X_train).float()
X_test=torch.from_numpy(X_test).float()
y_train=torch.from_numpy(y_train.values).float().unsqueeze(1)
y_test=torch.from_numpy(y_test.values).float().unsqueeze(1)
train_dataset=TensorDataset(X_train,y_train)
data_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(8,32)
        self.layer2=nn.Linear(32,64)
        self.layer3=nn.Linear(64,16)
        self.layer4=nn.Linear(16,1)
    def forward(self,x):
        x=F.relu(self.layer1(x))
        x=F.relu(self.layer2(x))
        x=F.relu(self.layer3(x))
        x=F.sigmoid(self.layer4(x))
        return x
model=NeuralNetwork()
criterion=nn.BCELoss()
optimizer=optim.Adam(model.parameters(),lr=0.002)
epochs=350
for epoch in range(epochs):
    model.train()
    current_loss=0.0
    for x_batch,y_batch in data_loader:
        optimizer.zero_grad()
        predictions=model(x_batch)
        loss=criterion(predictions,y_batch)
        loss.backward()
        optimizer.step()
        current_loss+=loss.item()
    print(f"{epoch+1} l:loss was {current_loss/len(data_loader)}")
with torch.no_grad():
    model.eval()
    predictions=model(X_test)
    loss=criterion(predictions,y_test)
    print("predictions",((predictions>=0.5)==y_test).float().mean().item())
