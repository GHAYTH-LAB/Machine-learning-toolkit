import pandas as pd 
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import root_mean_squared_error
df=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\rr_train.csv")
df1=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\rr_test.csv")
df.columns=df.columns.str.lower()
df1.columns=df1.columns.str.lower()
y_train=df["revenue"]
id_extracted=df1["id"]
X_test=df1.drop(columns="id")
X_train=df.drop(columns=["revenue"])
scaler=QuantileTransformer(n_quantiles=37)
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
X_train=torch.from_numpy(X_train).float()
X_test=torch.from_numpy(X_test).float()
y_train=torch.from_numpy(y_train.values).float().unsqueeze(1)
train_dataset=TensorDataset(X_train,y_train)
data_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(X_train.shape[1],64)
        self.layer2=nn.Linear(64,128)
        self.layer3=nn.Linear(128,32)
        self.layer4=nn.Linear(32,16)
        self.layer5=nn.Linear(16,1)
    def forward(self,x):
        x=F.relu(self.layer1(x))
        x=F.relu(self.layer2(x))
        x=F.relu(self.layer3(x))
        x=F.relu(self.layer4(x))
        x=self.layer5(x)
        return x
model=NeuralNetwork()
criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)
epochs=300
#Training loop
for epoch in range(epochs):
    model.train()
    curr_loss=0.00
    for x_batch,y_batch in data_loader:
        optimizer.zero_grad()
        predictions=model(x_batch)
        loss=criterion(predictions,y_batch)
        loss.backward()
        optimizer.step()
        curr_loss+=loss.item()
    if(epoch%5==0):
        print(f"epoch {epoch} avg loss={curr_loss/len(data_loader)}")
with torch.no_grad():
    model.eval()
    predictions=model(X_test)
submission=pd.DataFrame({
    "ID":id_extracted
    ,"revenue":predictions.numpy().flatten()
})
submission.to_csv("Submission kaggle Restaurant Revenue Prediction2.csv",index=False)

