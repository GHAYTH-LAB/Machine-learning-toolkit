import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
df=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\housing.csv")
print(df.shape)
print(df.columns)
print(df.isna().sum())
print(df.info())
print(df.duplicated().sum())
df.columns=(df.columns
            .str.lower()
            .str.replace("_"," "))
df["abs longitude"]=df["longitude"].apply(lambda x:abs(x))
print(df["abs longitude"])
df["longitude*latitude"]=df["longitude"]*df["latitude"]
df["how many persons in one room"]=df["total rooms"]/df["population"]
df=df.fillna({
    "total bedrooms":df["total bedrooms"].median()
})
df["ratio of bedrooms per rooms"]=df["total bedrooms"]/df["total rooms"]
df=df.drop(columns="longitude")
print(df.info())
y=df["median house value"]
X=df.drop(columns=["median house value"])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
cat_cols=X_train.select_dtypes(include=["str","object"]).columns
num_cols=X_train.select_dtypes(exclude=["str","object"]).columns
encoder=OneHotEncoder(sparse_output=False,handle_unknown="ignore")
X_train_cat=encoder.fit_transform(X_train[cat_cols])
X_test_cat=encoder.transform(X_test[cat_cols])
scaler=MinMaxScaler()
X_train_num=scaler.fit_transform(X_train[num_cols])
X_test_num=scaler.transform(X_test[num_cols])
X_train=np.hstack([X_train_num,X_train_cat])
X_test=np.hstack([X_test_num,X_test_cat])
X_train=torch.from_numpy(X_train).float()
X_test=torch.from_numpy(X_test).float()
y_train=torch.from_numpy(y_train.values).float().unsqueeze(1)
y_test=torch.from_numpy(y_test.values).float().unsqueeze(1)
train_dataset=TensorDataset(X_train,y_train)
train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(X_train.shape[1],64)
        self.layer2=nn.Linear(64,16)
        self.layer3=nn.Linear(16,32)
        self.layer4=nn.Linear(32,64)
        self.layer5=nn.Linear(64,16)
        self.layer6=nn.Linear(16,1)
    def forward(self,x):
        x=F.relu(self.layer1(x))
        x=F.relu(self.layer2(x))
        x=F.relu(self.layer3(x))
        x=F.relu(self.layer4(x))
        x=F.relu(self.layer5(x))
        x=self.layer6(x)
        return x
model=NeuralNetwork()
criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=0.005)
epochs=150
for epoch in range(epochs):
    model.train()
    curr_loss=0.00
    for x_batch,y_batch in train_loader:
        optimizer.zero_grad()
        predictions=model(x_batch)
        loss=criterion(predictions,y_batch)
        loss.backward()
        optimizer.step()
        curr_loss+=loss.item()
    print(f"epoch {epoch+1} loss= {curr_loss/len(train_loader)}")
with torch.no_grad():
    model.eval()
    predictions=model(X_test)
    loss=criterion(predictions,y_test)
    print(f"loss= {loss.item()}  r2_score= {r2_score(y_test.numpy(),predictions.numpy())} mean_absolute_error {mean_absolute_error(y_test.numpy(),predictions.numpy())} ")