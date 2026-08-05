import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.metrics import f1_score,accuracy_score,precision_score
import pandas as pd
import numpy as np
df=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\housing.csv")
print(df.duplicated().sum())
print(df.isna().sum())
df.columns=(df.columns
            .str.lower()
            .str.strip()
            .str.replace("_"," ")
            )
df=df.fillna({
    "total bedrooms":df["total bedrooms"].median()
})
y=df["ocean proximity"]
X=df.drop(columns="ocean proximity")
print(df.info())
print(df["ocean proximity"].unique())
class_names=df["ocean proximity"].unique()
class_to_index={
    name:i for i,name in enumerate(class_names)
}
y=y.map(class_to_index)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
X_train=torch.from_numpy(X_train).float()
X_test=torch.from_numpy(X_test).float()
y_train=torch.from_numpy(y_train.values).long()
y_test=torch.from_numpy(y_test.values).long()
train_dataset=TensorDataset(X_train,y_train)
train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(X_train.shape[1],64)
        self.layer2=nn.Linear(64,256)
        self.layer3=nn.Linear(256,32)
        self.layer4=nn.Linear(32,128)
        self.layer5=nn.Linear(128,16)
        self.layer6=nn.Linear(16,5)
    def forward(self,x):
        x=F.relu(self.layer1(x))
        x=F.relu(self.layer2(x))
        x=F.leaky_relu(self.layer3(x))
        x=F.relu(self.layer4(x))
        x=F.tanh(self.layer5(x))
        x=self.layer6(x)
        return x
model=NeuralNetwork()
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.002)
epochs=200
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
    print(f"epoch {epoch+1} avergae loss = {curr_loss/len(train_loader)}")
with torch.no_grad():
    model.eval()
    predictions=model(X_test)
    loss=criterion(predictions,y_test)
    class_predicted=torch.argmax(predictions,dim=1)
    print(f"loss= {loss.item()} accuracy_score= {accuracy_score(y_test.numpy(),class_predicted.numpy())} precision_score= {precision_score(y_test.numpy(),class_predicted.numpy(),average="weighted")} ")