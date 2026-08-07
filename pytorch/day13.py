import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer,OneHotEncoder
from sklearn.metrics import mean_absolute_error
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader
df=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\goals.csv")
print(df.shape)
print(df.columns)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
df.columns=(df.columns
            .str.lower()
            .str.replace("_"," ")
            .str.strip()
            )
print(df.info())
df["ratio right foot goals"]=df["right foot"]/df["goals"]
y=df["ratio right foot goals"]
X=df.drop(columns=["serial","ratio right foot goals"])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
cat_cols=X_train.select_dtypes(include=["str","object"]).columns
num_cols=X_train.select_dtypes(exclude=["str","object"]).columns
scaler=QuantileTransformer()
encoder=OneHotEncoder(sparse_output=False,handle_unknown="ignore")
X_train_cat=encoder.fit_transform(X_train[cat_cols])
X_test_cat=encoder.transform(X_test[cat_cols])
X_train_num=scaler.fit_transform(X_train[num_cols])
X_test_num=scaler.transform(X_test[num_cols])
X_train=np.hstack([X_train_cat,X_train_num])
X_test=np.hstack([X_test_cat,X_test_num])
X_train=torch.from_numpy(X_train).float().to(device)
X_test=torch.from_numpy(X_test).float().to(device)
y_train=torch.from_numpy(y_train.values).float().unsqueeze(1).to(device)
y_test=torch.from_numpy(y_test.values).float().unsqueeze(1).to(device)
train_dataset=TensorDataset(X_train,y_train)
train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
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
model=NeuralNetwork().to(device)
criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=0.002)
epochs=200
for epoch in range(epochs):
    model.train()
    curr_loss=0.00
    for x_batch,y_batch in train_loader:
        optimizer.zero_grad()
        predictions=model(x_batch)
        MSE=criterion(predictions,y_batch)
        MSE.backward()
        optimizer.step()
        curr_loss+=MSE.item()
    print(f"epoch {epoch} curr_loss= {curr_loss/len(train_loader)}")
with torch.no_grad():
    model.eval()
    predictions=model(X_test)
    MSE=criterion(predictions,y_test)
    print(f"MSE LOSS= {MSE.item()} mean_absolute_error= {mean_absolute_error(y_test.cpu().numpy(),predictions.cpu().numpy())}")


        

