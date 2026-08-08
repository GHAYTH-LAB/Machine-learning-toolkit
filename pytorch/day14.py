import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.metrics import r2_score,mean_absolute_error
df=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\Data.csv")
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
print(df.shape)
print(df.duplicated().sum())
print(df.isna().sum())
df.columns=(df.columns
            .str.lower()
            .str.strip()
            .str.replace("_"," ")
            )
print(df.info())
df=df.fillna({
    "club":df["club"].mode()[0]
})
print(df["league"].nunique())
df["average minutes per match"]=df["mins"]/df["matches played"]
df["substitutions per match"]=df["substitution"]/df["matches played"]
df["goals per match"]=df["goals"]/df["matches played"]
df["goals per minute"]=df["goals"]/df["mins"]
df["scores more than expected"]=df["goals"]>df["xg"]
df["shots per minute"]=df["shots"]/df["mins"]
df["shots to score"]=df["shots"]/df["goals"]
df["ratio of shots on target"]=df["ontarget"]/df["shots"]
y=df["xg"]
X=df.drop(columns="xg")
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
cat_cols=X_train.select_dtypes(include=["str","object"]).columns
num_cols=X_train.select_dtypes(exclude=["str","object"]).columns
scaler=StandardScaler()
X_train_num=scaler.fit_transform(X_train[num_cols])
X_test_num=scaler.transform(X_test[num_cols])
encoder=OneHotEncoder(sparse_output=False,handle_unknown="ignore")
X_train_cat=encoder.fit_transform(X_train[cat_cols])
X_test_cat=encoder.transform(X_test[cat_cols])
X_train=np.hstack([X_train_cat,X_train_num])
X_test=np.hstack([X_test_cat,X_test_num])
X_train=torch.from_numpy(X_train).float().to(device)
X_test=torch.from_numpy(X_test).float().to(device)
y_train=torch.from_numpy(y_train.values).float().unsqueeze(1).to(device)
y_test=torch.from_numpy(y_test.values).float().unsqueeze(1).to(device)
train_dataset=TensorDataset(X_train,y_train)
train_load=DataLoader(train_dataset,batch_size=32,shuffle=True)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(X_train.shape[1],128)
        self.layer2=nn.Linear(128,32)
        self.layer3=nn.Linear(32,16)
        self.layer4=nn.Linear(16,32)
        self.layer5=nn.Linear(32,1)
    def forward(self,x):
        x=F.relu(self.layer1(x))
        x=F.relu(self.layer2(x))
        x=F.relu(self.layer3(x))
        x=F.relu(self.layer4(x))
        x=self.layer5(x)
        return x
model=NeuralNetwork().to(device)
criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)
epochs=200
for epoch in range(epochs):
    model.train()
    curr_loss=0.00
    for x_batch,y_batch in train_load:
        optimizer.zero_grad()
        predictions=model(x_batch)
        loss=criterion(predictions,y_batch)
        loss.backward()
        optimizer.step()
        curr_loss+=loss.item()
    print(f"epoch {epoch} MSE LOSS= {curr_loss/len(train_load)}")
with torch.no_grad():
    model.eval()
    predictions=model(X_test)
    loss=criterion(predictions,y_test)
    print(f"MSE loss= {loss.item()} Mean absolute error {mean_absolute_error(y_test.cpu().numpy(),predictions.cpu().numpy())} r2_score {r2_score(y_test.cpu().numpy(),predictions.cpu().numpy())}")
