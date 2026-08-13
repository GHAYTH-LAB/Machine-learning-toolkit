import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader
from sklearn.preprocessing import QuantileTransformer,OneHotEncoder
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
df=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\train Allstate Claims Severity.csv")
df1=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\test Allstate Claims Severity.csv")
y_train=df["loss"]
X_train=df.drop(columns=["loss","id"])
id_extracted=df1["id"]
X_test=df1.drop(columns="id")
cat_cols=X_train.select_dtypes(include=["str","object"]).columns
num_cols=X_train.select_dtypes(exclude=["str","object"]).columns
scaler=QuantileTransformer()
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
train_tensor=TensorDataset(X_train,y_train)
train_loader=DataLoader(train_tensor,batch_size=32,shuffle=True)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(X_train.shape[1],64)
        self.layer2=nn.Linear(64,32)
        self.layer3=nn.Linear(32,16)
        self.layer4=nn.Linear(16,64)
        self.layer5=nn.Linear(64,128)
        self.layer6=nn.Linear(128,32)
        self.layer7=nn.Linear(32,1)
    def forward(self,x):
        x=F.relu(self.layer1(x))
        x=F.relu(self.layer2(x))
        x=F.relu(self.layer3(x))
        x=F.relu(self.layer4(x))
        x=F.relu(self.layer5(x))
        x=F.relu(self.layer6(x))
        x=self.layer7(x)
        return x
model=NeuralNetwork().to(device)
optimizer=optim.Adam(model.parameters(),lr=0.001)
criterion=nn.MSELoss()
epochs=400
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
    print(f"epoch is {epoch} avg loss= {curr_loss/len(train_loader)}")
with torch.no_grad():
    model.eval()
    predictions=model(X_test)
submission=pd.DataFrame({
    "id":id_extracted
    ,"loss":predictions.cpu().numpy().flatten()
})
submission.to_csv("Kaggle submission Allstate Claims Severity.csv",index=False)
