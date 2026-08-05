import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,QuantileTransformer
from sklearn.metrics import f1_score,accuracy_score
df=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\loan_approval_binary.csv")
torch.manual_seed(41)
df.columns=df.columns.str.lower()
y=df["loanapproved"]
X=df.drop(columns=["applicantid","loanapproved"])
class_names=y.unique()
class_to_index={
    name:i for i,name in enumerate(class_names)
}
y=y.map(class_to_index)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
num_cols=X_train.select_dtypes(exclude=["str","object"]).columns
cat_cols=X_train.select_dtypes(include=["str","object"]).columns
scaler=QuantileTransformer()
X_train_num=scaler.fit_transform(X_train[num_cols])
X_test_num=scaler.transform(X_test[num_cols])
encoder=OneHotEncoder(sparse_output=False,handle_unknown="ignore")
X_train_cat=encoder.fit_transform(X_train[cat_cols])
X_test_cat=encoder.transform(X_test[cat_cols])
X_train=np.hstack([X_train_num,X_train_cat])
X_test=np.hstack([X_test_num,X_test_cat])
X_train=torch.from_numpy(X_train).float().to(device)
X_test=torch.from_numpy(X_test).float().to(device)
y_train=torch.from_numpy(y_train.values).float().unsqueeze(1).to(device)
y_test=torch.from_numpy(y_test.values).float().unsqueeze(1).to(device)
train_dataset=TensorDataset(X_train,y_train)
data_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(X_train.shape[1],64)
        self.layer2=nn.Linear(64,16)
        self.layer3=nn.Linear(16,32)
        self.layer4=nn.Linear(32,16)
        self.layer5=nn.Linear(16,1)
    def forward(self,x):
        x=F.relu(self.layer1(x))
        x=F.relu(self.layer2(x))
        x=F.relu(self.layer3(x))
        x=F.relu(self.layer4(x))
        x=F.sigmoid(self.layer5(x))
        return x
model=NeuralNetwork().to(device)
criterion=nn.BCELoss()
optimizer=optim.Adam(model.parameters(),lr=0.002)
epochs=200
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
    print(f"epoch {epoch+1} loss= {curr_loss/len(data_loader)}")
with torch.no_grad():
    model.eval()
    predictions=model(X_test)
    loss=criterion(predictions,y_test)
    predicted_labels=(predictions>=0.5).float()
    print(f"accuracy= {accuracy_score(y_test.cpu().numpy(),predicted_labels.cpu().numpy())} f1 score={f1_score(y_test.cpu().numpy(),predicted_labels.cpu().numpy(),average="weighted")} loss= {loss.item()} ")
