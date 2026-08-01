#Implemented my first Pytorch Neural Network(achieved accuracy =0.9736841917037964)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X,y=load_breast_cancer(return_X_y=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
X_train_scaled=torch.from_numpy(X_train_scaled).float()
X_test_scaled=torch.from_numpy(X_test_scaled).float()
y_train=torch.from_numpy(y_train).float().unsqueeze(1)
y_test=torch.from_numpy(y_test).float().unsqueeze(1)
train_dataset=TensorDataset(X_train_scaled,y_train)
train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
class neuralnetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(30,64)
        self.fc2=nn.Linear(64,32)
        self.fc3=nn.Linear(32,16)
        self.fc4=nn.Linear(16,1)
    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.leaky_relu(self.fc3(x))
        x=F.sigmoid(self.fc4(x))
        return x
model=neuralnetwork()
criterion=nn.BCELoss()
optimizer=optim.Adam(model.parameters(),lr=0.01)
epochs=20
for epoch in range(epochs):
    model.train()
    running_loss=0.0
    for x_batch,y_batch in train_loader:
        optimizer.zero_grad()
        predictions=model(x_batch)
        loss=criterion(predictions,y_batch)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
    print(f"epoch{epoch+1}:loss was {running_loss/len(train_loader)}")
with torch.no_grad():
    model.eval()
    preds = model(X_test_scaled)
    loss = criterion(preds, y_test).item()
    accuracy = ((preds >= 0.5) == y_test).float().mean().item()
    print("accuracy=",accuracy)
