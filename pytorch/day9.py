import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader
df=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\creatures_dataset.csv")
torch.manual_seed(42)
print(df.columns)
y=df["creature_type"]
X=df.drop(columns=["id","creature_type"])
class_names = y.unique()
class_to_index = {
    name: i for i, name in enumerate(class_names)
}
y = y.map(class_to_index)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
scaler=QuantileTransformer(n_quantiles=500)
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
X_train=torch.from_numpy(X_train).float()
X_test=torch.from_numpy(X_test).float()
y_train=torch.from_numpy(y_train.values).long()
y_test=torch.from_numpy(y_test.values).long()
train_dataset=TensorDataset(X_train,y_train)
data_loader=DataLoader(train_dataset,batch_size=16,shuffle=True)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(X_train.shape[1],64)
        self.layer2=nn.Linear(64,128)
        self.layer3=nn.Linear(128,32)
        self.layer4=nn.Linear(32,64)
        self.layer5=nn.Linear(64,32)
        self.layer6=nn.Linear(32,4)
    def forward(self,x):
        x=F.relu(self.layer1(x))
        x=F.relu(self.layer2(x))
        x=F.tanh(self.layer3(x))
        x=F.tanh(self.layer4(x))
        x=F.leaky_relu(self.layer5(x))
        x=self.layer6(x)
        return x
model=NeuralNetwork()
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)
epochs=150
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
    if (epoch%10==0):
        print(f"epoch {epoch}  average_loss= {curr_loss/len(data_loader)}")
with torch.no_grad():
    model.eval()
    predictions=model(X_test)
    predicted_class_position=torch.argmax(predictions,dim=1)
predicted_labels=np.array(class_names[predicted_class_position.numpy()])
loss=criterion(predictions,y_test)
print("loss",loss.item())
y_pred = predicted_class_position.numpy()
y_true = y_test.numpy()

print(f"accuracy= {accuracy_score(y_true,y_pred)} f1 score = {f1_score(y_true,y_pred,average='weighted')}")
prediction=pd.DataFrame({
    "id":range(1,len(predicted_labels)+1)
    ,"predictions":predicted_labels
})
prediction.to_csv("predictions.csv",index=False)