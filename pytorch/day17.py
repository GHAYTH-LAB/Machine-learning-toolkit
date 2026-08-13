import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader
from sklearn.preprocessing import QuantileTransformer,OneHotEncoder
from sklearn.metrics import accuracy_score,f1_score,recall_score
import math
df=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\train Costa Rican Household Poverty Level Prediction.csv")
df1=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\test Costa Rican Household Poverty Level Prediction.csv")
pd.set_option('display.max_info_columns', 300)
torch.manual_seed(41)
for d in[df,df1]:
    d.columns=d.columns.str.lower()
a,b,c,e,f=df["v2a1"].mode()[0],df["v18q1"].mode()[0],df["rez_esc"].mode()[0],df["sqbmeaned"].mode()[0],df["meaneduc"].median()
for d in[df,df1]:
    d.fillna({
        "v2a1":a
        ,"v18q1":b
        ,"rez_esc":c
        ,"sqbmeaned":e
        ,"meaneduc":f
},inplace=True
)
for d in [df,df1]:
    d["number of females> number of males"]=(d["r4m3"]>d["r4h3"]).astype(int)
    d["house price per size"]=d["v2a1"]/d["tamhog"]
    d["person part of rent"]=d["v2a1"]/d["tamviv"]
    d["adults>children"]=(d["hogar_nin"]<d["hogar_adul"]).astype(int)
    d["age"]=d["agesq"]**0.5
y_train=df["target"]
X_train=df.drop(columns=["target","id","idhogar"])
id_extracted=df1["id"]
X_test=df1.drop(columns=["id","idhogar"])
class_names = df["target"].unique()
class_to_index = {
    name: i for i, name in enumerate(class_names)
}
index_to_class = {
    i: name for name, i in class_to_index.items()
}
y_train = y_train.map(class_to_index)
num_cols=X_train.select_dtypes(exclude=["object","str"]).columns
cat_cols=X_train.select_dtypes(include=["object","str"]).columns
scaler=QuantileTransformer()
X_train_num=scaler.fit_transform(X_train[num_cols])
X_test_num=scaler.transform(X_test[num_cols])
encoder=OneHotEncoder(sparse_output=False,handle_unknown="ignore")
X_train_cat=encoder.fit_transform(X_train[cat_cols])
X_test_cat=encoder.transform(X_test[cat_cols])
X_train=np.hstack([X_train_num,X_train_cat])
X_test=np.hstack([X_test_num,X_test_cat])
X_train=torch.from_numpy(X_train).float()
X_test=torch.from_numpy(X_test).float()
y_train=torch.from_numpy(y_train.values).long()
train_dataset=TensorDataset(X_train,y_train)
data_loader=DataLoader(train_dataset,batch_size=16,shuffle=True)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(X_train.shape[1],64)
        self.layer2=nn.Linear(64,32)
        self.layer3=nn.Linear(32,16)
        self.layer4=nn.Linear(16,128)
        self.layer5=nn.Linear(128,32)
        self.layer6=nn.Linear(32,4)
    def forward(self,x):
        x=F.relu(self.layer1(x))
        x=F.relu(self.layer2(x))
        x=F.relu(self.layer3(x))
        x=F.relu(self.layer4(x))
        x=F.relu(self.layer5(x))
        x=self.layer6(x)
        return x
model=NeuralNetwork()
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)
epochs=400
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
    print(f"epoch = {epoch} loss= {curr_loss/len(data_loader)}")
with torch.no_grad():
    model.eval()
    predictions=model(X_test)
    class_predicted=torch.argmax(predictions,dim=1)
    predicted_labels = pd.Series(class_predicted.numpy()).map(index_to_class)
submission=pd.DataFrame({
        "Id":id_extracted
        ,"Target":predicted_labels
}
)
submission.to_csv("submission Costa Rican Household Poverty Level Prediction.csv",index=False)