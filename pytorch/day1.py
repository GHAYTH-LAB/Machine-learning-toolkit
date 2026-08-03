import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader
from sklearn.preprocessing import OneHotEncoder,QuantileTransformer
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
df=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\train Ghouls, Goblins, and Ghosts.csv")
df1=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\test Ghouls, Goblins, and Ghosts.csv")
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
#Data handling,cleaning and preprocessing
for d in [df,df1]:
    d.columns=(d.columns
               .str.strip()
               .str.lower()
               .str.replace("_"," "))
print(df.columns)
bone_length=df["bone length"].quantile(0.75)
rotting_length=df["rotting flesh"].quantile(0.75)
hair_length=df["hair length"].quantile(0.75)
has_soul=df["has soul"].quantile(0.75)
for d in [df,df1]:
    d["tall bone length"]=(d["bone length"]>bone_length).astype(int)
    d["much rotting flesh"]=(d["rotting flesh"]>rotting_length).astype(int)
    d["tall hair length"]=(d["hair length"]>hair_length).astype(int)
    d["has much soul"]=(d["has soul"]>has_soul).astype(int)
    d["bone length*hairlength"]=d["bone length"]*d["hair length"]
    d["rotting flesh*has_soul"]=d["rotting flesh"]*d["has soul"]
id_extracted=df1["id"]
class_names = sorted(df["type"].unique())
label_to_index = {
    label: i for i, label in enumerate(class_names)
}
y_train = df["type"].map(label_to_index)
X_train=df.drop(columns=["type","id"])
X_test=df1.drop(columns="id")
cat_cols=X_train.select_dtypes(include=["str","object"]).columns
num_cols=X_train.select_dtypes(exclude=["str","object"]).columns
scaler=QuantileTransformer(n_quantiles=X_train.shape[0])
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
train_loader=DataLoader(train_dataset,shuffle=True,batch_size=32)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(X_train.shape[1],256)
        self.layer2=nn.Linear(256,128)
        self.layer3=nn.Linear(128,64)
        self.layer4=nn.Linear(64,16)
        self.layer5=nn.Linear(16,3)
    def forward(self,x):
        x=F.relu(self.layer1(x))
        x=F.relu(self.layer2(x))
        x=F.relu(self.layer3(x))
        x=F.relu(self.layer4(x))
        x=self.layer5(x)
        return x
model=NeuralNetwork()
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)
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
    print(f"epoch= {epoch+1} loss: {curr_loss/len(train_loader)} ")
with torch.no_grad():
    model.eval()
    predictions=model(X_test)
    predicted_classes_position=torch.argmax(predictions,dim=1)
class_names = sorted(df["type"].unique())
predicted_labels=np.array(class_names)[predicted_classes_position]
submission=pd.DataFrame({
    "id":id_extracted
    ,"type":predicted_labels
})
submission.to_csv("NN_Ghost_submission.csv", index=False)