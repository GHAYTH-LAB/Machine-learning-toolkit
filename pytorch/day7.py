import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
df=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\train Forest Cover Type Prediction.csv")
df1=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\test Forest Cover Type Prediction.csv")
for d in [df,df1]:
    d.columns=(d.columns
               .str.lower()
               .str.replace("_"," ")
               .str.strip())
elevation_thereshold=df["elevation"].quantile(0.8)
horizontal_thereshold=df["horizontal distance to hydrology"].quantile(0.8)
vertical_thereshold=df["vertical distance to hydrology"].quantile(0.8)
fire_thereshold=df["horizontal distance to fire points"].quantile(0.8)
hillshade_9am_thereshold=df["hillshade 9am"].quantile(0.75)
hillshade_3pm_thereshold=df["hillshade 3pm"].quantile(0.75)
hillshade_noon_thereshold=df["hillshade noon"].quantile(0.75)
roadways_distance_thereshold=df["horizontal distance to roadways"].quantile(0.8)
for d in [df,df1]:
    d["high elevation"]=(d["elevation"]>elevation_thereshold).astype(int)
    d["forest is likely flat"]=(d["slope"]<=10).astype(int)
    d["forest flat medium"]=((d["slope"]>10) & (d["slope"]<=20)).astype(int)
    d["forest highly inclined"]=(d["slope"]>20).astype(int)
    d["The location is directly next to the nearest water feature"]=(((d["horizontal distance to hydrology"]==0) &(d["vertical distance to hydrology"]==0)).astype(int))
    d["The location is far from hydrology"]=((d["horizontal distance to hydrology"]>horizontal_thereshold) | (abs(d["vertical distance to hydrology"])>vertical_thereshold)).astype(int)
    d["far from roadways"]=(d["horizontal distance to roadways"]>roadways_distance_thereshold)
    d["strong sunlight"]=((d["hillshade 9am"]>=hillshade_9am_thereshold) & (d["hillshade 3pm"]>=hillshade_3pm_thereshold) & (d["hillshade noon"]>=hillshade_noon_thereshold)).astype(int)
    d["far from fire points"]=(d["horizontal distance to fire points"]>fire_thereshold).astype(int)
    d["elevation slope"] = d["elevation"] * d["slope"]
    d["hillshade mean"]=(d["hillshade 3pm"]+d["hillshade 9am"]+d["hillshade noon"])/3
    d["elevation minus_water"] = (d["elevation"] -d["vertical distance to hydrology"])
y_train=df["cover type"]
X_train=df.drop(columns=["cover type","id"])
X_train,X_test,y_train,y_test=train_test_split(X_train,y_train,test_size=0.2,random_state=42)
id_extracted=df1["id"]
scaler=QuantileTransformer()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
X_train=torch.from_numpy(X_train).float().to(device)
X_test=torch.from_numpy(X_test).float().to(device)
y_train=torch.from_numpy(y_train.values-1).long().to(device)
y_test=torch.from_numpy(y_test.values-1).long().to(device)
train_dataset=TensorDataset(X_train,y_train)
data_loader=DataLoader(train_dataset,batch_size=16,shuffle=True)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(X_train.shape[1],128)
        self.layer2=nn.Linear(128,256)
        self.layer3=nn.Linear(256,128)
        self.layer4=nn.Linear(128,64)
        self.layer5=nn.Linear(64,32)
        self.layer6=nn.Linear(32,16)
        self.layer7=nn.Linear(16,7)
    def forward(self,x):
        x=F.relu(self.layer1(x))
        x=F.relu(self.layer2(x))
        x=F.relu(self.layer3(x))
        x=F.relu(self.layer4(x))
        x=F.leaky_relu(self.layer5(x))
        x=F.relu(self.layer6(x))
        x=F.softmax(self.layer7(x))
        return x
model=NeuralNetwork().to(device)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.002)
epochs=300
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
    print(f"epoch {epoch+1} the loss is : {curr_loss/len(data_loader)}")
with torch.no_grad():
    model.eval()
    predictions=model(X_test)
    loss=criterion(predictions,y_test)
    predicted_classes = torch.argmax(predictions, dim=1)
    accuracy = (predicted_classes == y_test).float().mean()
    print("Loss:", loss.item())
    print("Accuracy:", accuracy.item())
#to add a kaggle submission
# Prepare the Kaggle test dataset
X_submission = df1.drop(columns=["id"])

# Use the scaler fitted on the training data
X_submission = scaler.transform(X_submission)

# Convert to tensor and move to the same device as the model
X_submission = torch.from_numpy(X_submission).float().to(device)

# Make predictions
model.eval()

with torch.no_grad():
    predictions = model(X_submission)

    # Get the class with the highest logit
    predicted_classes = torch.argmax(predictions, dim=1)

# We changed the original classes from 1-7 to 0-6
# so convert them back to 1-7
predicted_classes = predicted_classes + 1

# Move from GPU to CPU and convert to NumPy
predicted_classes = predicted_classes.cpu().numpy()

# Create Kaggle submission
submission = pd.DataFrame({
    "id": df1["id"],
    "cover type": predicted_classes
})

# Save the submission file
submission.to_csv("submission.csv", index=False)

print(submission.head())
print(submission.shape)