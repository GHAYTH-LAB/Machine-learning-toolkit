import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader
from sklearn.metrics import roc_auc_score,f1_score
from sklearn.preprocessing import QuantileTransformer,OneHotEncoder
torch.manual_seed(41)
df=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\train Predicting Smartphone Addiction.csv")
df1=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\test Predicting Smartphone Addiction.csv")
print(df.isna().sum())
print(df.info())
print(df.shape)
for d in [df,df1]:
    d.columns=(d.columns
               .str.lower()
               .str.replace("_"," "))
age_filler=df["age"].median()
daily_screen_time_hours_filler=df["daily screen time hours"].median()
social_media_hours_filler=df["social media hours"].median()
gaming_hours_filler=df["gaming hours"].median()
work_study_hours_filler=df["work study hours"].median()
sleep_hours_filler=df["sleep hours"].median()
notifications_per_day_filler=df["notifications per day"].median()
app_opens_per_day_filler=df["app opens per day"].median()
weekend_screen_time_filler=df["weekend screen time"].median()
gender_filler=df["gender"].mode()[0]
stress_level_filler=df["stress level"].mode()[0]
academic_work_impact_filler=df["academic work impact"].mode()[0]
for d in [df,df1]:
    d.fillna({
        "age":age_filler
        ,"daily screen time hours":daily_screen_time_hours_filler
        ,"social media hours":social_media_hours_filler
        ,"gaming hours":gaming_hours_filler
        ,"work study hours":work_study_hours_filler
        ,"sleep hours":sleep_hours_filler
        ,"notifications per day":notifications_per_day_filler
        ,"app opens per day":app_opens_per_day_filler
        ,"weekend screen time":weekend_screen_time_filler
        ,"gender":gender_filler
        ,"stress level":stress_level_filler
        ,"academic work impact":academic_work_impact_filler
    },inplace=True)
for d in [df,df1]:
    d["daily screen time minutes"]=d["daily screen time hours"]*60
    d["social media minutes"]=d["social media hours"]*60
    d["gaming minutes"]=d["gaming hours"]*60
    d["work study minutes"]=d["work study hours"]*60
    d["sleep in minutes"]=d["sleep hours"]*60
    d["daily screen times pourcentage"]=d["daily screen time hours"]/24
    d["social media pourcentage"]=d["social media hours"]/24
    d["sleep hours pourcentage"]=d["sleep hours"]/24
    d["sleep hours not too much"]=(d["sleep hours"]<6).astype(int)
    d["ratio social media"]=d["social media hours"]/d["daily screen time hours"]
    d["ratio gaming hours"]=d["gaming hours"]/d["daily screen time hours"]
    d["work study pourcentage"]=d["work study hours"]/24
    d["worker"]=d["daily screen time hours"]<d["work study hours"]
    d["weekend/normal days screen time"]=d["weekend screen time"]/d["daily screen time hours"]
    d["apps opened per hour"]=d["app opens per day"]/d["daily screen time hours"]
id_extracted=df1["id"]
y_train=df["addicted label"]
X_train=df.drop(columns=["id","addicted label"])
X_test=df1.drop(columns=["id"])
num_cols=X_train.select_dtypes(exclude=["object","str"]).columns
cat_cols=X_train.select_dtypes(include=["str","object"]).columns
scaler=QuantileTransformer(n_quantiles=100)
X_train_num=scaler.fit_transform(X_train[num_cols])
X_test_num=scaler.transform(X_test[num_cols])
encoder=OneHotEncoder(sparse_output=False,handle_unknown="ignore")
X_train_cat=encoder.fit_transform(X_train[cat_cols])
X_test_cat=encoder.transform(X_test[cat_cols])
X_train=np.hstack([X_train_num,X_train_cat])
X_test=np.hstack([X_test_num,X_test_cat])
X_train=torch.from_numpy(X_train).float()
X_test=torch.from_numpy(X_test).float()
y_train=torch.from_numpy(y_train.values).float().unsqueeze(1)
train_dataset=TensorDataset(X_train,y_train)
train_loader=DataLoader(train_dataset,batch_size=16,shuffle=True)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(X_train.shape[1],128)
        self.layer2=nn.Linear(128,32)
        self.layer3=nn.Linear(32,64)
        self.layer4=nn.Linear(64,16)
        self.layer5=nn.Linear(16,32)
        self.layer6=nn.Linear(32,1)
    def forward(self,x):
        x=F.relu(self.layer1(x))
        x=F.relu(self.layer2(x))
        x=F.relu(self.layer3(x))
        x=F.relu(self.layer4(x))
        x=F.relu(self.layer5(x))
        x=F.sigmoid(self.layer6(x))
        return x
model=NeuralNetwork()
criterion=nn.BCELoss()
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
        curr_loss+=loss.item()
        optimizer.step()
    print(f"{epoch+1} loss= {curr_loss/len(train_loader)}")
with torch.no_grad():
    model.eval()
    predictions=model(X_test)
submission=pd.DataFrame({
    "id":id_extracted
    ,"addicted_label":predictions.squeeze().numpy()
})
submission.to_csv("submission Predicting Smartphone Addiction.csv",index=False)
