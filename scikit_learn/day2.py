#in this day I tried my first kaggle problem(Tbh it was not that easy hhhh but i will keep grinding till I improve )
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import QuantileTransformer,OneHotEncoder
from sklearn.metrics import accuracy_score,recall_score,f1_score
df=pd.read_csv(r"C:\Users\abidli\kaggle\train.csv")
df1=pd.read_csv(r"C:\Users\abidli\kaggle\test.csv")
test_ids=df1["PassengerId"]
df.columns=(df.columns
            .str.strip()
            .str.lower()
            )
df=df.fillna({
    "homeplanet":df["homeplanet"].mode()[0]
    ,"cryosleep":df["cryosleep"].mode()[0]
})
df["cryosleep"]=df["cryosleep"].astype(int)
df["destination"]=(df["destination"]
                    .str.lower()
                    .str.strip()
                    .str.replace(" ",""))
df["destination type"]=df["destination"].str.split("-",n=1).str[0]
df["destination coordinates"]=df["destination"].str.split("-",n=1).str[1]
df=df.drop(columns="destination")
df=df.fillna({
        "destination type":df["destination type"].mode()[0]
        ,"destination coordinates":df["destination coordinates"].mode()[0]
        ,"vip":df["vip"].mode()[0]
        ,"age":df["age"].median()
})
df["age"]=df["age"].astype(int)
df["vip"]=df["vip"].astype(int)
df["deck"]=df["cabin"].str.split("/",n=2).str[0]
df["num"]=df["cabin"].str.split("/",n=2).str[1]
df["side"]=df["cabin"].str.split("/",n=2).str[2]
df=df.fillna({
        "deck":df["deck"].mode()[0]
        ,"num":df["num"].mode()[0]
        ,"side":df["side"].mode()[0]
        ,"foodcourt":df["foodcourt"].median()
        ,"shoppingmall":df["shoppingmall"].median()
        ,"spa":df["spa"].median()
        ,"vrdeck":df["vrdeck"].median()
        ,"roomservice":df["roomservice"].median()
})
df["side is s"]=df["side"]=="S"
df["side is p"]=df["side"]=="P"
side_mode = df["side"].mode()[0]
df=df.drop(columns=["side","cabin"])
df[["side is s","side is p"]]=df[["side is s","side is p"]].astype(int)
df["total payed"]=df["roomservice"]+df["foodcourt"]+df["shoppingmall"]+df["spa"]+df["vrdeck"]
df["name"] = df["name"].fillna("Unknown Unknown")
df["first name"]=df["name"].str.split(" ",n=1).str[0]
df["second name"]=df["name"].str.split(" ",n=1).str[1]
df[["destination type","destination coordinates","deck","first name","second name"]]=df[["destination type","destination coordinates","deck","first name","second name"]].astype(str)
df["num"]=df["num"].astype(int)
df=df.drop(columns="passengerid")
#********
df1.columns=(df1.columns
            .str.strip()
            .str.lower()
            )
df1=df1.fillna({
    "homeplanet":df["homeplanet"].mode()[0]
    ,"cryosleep":df["cryosleep"].mode()[0]
})
df1["cryosleep"]=df1["cryosleep"].astype(int)
df1["destination"]=(df1["destination"]
                    .str.lower()
                    .str.strip()
                    .str.replace(" ",""))
df1["destination type"]=df1["destination"].str.split("-",n=1).str[0]
df1["destination coordinates"]=df1["destination"].str.split("-",n=1).str[1]
df1=df1.drop(columns="destination")
df1=df1.fillna({
        "destination type":df["destination type"].mode()[0]
        ,"destination coordinates":df["destination coordinates"].mode()[0]
        ,"vip":df["vip"].mode()[0]
        ,"age":df["age"].median()
})
df1["age"]=df1["age"].astype(int)
df1["vip"]=df1["vip"].astype(int)
df1["deck"]=df1["cabin"].str.split("/",n=2).str[0]
df1["num"]=df1["cabin"].str.split("/",n=2).str[1]
df1["side"]=df1["cabin"].str.split("/",n=2).str[2]
df1=df1.fillna({
        "deck":df["deck"].mode()[0]
        ,"num":df["num"].mode()[0]
        ,"side":side_mode
        ,"foodcourt":df["foodcourt"].median()
        ,"shoppingmall":df["shoppingmall"].median()
        ,"spa":df["spa"].median()
        ,"vrdeck":df["vrdeck"].median()
        ,"roomservice":df["roomservice"].median()
})
df1["side is s"]=df1["side"]=="S"
df1["side is p"]=df1["side"]=="P"
df1=df1.drop(columns=["side","cabin"])
df1[["side is s","side is p"]]=df1[["side is s","side is p"]].astype(int)
df1["total payed"]=df1["roomservice"]+df1["foodcourt"]+df1["shoppingmall"]+df1["spa"]+df1["vrdeck"]
df1["name"] = df1["name"].fillna("Unknown Unknown")
df1["first name"]=df1["name"].str.split(" ",n=1).str[0]
df1["second name"]=df1["name"].str.split(" ",n=1).str[1]
df1[["destination type","destination coordinates","deck","first name","second name"]]=df1[["destination type","destination coordinates","deck","first name","second name"]].astype(str)
df1["num"]=df1["num"].astype(int)
df1=df1.drop(columns="passengerid")
y_train=df["transported"]
X_train=df.drop(columns=["transported"])
X_test=df1
categorical_columns=X_train.select_dtypes(include=["object","string"]).columns
numerical_columns=X_train.select_dtypes(exclude=["object","string"]).columns
encoder=OneHotEncoder(handle_unknown="ignore",sparse_output=False)
X_train_cat=encoder.fit_transform(X_train[categorical_columns])
X_test_cat=encoder.transform(X_test[categorical_columns])
scaler=QuantileTransformer()
X_train_num=scaler.fit_transform(X_train[numerical_columns])
X_test_num=scaler.transform(X_test[numerical_columns])
X_train=np.hstack([X_train_cat,X_train_num])
X_test=np.hstack([X_test_cat,X_test_num])
Voting=VotingClassifier(
    estimators=[
        ("rf",RandomForestClassifier(random_state=42))
        ,("log",LogisticRegression(solver="liblinear"))
        ,("knn",KNeighborsClassifier())
    ]
)
Grid=GridSearchCV(
    estimator=Voting
    ,param_grid={
    "rf__n_estimators":[100,200],
    "rf__max_depth":[None,10],
    "rf__min_samples_leaf":[5,10],
    "log__C":[0.1,1,10],
    "knn__n_neighbors":[3,5,7],
    "weights":[
        [1,1,1],
        [2,1,1],
        [1,2,1]
    ],
    "voting":["soft"]
}
    ,cv=5
    ,scoring="accuracy"
    ,n_jobs=-1
)

Grid.fit(X_train,y_train)
print(Grid.best_score_)
print(Grid.best_params_)
predictions=Grid.predict(X_test)
submission=pd.DataFrame(
    {
        "PassengerId":test_ids
        ,"Transported":predictions.astype(bool)
    }
)
submission.to_csv("submission.csv",index=False)