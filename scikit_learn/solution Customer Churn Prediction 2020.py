#My solution for kaggle Customer Churn Prediction 2020
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.ensemble import VotingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder,QuantileTransformer
from sklearn.metrics import accuracy_score
df=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\train Customer Churn Prediction 2020.csv")
df1=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\test Customer Churn Prediction 2020.csv")
print(df.shape)
print(df.isna().sum())
print(df.duplicated().sum())
for d in [df,df1]:
    d.columns=(d.columns
               .str.lower()
               .str.strip()
               .str.replace("_"," "))
print(df.info())
print(df.columns)
print(df["state"].nunique())
x=df["account length"].median()
for d in [df,df1]:
    d["account length above median"]=(d["account length"]>x).astype(int)
    d["area code"]=(d["area code"].str.split("_",n=2).str[2]).astype(int)
    d.drop(columns="area code",inplace=True)
    d["international plan"]=(d["international plan"]
                             .str.lower()
                             .str.strip()
                             )
    d["international plan"]=d["international plan"].apply(lambda x:0 if x=="no" else 1)
    d["international plan"]=d["international plan"].astype(int)
    d["voice mail plan"]=d["voice mail plan"].apply(lambda x:0 if x=="no" else 1)
    d["voice mail plan"]=d["voice mail plan"].astype(int)
    d["without voice mail"]=(d["number vmail messages"]==0).astype(int)
    d["did not complain"]=(d["number customer service calls"]==0).astype(int)
    d["day charge per minute"]=np.where(d["total day minutes"]==0,0,d["total day charge"]/d["total day minutes"])
    d["evening charge per minute"]=np.where(d["total eve minutes"]==0,0,d["total eve charge"]/d["total eve minutes"])
    d["night charge per minute"]=np.where(d["total night minutes"]==0,0,d["total night charge"]/d["total night minutes"])
    d["international charge per minute"]=np.where(d["total intl minutes"]==0,0,d["total intl charge"]/d["total intl minutes"])
    d["ratio minute price"]=(d["day charge per minute"]+d["evening charge per minute"]+d["night charge per minute"]+d["international charge per minute"])/4
    d["total charge"]=d["total intl charge"]+d["total night charge"]+d["total eve charge"]+d["total day charge"]
    d["total calls"]=d["total day calls"]+d["total eve calls"]+d["total night calls"]+d["total intl calls"]
    d["day call duration"]=np.where(d["total day calls"]==0,0,d["total day minutes"]/d["total day calls"])
    d["evening call duration"]=np.where(d["total eve calls"]==0,0,d["total eve minutes"]/d["total eve calls"])
    d["night call duration"]=np.where(d["total night calls"]==0,0,d["total night minutes"]/d["total night calls"])
    d["intl call duration"]=np.where(d["total intl calls"]==0,0,d["total intl minutes"]/d["total intl calls"])
    d["intl plan low usage"] = ((d["international plan"]==1) & (d["total intl calls"]<=2)).astype(int)
    d["intl plan x charge"] = d["international plan"] * d["total intl charge"]
print(df["international plan"])
print(df["voice mail plan"])
y_train=df["churn"]
X_train=df.drop(columns="churn")
X_test=df1
num_cols=X_train.select_dtypes(exclude=["str","object"]).columns
cat_cols=X_train.select_dtypes(include=["object","str"]).columns
scaler=QuantileTransformer(n_quantiles=200)
X_train_num=scaler.fit_transform(X_train[num_cols])
X_test_num=scaler.transform(X_test[num_cols])
encoder=OneHotEncoder(sparse_output=False,handle_unknown="ignore")
X_train_cat=encoder.fit_transform(X_train[cat_cols])
X_test_cat=encoder.transform(X_test[cat_cols])
X_train=np.hstack([X_train_num,X_train_cat])
X_test=np.hstack([X_test_num,X_test_cat])
voting=VotingClassifier(
    estimators=[
        ("cat",CatBoostClassifier(random_state=42))
        ,("xg",XGBClassifier(random_state=42))
        ,("lgbm",LGBMClassifier(random_state=42))
        ,("knn",KNeighborsClassifier())
    ]
    ,voting="hard"
)
custome_cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
Grid=GridSearchCV(
    estimator=voting
    ,param_grid={
        "cat__iterations":[250,300]
        ,"cat__learning_rate":[0.05,0.1]
        ,"cat__eval_metric":["Accuracy"]
        ,"xg__n_estimators":[250,300]
        ,"xg__learning_rate":[0.05,0.1]
        ,"lgbm__n_estimators":[250,300]
        ,"lgbm__num_leaves":[15,31]
        ,"knn__n_neighbors":[5,3]
    }
    ,scoring="accuracy"
    ,cv=custome_cv
    ,n_jobs=-1
)
Grid.fit(X_train,y_train)
predictions=Grid.predict(X_test)
print("Grid best params",Grid.best_params_)
print("best accuracy score",Grid.best_score_)
submission=pd.DataFrame({
    "id":range(1,len(predictions)+1)
    ,"churn":predictions
})
submission.to_csv("kaggle submission Customer Churn Prediction 2020.csv",index=False)