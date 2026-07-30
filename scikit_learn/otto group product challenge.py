import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.ensemble import VotingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import log_loss
df=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\otto group train.csv")
df1=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\otto group test.csv")
print(df.head())
print(df.tail())
print(df.info())
print(df.columns)
print(df.describe())
for d in [df,df1]:
    d.columns=(d.columns
               .str.lower()
               .str.strip()
               .str.replace("_"," "))
num_cols=df.select_dtypes(exclude=["object","str"]).columns
num_cols=num_cols.drop("id")
for d_temp in [df,df1]:
    d_temp["sum"]=d_temp[num_cols].sum(axis=1)
    d_temp["mean"]=d_temp[num_cols].mean(axis=1)
    d_temp["median"]=d_temp[num_cols].median(axis=1)
    d_temp["std"]=d_temp[num_cols].std(axis=1)
for d in [df,df1]:
    d["unique value"]=(d[num_cols].nunique(axis=1)==1).astype(int)
    d["first quantile"]=d[num_cols].quantile(0.25,axis=1)
    d["third quantile"]=d[num_cols].quantile(0.75,axis=1)
    d["range"]=d[num_cols].max(axis=1)-d[num_cols].min(axis=1)
for d in [df,df1]:
    for col in num_cols:
        d[f"{col} ratio"]=d[col]/(d[num_cols].sum(axis=1)+ 1e-6)
id_extracted=df1["id"]
y_train=df["target"]
X_train=df.drop(columns=["id","target"])
X_test=df1.drop(columns="id")
scaler=QuantileTransformer()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
voting=VotingClassifier(
    estimators=[
        ("cat",CatBoostClassifier(random_state=42,auto_class_weights="Balanced",verbose=False))
        ,("lgbm",LGBMClassifier(random_state=42,class_weight="balanced",verbosity=-1))
        ,("xg",XGBClassifier(random_state=42))
    ]
    ,voting="soft"
)
cv_modified=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
Grid=GridSearchCV(
    estimator=voting
    ,param_grid={
        "cat__iterations":[300,350]
        ,"cat__learning_rate":[0.05,0.1]
        ,"lgbm__n_estimators":[350,400]
        ,"lgbm__learning_rate":[0.05]
        ,"xg__n_estimators":[300,350]
        ,"xg__learning_rate":[0.05,0.1]
    }
    ,cv=cv_modified
    ,scoring="neg_log_loss"
    ,n_jobs=-1
)
Grid.fit(X_train,y_train)
predictions=Grid.predict_proba(X_test)
print(Grid.best_params_)
print("best log loss score:",abs(Grid.best_score_))
submission=pd.DataFrame({
    "id":id_extracted
    ,"Class_1":predictions[:,0]
    ,"Class_2":predictions[:,1]
    ,"Class_3":predictions[:,2]
    ,"Class_4":predictions[:,3]
    ,"Class_5":predictions[:,4]
    ,"Class_6":predictions[:,5]
    ,"Class_7":predictions[:,6]
    ,"Class_8":predictions[:,7]
    ,"Class_9":predictions[:,8]
})
submission.to_csv("submission_otto group product challenge.csv",index=False)