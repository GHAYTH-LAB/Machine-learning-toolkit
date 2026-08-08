#kaggle Urban Mobility Prediction competition(condition: only Linear models)
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.linear_model import Lasso,ElasticNet,Ridge
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import accuracy_score,root_mean_squared_log_error,f1_score,make_scorer
df=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\train Urban Mobility Prediction.csv")
df1=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\test Urban Mobility Prediction.csv")
for d in [df,df1]:
    d.columns=(d.columns
               .str.lower()
               .str.replace("_"," ")
               .str.strip())
print(df.info())
print("------------")
print(df1.info())
for d in [df,df1]:
    d["feature humidity"] = d["feature humidity"].fillna(
        df["feature humidity"].median()
    )
    d["feature wind"] = d["feature wind"].fillna(
        df["feature wind"].median()
    )
for d in [df,df1]:
    d["season spring"]=(d["season code"]==1).astype(int)
    d["season summer"]=(d["season code"]==2).astype(int)
    d["season fall"]=(d["season code"]==3).astype(int)
    d["season winter"]=(d["season code"]==4).astype(int)
    d["clear weather"]=(d["category weather"]==1).astype(int)
    d["mist weather"]=(d["category weather"]==2).astype(int)
    d["light snow weather"]=(d["category weather"]==3).astype(int)
    d["heavy rain weather"]=(d["category weather"]==4).astype(int)
    d["absolute difference in temperature"]=abs(d["feature feel temp"]-d["feature temp"])
    d["morning"] = ((d["hour"] >= 6) & (d["hour"] < 12)).astype(int)
    d["afternoon"] = ((d["hour"] >= 12) & (d["hour"] < 18)).astype(int)
    d["evening"] = ((d["hour"] >= 18) & (d["hour"] < 22)).astype(int)
    d["night"] = ((d["hour"] < 6) | (d["hour"] >= 22)).astype(int)
    d["bike on weekend"]=((d["day of week"]==5) | (d["day of week"]==6)).astype(int)
    d["year = 2011"]=(d["year"]==2011).astype(int)
    d["year = 2012"]=(d["year"]==2012).astype(int)
    d["humidity * wind"]=d["feature humidity"]*d["feature wind"]
    d["hour sin"] = np.sin(2 * np.pi * d["hour"] / 24)
    d["hour cos"] = np.cos(2 * np.pi * d["hour"] / 24)
    d=d.drop(columns=["season winter", "clear weather"])

id_dropped=df1["id"]
y_train=df["demand"]
X_train=df.drop(columns=["id","demand"])
X_test=df1.drop(columns="id")
scaler=QuantileTransformer()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
def safe_rmsle(y_true, y_pred):
    y_pred = np.maximum(y_pred, 0)
    return root_mean_squared_log_error(y_true, y_pred)
rmsle_scorer = make_scorer(safe_rmsle, greater_is_better=False)
voting=VotingRegressor(
    estimators=[
        ("lasso",Lasso(max_iter=10000))
        ,("ridge",Ridge())
        ,("elastic",ElasticNet(max_iter=10000))
    ]
)
Grid=GridSearchCV(
    estimator=voting
    ,param_grid={
        "lasso__alpha":[1.0,0.1]
        ,"lasso__fit_intercept":[True]
        ,"ridge__alpha":[0.1,1]
        ,"ridge__fit_intercept":[True]
        ,"ridge__max_iter":[None]
        ,"elastic__alpha":[1.0,0.1]
        ,"elastic__fit_intercept":[True]
    }
    ,scoring=rmsle_scorer
    ,cv=5
    ,n_jobs=-1
)
y_train_log = np.log1p(y_train)
Grid.fit(X_train, y_train_log)
predictions = np.expm1(Grid.predict(X_test))
predictions = np.maximum(predictions, 0)
print(Grid.best_params_)
print(Grid.best_score_)
submission=pd.DataFrame(
    {
    "id":id_dropped
    ,"demand":predictions
    }
)
submission.to_csv("submission_Urban_Mobility_Prediction.csv",index=False)