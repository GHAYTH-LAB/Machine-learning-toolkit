import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import OneHotEncoder,QuantileTransformer
from sklearn.metrics import accuracy_score
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
id_extracted=df1["id"]
X_test=df1.drop(columns="id")
scaler=QuantileTransformer()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
My_cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
voting=VotingClassifier(
    estimators=[
        ("cat",CatBoostClassifier(random_state=42))
        ,("lgbm",LGBMClassifier(random_state=42))
        ,("knn",KNeighborsClassifier())
    ]
    ,voting="soft"
)
Grid=GridSearchCV(
    estimator=voting
    ,param_grid={
        "cat__iterations":[350,400]
        ,"cat__learning_rate":[0.01,0.03]
        ,"lgbm__n_estimators":[450,400]
        ,"lgbm__learning_rate":[0.01,0.03]
        ,"knn__n_neighbors":[5]

    }
    ,scoring="accuracy"
    ,cv=My_cv
    ,n_jobs=-1
)
Grid.fit(X_train,y_train)
predictions=Grid.predict(X_test)
print(Grid.best_score_)
print(Grid.best_params_)
submission=pd.DataFrame({
    "Id":id_extracted
    ,"Cover_Type":predictions
})
submission.to_csv("initial submission kaggle Forest Cover Type Prediction.csv",index=False)

