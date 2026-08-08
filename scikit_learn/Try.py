import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV,StratifiedKFold,train_test_split
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import OneHotEncoder,QuantileTransformer
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import accuracy_score,f1_score,mean_absolute_error,mean_squared_error,r2_score
df=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\Data.csv")
print(df.shape)
print(df.duplicated().sum())
print(df.isna().sum())
df.columns=(df.columns
            .str.lower()
            .str.strip()
            .str.replace("_"," ")
            )
print(df.info())
df=df.fillna({
    "club":df["club"].mode()[0]
})
print(df["league"].nunique())
df["average minutes per match"]=df["mins"]/df["matches played"]
df["substitutions per match"]=df["substitution"]/df["matches played"]
df["goals per match"]=df["goals"]/df["matches played"]
df["goals per minute"]=df["goals"]/df["mins"]
df["scores more than expected"]=df["goals"]>df["xg"]
df["shots per minute"]=df["shots"]/df["mins"]
df["shots to score"]=df["shots"]/df["goals"]
df["ratio of shots on target"]=df["ontarget"]/df["shots"]
y=df["xg per avg match"]
X=df.drop(columns="xg per avg match")
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
num_cols=X_train.select_dtypes(exclude=["str","object"]).columns
cat_cols=X_train.select_dtypes(include=["str","object"]).columns
scaler=QuantileTransformer()
X_train_num=scaler.fit_transform(X_train[num_cols])
X_test_num=scaler.transform(X_test[num_cols])
encoder=OneHotEncoder(sparse_output=False,handle_unknown="ignore")
X_train_cat=encoder.fit_transform(X_train[cat_cols])
X_test_cat=encoder.transform(X_test[cat_cols])
X_train=np.hstack([X_train_cat,X_train_num])
X_test=np.hstack([X_test_cat,X_test_num])
voting=VotingRegressor(
    estimators=[
        ("cat",CatBoostRegressor(random_state=42))
        ,("xg",XGBRegressor(random_state=42))
        ,("lgbm",LGBMRegressor(random_state=42))
    ]
)
Grid=GridSearchCV(
    estimator=voting
    ,param_grid={
        "cat__iterations":[150,200]
        ,"cat__learning_rate":[0.1,0.3]
        ,"lgbm__n_estimators":[150,100]
        ,"lgbm__learning_rate":[0.1,0.3]
        ,"xg__n_estimators":[200,150]
        ,"xg__learning_rate":[0.1,0.03]
    }
    ,cv=5
    ,n_jobs=-1
    ,scoring="neg_mean_absolute_error"
)
Grid.fit(X_train,y_train)
predictions=Grid.predict(X_test)
print(Grid.best_params_)
print(f"{abs(Grid.best_score_)} r2_scor= {r2_score(y_test,predictions)}")
