import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
from sklearn.ensemble import VotingClassifier,RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import OneHotEncoder,QuantileTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import auc
df=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\train Santander Customer Satisfaction.csv")
df1=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\test Santander Customer Satisfaction.csv")
#changing the standard params for displaying all the columns
pd.set_option('display.max_info_columns', 400)
#standarize the columns names
for d in [df,df1]:
    d.columns=(d.columns
                .str.lower()
                .str.strip()
                .str.replace("_"," "))
#delete the colomns with a single value(unnecessary)
cols_to_delete = []
for col in df.columns:
    if df[col].nunique() == 1:
        cols_to_delete.append(col)
df = df.drop(columns=cols_to_delete)
df1 = df1.drop(columns=cols_to_delete)
id_extracted=df1["id"]
y_train=df["target"]
X_train=df.drop(columns=["id","target"])
X_test=df1.drop(columns=["id"])
scaler=QuantileTransformer()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
voting=VotingClassifier(
    estimators=[
        ("cat",CatBoostClassifier(random_state=42,auto_class_weights="Balanced",verbose=0))
        ,("lgbm",LGBMClassifier(random_state=42,class_weight="balanced"))
        ,("rf",RandomForestClassifier(random_state=42,class_weight="balanced"))
    ]
    ,voting="soft"
)
cv_modified=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
Grid=GridSearchCV(
    estimator=voting
    ,param_grid={
        "rf__n_estimators":[300,350]
        ,"lgbm__n_estimators":[300,350]
        ,"lgbm__learning_rate":[0.1]
        ,"cat__iterations":[300,350]
        ,"cat__learning_rate":[0.05,0.1]
    }
    ,scoring="roc_auc"
    ,cv=cv_modified
    ,n_jobs=-1
)
Grid.fit(X_train,y_train)
predictions=Grid.predict_proba(X_test)
print(Grid.best_params_)
print("best score",Grid.best_score_)
submission=pd.DataFrame({
    "ID":id_extracted
    ,"TARGET":predictions[:,1]

}
)
#Downloading the submission file
submission.to_csv("submission_kaggle_santader_customer_satsifaction.csv",index=False)