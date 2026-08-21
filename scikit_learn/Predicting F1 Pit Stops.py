import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import VotingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
df=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\train Predicting F1 Pit Stops.csv")
df1=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\test Predicting F1 Pit Stops.csv")
for d in [df,df1]:
    d.columns=(d.columns
               .str.lower()
               .str.replace("_","",regex=False)
               .str.strip()
               .str.replace(" ","",regex=False))
for d in [df,df1]:
    d[["drivername","drivernum"]]=d["driver"].str.extract(r"([A-Za-z]+)(\d+)?")
    d["drivernum"]=pd.to_numeric(d["drivernum"],errors="coerce")
    d["laptimedelta_negative"]=(d["laptimedelta"]<0).astype(int)
    d["race_place"]=d["race"].str.split(" ",n=1).str[0]
    d["race_place"]=d["race_place"].replace("Pre-Season","unknown")
    d["grandprix"]=(d["race"]!="Pre-Season Testing").astype(int)
    d["ratiolapnumberstint"]=d["lapnumber"]/d["stint"]
    d["ratiotyrelifelapnumber"]=d["tyrelife"]/d["lapnumber"]
    d["ratiotyrelifestint"]=d["tyrelife"]/d["stint"]
    d["position_stint"]=d["position"]*d["stint"]
    d.drop(columns="driver",inplace=True)
for d in [df,df1]:
    d.replace([np.inf,-np.inf],np.nan,inplace=True)
    numeric_cols=d.select_dtypes(include=[np.number]).columns
    d[numeric_cols]=d[numeric_cols].fillna(d[numeric_cols].median())
    categorical_cols=d.select_dtypes(include=["object","string"]).columns
    d[categorical_cols]=d[categorical_cols].fillna("unknown")
id_extracted=df1["id"]
y_train=df["pitnextlap"]
X_train=df.drop(columns=["pitnextlap","id"])
X_test=df1.drop(columns=["id"])
cat_cols=X_train.select_dtypes(include=["object","string"]).columns
num_cols=X_train.select_dtypes(exclude=["object","string"]).columns
X_train_num=X_train[num_cols].to_numpy(dtype=np.float32)
X_test_num=X_test[num_cols].to_numpy(dtype=np.float32)
encoder=OneHotEncoder(sparse_output=False,handle_unknown="ignore",dtype=np.float32)
X_train_cat=encoder.fit_transform(X_train[cat_cols])
X_test_cat=encoder.transform(X_test[cat_cols])
X_train=np.hstack([X_train_cat,X_train_num]).astype(np.float32)
X_test=np.hstack([X_test_cat,X_test_num]).astype(np.float32)
print("X_train shape:",X_train.shape)
print("X_train dtype:",X_train.dtype)
print("X_train memory:",round(X_train.nbytes/1024**3,2),"GB")
voting=VotingClassifier(
    estimators=[
        ("xgb",XGBClassifier(random_state=42,tree_method="hist")),
        ("lgbm",LGBMClassifier(random_state=42,verbosity=-1)),
        ("cat",CatBoostClassifier(random_state=42,verbose=0))
    ],
   voting="soft"
)
custom_cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
Grid=GridSearchCV(
    estimator=voting,
    param_grid={
        "xgb__n_estimators":[300,350],
        "xgb__learning_rate":[0.05,0.1],
        "lgbm__n_estimators":[300,350],
        "cat__iterations":[300,350],
        "cat__learning_rate":[0.1]
    },
    n_jobs=1,
    cv=custom_cv,
    scoring="roc_auc",
    verbose=2
)
Grid.fit(X_train,y_train)
predictions=Grid.predict_proba(X_test)[:,1]
print("\nBest parameters:")
print(Grid.best_params_)
print("\nBest CV ROC-AUC:",Grid.best_score_)
submission=pd.DataFrame({
    "id":id_extracted,
    "PitNextLap":predictions
})
submission.to_csv("submission_Predicting_F1_Pit_Stops.csv",index=False)