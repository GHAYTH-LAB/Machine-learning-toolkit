import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
from sklearn.ensemble import VotingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import QuantileTransformer,OneHotEncoder
from sklearn.decomposition import PCA
from scipy.sparse import hstack
from sklearn.metrics import log_loss
#Load Data
df=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\train BNP Paribas Cardif Claims Management.csv")
df1=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\test BNP Paribas Cardif Claims Management.csv")
pd.set_option('display.max_info_columns', 500)
for d in [df,df1]:
    d.columns=(d.columns
               .str.lower()
               .str.strip()
               )
print(df.info())
print(df.isna().sum().to_string())
num_cols=[]
cat_cols=[]
for col in df.columns:
    if ((col=="id") or (col=="target")):
        continue
    elif (df[col].dtype=="object" or df[col].dtype=="str"):
        cat_cols.append(col)
    else:
        num_cols.append(col)
for d in [df,df1]:
    d["nan_count"]=d[num_cols+cat_cols].isna().sum(axis=1)
    d["nan_ratio"]=d[num_cols+cat_cols].isna().sum(axis=1)/len(num_cols+cat_cols)
for col in num_cols:
    num_imputer=df[col].median()
    if df[col].isna().any():
        df[col]=df[col].fillna(num_imputer)
    if df1[col].isna().any():
        df1[col]=df1[col].fillna(num_imputer)
for col in cat_cols:
    cat_imputer=df[col].mode()[0]
    if df[col].isna().any():
        df[col]=df[col].fillna(cat_imputer)
    if df1[col].isna().any():
        df1[col]=df1[col].fillna(cat_imputer)
def add_row_stats(d, num_cols):
    stats = pd.DataFrame({
        "max": d[num_cols].max(axis=1),
        "min": d[num_cols].min(axis=1),
        "mean": d[num_cols].mean(axis=1),
        "median": d[num_cols].median(axis=1),
        "std": d[num_cols].std(axis=1),
        "range": d[num_cols].max(axis=1) - d[num_cols].min(axis=1),
        "pos_max": np.argmax(d[num_cols].values, axis=1),
        "pos_min": np.argmin(d[num_cols].values, axis=1),
        "first_quantile": d[num_cols].quantile(0.25, axis=1),
        "third_quantile": d[num_cols].quantile(0.75, axis=1),
    }, index=d.index)
    return pd.concat([d, stats], axis=1)
df = add_row_stats(df, num_cols)
df1 = add_row_stats(df1, num_cols)
engineered_cols=["max","min","mean","median","std","range","pos_max","pos_min","first_quantile","third_quantile","nan_count","nan_ratio"]
for x in engineered_cols:
    num_cols.append(x)
y_train=df["target"]
X_train=df.drop(columns=["id","target"])
id_extracted=df1["id"]
X_test=df1.drop(columns="id")
scaler=QuantileTransformer()
X_train_num=scaler.fit_transform(X_train[num_cols])
X_test_num=scaler.transform(X_test[num_cols])
encoder=OneHotEncoder(handle_unknown="ignore")
X_train_cat=encoder.fit_transform(X_train[cat_cols])
X_test_cat=encoder.transform(X_test[cat_cols])
X_train=hstack([X_train_num,X_train_cat])
X_test=hstack([X_test_num,X_test_cat])
custom_cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
cat_Grid=GridSearchCV(
    estimator=CatBoostClassifier(random_state=42,verbose=0)
    ,param_grid={
        "iterations":[350,400]
        ,"learning_rate":[0.1,0.05]
        ,"depth":[6,8]
        ,"l2_leaf_reg":[1,3,5]
    }
    ,cv=custom_cv
    ,n_jobs=-1
    ,scoring="neg_log_loss"
)
cat_Grid.fit(X_train,y_train)
xgb_Grid=GridSearchCV(
    estimator=XGBClassifier(random_state=42)
    ,param_grid={
       "n_estimators":[400,450]
       ,"learning_rate":[0.05,0.1]
       ,"max_depth":[6,8,7]
    }
    ,cv=custom_cv
    ,n_jobs=-1
    ,scoring="neg_log_loss"
)
xgb_Grid.fit(X_train,y_train)
lgbm_Grid=GridSearchCV(
    estimator=LGBMClassifier(random_state=42,verbose=-1)
    ,param_grid={
        "learning_rate":[0.05,0.1]
        ,"n_estimators":[350,400]
        ,"max_depth":[4,6,8]
        ,"num_leaves":[15, 31, 63]
    }
    ,cv=custom_cv
    ,n_jobs=-1
    ,scoring="neg_log_loss"
)
lgbm_Grid.fit(X_train,y_train)
voting=VotingClassifier(
    estimators=[
        ("cat",cat_Grid.best_estimator_)
        ,("lgbm",lgbm_Grid.best_estimator_)
        ,("xg",xgb_Grid.best_estimator_)
    ]
    ,voting="soft"
)
voting.fit(X_train,y_train)
predictions=voting.predict_proba(X_test)[:,1]
print("CatBoost best params:", cat_Grid.best_params_)
print("CatBoost best score:", -cat_Grid.best_score_)
print("XGBoost best params:", xgb_Grid.best_params_)
print("XGBoost best score:", -xgb_Grid.best_score_)
print("LightGBM best params:", lgbm_Grid.best_params_)
print("LightGBM best score:", -lgbm_Grid.best_score_)
submission=pd.DataFrame({
    "ID":id_extracted
    ,"PredictedProb":predictions
})
submission.to_csv("submission_Kaggle_BNP_Paribas_Cardif_Claims_Management.csv",index=False)
