import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
from sklearn.ensemble import VotingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import quantile_transform,OneHotEncoder
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

for d in [df1,df]:
    d["max"]=d[num_cols].max(axis=1)
    d["min"]=d[num_cols].min(axis=1)
    d["mean"]=d[num_cols].mean(axis=1)
    d["median"]=d[num_cols].median(axis=1)
    d["std"]=d[num_cols].std(axis=1)
    d["range"]=d[num_cols].max(axis=1)-d[num_cols].min(axis=1)
    d["pos_max"]=np.argmax(d[num_cols].values,axis=1)
    d["min_pos"]=np.argmin(d[num_cols].values,axis=1)
    d["first_quantile"]=d[num_cols].quantile(0.25,axis=1)
    d["third_quantile"]=d[num_cols].quantile(0.75,axis=1)
for d in [df,df1]:
    for col in num_cols:
        d[f"{col} contribution"]=d[col]/(d[num_cols].sum(axis=1))
print(df.columns)