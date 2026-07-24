import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestRegressor,VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,QuantileTransformer
from sklearn.metrics import root_mean_squared_error

train=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\train house.csv")
test=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\test house.csv")
df=train
#cleaning the train dataset and preprocessing it
df.columns=df.columns.str.lower()
columns_one_hot_encoder=["mszoning","street","lotshape","landcontour","utilities","lotconfig","landslope","neighborhood","condition1","condition2","bldgtype","housestyle","roofstyle","exterior1st","exterior2nd","exterqual","extercond","foundation","bsmtqual","bsmtcond","bsmtexposure","bsmtfintype1","bsmtfintype2","heating","heatingqc","electrical","kitchenqual","functional","fireplacesqu","garagetype","garagefinish","garagequal","garagecond","paveddrive","fence"," miscfeature"]

def mode_or_none(x):
    m = x.mode()
    return m[0] if not m.empty else "None"

df["alley"]        = df.groupby("street")["alley"].transform(lambda x: x.fillna(mode_or_none(x)))
df["masvnrtype"]   = df.groupby(["exterior1st","exterior2nd"])["masvnrtype"].transform(lambda x: x.fillna(mode_or_none(x)))
df["masvnrarea"]   = df.groupby("masvnrtype")["masvnrarea"].transform(lambda x: x.fillna(x.median()))
df["bsmtqual"]     = df.groupby("foundation")["bsmtqual"].transform(lambda x: x.fillna(mode_or_none(x)))
df["bsmtcond"]     = df.groupby("foundation")["bsmtcond"].transform(lambda x: x.fillna(mode_or_none(x)))
df["bsmtexposure"] = df.groupby(["bsmtcond","bsmtqual"])["bsmtexposure"].transform(lambda x: x.fillna(mode_or_none(x)))
df["bsmtfintype1"] = df.groupby(["bsmtcond","bsmtqual"])["bsmtfintype1"].transform(lambda x: x.fillna(mode_or_none(x)))
df["bsmtfintype2"] = df.groupby(["bsmtcond","bsmtqual"])["bsmtfintype2"].transform(lambda x: x.fillna(mode_or_none(x)))
df["fireplacequ"]  = df.groupby(["fireplaces","functional"])["fireplacequ"].transform(lambda x: x.fillna(mode_or_none(x)))
df["garagefinish"] = df.groupby(["garagetype","garageyrblt"])["garagefinish"].transform(lambda x: x.fillna(mode_or_none(x)))
df["garagequal"]   = df.groupby("garagetype")["garagequal"].transform(lambda x: x.fillna(mode_or_none(x)))
df["garagecond"]   = df.groupby("garagetype")["garagecond"].transform(lambda x: x.fillna(mode_or_none(x)))
df["fence"]        = df.groupby("housestyle")["fence"].transform(lambda x: x.fillna(mode_or_none(x)))
df["miscfeature"]  = df["miscfeature"].fillna("unknown")

df["centralair"]=df["centralair"].apply(lambda x:1 if x=="Y" else 0)
df["centralair"]=df["centralair"].astype(int)
df=df.drop(columns="poolqc")
df1=test
id=df1["Id"]
df=df.drop(columns="id")
df1=df1.drop(columns="Id")
#cleaning and preprocessing the test dataset
df1.columns=df1.columns.str.lower()
columns_one_hot_encoder=["mszoning","street","lotshape","landcontour","utilities","lotconfig","landslope","neighborhood","condition1","condition2","bldgtype","housestyle","roofstyle","exterior1st","exterior2nd","exterqual","extercond","foundation","bsmtqual","bsmtcond","bsmtexposure","bsmtfintype1","bsmtfintype2","heating","heatingqc","electrical","kitchenqual","functional","fireplacesqu","garagetype","garagefinish","garagequal","garagecond","paveddrive","fence"," miscfeature"]

def apply_group_fill(target_df, group_cols, col, train_group_map, fallback):
    idx = target_df.set_index(group_cols).index
    filled = pd.Series(idx.map(train_group_map), index=target_df.index)
    return target_df[col].fillna(filled).fillna(fallback)

alley_map        = df.groupby("street")["alley"].agg(mode_or_none)
masvnrtype_map   = df.groupby(["exterior1st","exterior2nd"])["masvnrtype"].agg(mode_or_none)
masvnrarea_map   = df.groupby("masvnrtype")["masvnrarea"].median()
bsmtqual_map     = df.groupby("foundation")["bsmtqual"].agg(mode_or_none)
bsmtcond_map     = df.groupby("foundation")["bsmtcond"].agg(mode_or_none)
bsmtexposure_map = df.groupby(["bsmtcond","bsmtqual"])["bsmtexposure"].agg(mode_or_none)
bsmtfintype1_map = df.groupby(["bsmtcond","bsmtqual"])["bsmtfintype1"].agg(mode_or_none)
bsmtfintype2_map = df.groupby(["bsmtcond","bsmtqual"])["bsmtfintype2"].agg(mode_or_none)
fireplacequ_map  = df.groupby(["fireplaces","functional"])["fireplacequ"].agg(mode_or_none)
garagefinish_map = df.groupby(["garagetype","garageyrblt"])["garagefinish"].agg(mode_or_none)
garagequal_map   = df.groupby("garagetype")["garagequal"].agg(mode_or_none)
garagecond_map   = df.groupby("garagetype")["garagecond"].agg(mode_or_none)
fence_map        = df.groupby("housestyle")["fence"].agg(mode_or_none)

df1["alley"]        = apply_group_fill(df1, ["street"], "alley", alley_map, df["alley"].mode()[0])
df1["masvnrtype"]   = apply_group_fill(df1, ["exterior1st","exterior2nd"], "masvnrtype", masvnrtype_map, df["masvnrtype"].mode()[0])
df1["masvnrarea"]   = apply_group_fill(df1, ["masvnrtype"], "masvnrarea", masvnrarea_map, df["masvnrarea"].median())
df1["bsmtqual"]     = apply_group_fill(df1, ["foundation"], "bsmtqual", bsmtqual_map, df["bsmtqual"].mode()[0])
df1["bsmtcond"]     = apply_group_fill(df1, ["foundation"], "bsmtcond", bsmtcond_map, df["bsmtcond"].mode()[0])
df1["bsmtexposure"] = apply_group_fill(df1, ["bsmtcond","bsmtqual"], "bsmtexposure", bsmtexposure_map, df["bsmtexposure"].mode()[0])
df1["bsmtfintype1"] = apply_group_fill(df1, ["bsmtcond","bsmtqual"], "bsmtfintype1", bsmtfintype1_map, df["bsmtfintype1"].mode()[0])
df1["bsmtfintype2"] = apply_group_fill(df1, ["bsmtcond","bsmtqual"], "bsmtfintype2", bsmtfintype2_map, df["bsmtfintype2"].mode()[0])
df1["fireplacequ"]  = apply_group_fill(df1, ["fireplaces","functional"], "fireplacequ", fireplacequ_map, df["fireplacequ"].mode()[0])
df1["garagefinish"] = apply_group_fill(df1, ["garagetype","garageyrblt"], "garagefinish", garagefinish_map, df["garagefinish"].mode()[0])
df1["garagequal"]   = apply_group_fill(df1, ["garagetype"], "garagequal", garagequal_map, df["garagequal"].mode()[0])
df1["garagecond"]   = apply_group_fill(df1, ["garagetype"], "garagecond", garagecond_map, df["garagecond"].mode()[0])
df1["fence"]        = apply_group_fill(df1, ["housestyle"], "fence", fence_map, df["fence"].mode()[0])
df1["miscfeature"]  = df1["miscfeature"].fillna("unknown")

df1["centralair"]=df1["centralair"].apply(lambda x:1 if x=="Y" else 0)
df1["centralair"]=df1["centralair"].astype(int)
df1=df1.drop(columns="poolqc")
y_train=df["saleprice"]
X_train=df.drop(columns="saleprice")
X_test=df1
categorical_columns=X_train.select_dtypes(include=["object","str"]).columns
numerical_columns=X_train.select_dtypes(exclude=["object","str"]).columns
encoder=OneHotEncoder(sparse_output=False,handle_unknown="ignore")
X_train_cat=encoder.fit_transform(X_train[categorical_columns])
X_test_cat=encoder.transform(X_test[categorical_columns])
scaler=QuantileTransformer()
X_train_num=scaler.fit_transform(X_train[numerical_columns])
X_test_num=scaler.transform(X_test[numerical_columns])
X_train=np.hstack([X_train_cat,X_train_num])
X_test=np.hstack([X_test_cat,X_test_num])
voting=VotingRegressor(
    estimators=[
        ("rf",RandomForestRegressor(random_state=42))
        ,("knn",KNeighborsRegressor())
        ,("xg",XGBRegressor())
    ]
)
Grid=GridSearchCV(
    estimator=voting
    ,param_grid={
        "rf__n_estimators":[200,300,350]
        ,"rf__max_depth":[None,5,6,7]
        ,"rf__min_samples_split":[5,10,15]
        ,"rf__min_samples_leaf":[5,7,10]
        ,"knn__n_neighbors":[5,7]
        ,"xg__n_estimators":[150,250,300]
        ,"xg__learning_rate":[0.01,0.1]
        ,"weights":[
            [1,1,1]
            ,[2,1,1]
            ,[1,2,1]
            ,[1,1,2]
        ]

    }
    ,n_jobs=-1
)
Grid.fit(X_train,y_train)
predictions=Grid.predict(X_test)
submission=pd.DataFrame({
    "Id":id
    ,"SalePrice":predictions
})
submission.to_csv("submission.csv",index=False)