import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import OneHotEncoder,QuantileTransformer
df=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\train Ghouls, Goblins, and Ghosts.csv")
df1=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\test Ghouls, Goblins, and Ghosts.csv")
#Data handling,cleaning and preprocessing
for d in [df,df1]:
    d.columns=(d.columns
               .str.strip()
               .str.lower()
               .str.replace("_"," "))
print(df.columns)
bone_length=df["bone length"].quantile(0.75)
rotting_length=df["rotting flesh"].quantile(0.75)
hair_length=df["hair length"].quantile(0.75)
has_soul=df["has soul"].quantile(0.75)
for d in [df,df1]:
    d["tall bone length"]=(d["bone length"]>bone_length).astype(int)
    d["much rotting flesh"]=(d["rotting flesh"]>rotting_length).astype(int)
    d["tall hair length"]=(d["hair length"]>hair_length).astype(int)
    d["has much soul"]=(d["has soul"]>has_soul).astype(int)
id_extracted=df1["id"]
y_train=df["type"]
X_train=df.drop(columns=["type","id"])
X_test=df1.drop(columns="id")
cat_cols=X_train.select_dtypes(include=["str","object"]).columns
num_cols=X_train.select_dtypes(exclude=["str","object"]).columns
scaler=QuantileTransformer(n_quantiles=371)
X_train_num=scaler.fit_transform(X_train[num_cols])
X_test_num=scaler.transform(X_test[num_cols])
encoder=OneHotEncoder(sparse_output=False,handle_unknown="ignore")
X_train_cat=encoder.fit_transform(X_train[cat_cols])
X_test_cat=encoder.transform(X_test[cat_cols])
X_train=np.hstack([X_train_cat,X_train_num])
X_test=np.hstack([X_test_cat,X_test_num])
custom_cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
voting=VotingClassifier(
    estimators=[
        ("cat",CatBoostClassifier(random_state=42))
        ,("xg",XGBClassifier(random_state=42))
        ,("lgbm",LGBMClassifier(random_state=42))
    ]
    ,voting="soft"
)
Grid=GridSearchCV(
    estimator=voting
    ,param_grid={
        "cat__iterations":[200,180]
        ,"cat__learning_rate":[0.03,0.05]
        ,"xg__n_estimators":[200,250]
        ,"xg__learning_rate":[0.03,0.05]
        ,"lgbm__n_estimators":[200,250]
        ,"lgbm__learning_rate":[0.03,0.05]
        ,"lgbm__num_leaves": [31,63]
    }
    ,scoring="accuracy"
    ,n_jobs=-1
    ,cv=custom_cv
)
Grid.fit(X_train,y_train)
predictions=Grid.predict(X_test)
print(Grid.best_score_)
print(Grid.best_params_)
submission=pd.DataFrame({
    "id":id_extracted
    ,"type":predictions
})
submission.to_csv("kaggle submission Ghost.csv",index=False)
