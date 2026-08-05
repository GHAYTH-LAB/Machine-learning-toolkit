import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
from sklearn.ensemble import VotingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.metrics import f1_score,accuracy_score,precision_score
df=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\customer_churn.csv")
print(df.info())
print(df.shape)
print(df.columns)
print(df.isna().sum())
print(df.duplicated().sum())
df.columns=(df.columns
            .str.replace("_"," ")
            .str.strip())
df["customer id"]=df["customer id"].str.split("T",n=1).str[1]
print(df["customer id"])
df["partner"]=df["partner"].apply(lambda x:1 if (x=="Yes")else 0)
df["gender"]=(df["gender"]=="Male").astype(int)
df["paperless billing"]=(df["paperless billing"]=="Yes").astype(int)
y = df["churn"].map({"Yes": 1, "No": 0})
X = df.drop(columns=["churn", "customer id"])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
df=df.fillna({
    "total charges":df["total charges"].median()
    ,"satisfaction score":df["satisfaction score"].median()
})
cat_cols=X_train.select_dtypes(include=["str","object"]).columns
num_cols=X_train.select_dtypes(exclude=["str","object"]).columns
encoder=OneHotEncoder(sparse_output=False,handle_unknown="ignore")
X_train_cat=encoder.fit_transform(X_train[cat_cols])
X_test_cat=encoder.transform(X_test[cat_cols])
scaler=StandardScaler()
X_train_num=scaler.fit_transform(X_train[num_cols])
X_test_num=scaler.transform(X_test[num_cols])
X_train=np.hstack([X_train_cat,X_train_num])
X_test=np.hstack([X_test_cat,X_test_num])
voting=VotingClassifier(
    estimators=[
        ("cat",CatBoostClassifier(random_state=42))
        ,("lgbm",LGBMClassifier(random_state=42))
        ,("xg",XGBClassifier(random_state=42))
    ]
)
custom_cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
Grid=GridSearchCV(
    estimator=voting
    ,param_grid={
        "cat__iterations":[200,250]
        ,"cat__learning_rate":[0.1,0.3]
        ,"lgbm__n_estimators":[250,200]
        ,"lgbm__learning_rate":[0.1]
        ,"xg__n_estimators":[200,250]
        ,"xg__learning_rate":[0.3,0.1]
    }
    ,cv=custom_cv
    ,n_jobs=-1
    ,scoring="f1"
)
Grid.fit(X_train,y_train)
predictions=Grid.predict(X_test)
print(Grid.best_score_)
print(Grid.best_params_)
print(f1_score(y_test,predictions,average="weighted"))