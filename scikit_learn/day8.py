#"Give me some credit" kaggle competition
#Importing the necessary libraries 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import VotingClassifier,RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import OneHotEncoder,QuantileTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import auc,precision_score,accuracy_score,f1_score
#Data preprocessing
df=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\cs-training.csv")
df1=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\cs-test.csv")
df.columns=df.columns.str.lower()
df=df.drop(columns="unnamed: 0")
df.columns=(df.columns
            .str.lower()
            .str.replace("-"," ")
            .str.strip())
#Remvoing outliers
df=df[df["age"]<100]
df=df[df["revolvingutilizationofunsecuredlines"]<=1]
monthly_income_imputer=df["monthlyincome"].median()
numberofdependents_income_imputer=df["numberofdependents"].mode()[0]
df=df.fillna({
    "monthlyincome":monthly_income_imputer
    ,"numberofdependents":numberofdependents_income_imputer
})
df["living alone"]=(df["numberofdependents"]==0).astype(int)
df["not working"]=((df["age"]>=60) & (df["monthlyincome"]==0)).astype(int)
df["number of total loans"]=df["numberrealestateloansorlines"]+df["numberofopencreditlinesandloans"]
df["did not get any loan"]=(df["number of total loans"]==0).astype(int)
df["revolvingutilizationofunsecuredlines squared"]=df["revolvingutilizationofunsecuredlines"]*df["revolvingutilizationofunsecuredlines"]
#..
df1.columns=(df1.columns
             .str.lower()
            .str.replace("-"," ")
            .str.strip()
)
df1=df1.drop(columns="unnamed: 0")
df1=df1.fillna({
    "monthlyincome":monthly_income_imputer
    ,"numberofdependents":numberofdependents_income_imputer
})
df1["living alone"]=(df1["numberofdependents"]==0).astype(int)
df1["not working"]=((df1["age"]>=60) & (df1["monthlyincome"]==0)).astype(int)
df1["number of total loans"]=df1["numberrealestateloansorlines"]+df1["numberofopencreditlinesandloans"]
df1["did not get any loan"]=(df1["number of total loans"]==0).astype(int)
df1["revolvingutilizationofunsecuredlines squared"]=df1["revolvingutilizationofunsecuredlines"]*df1["revolvingutilizationofunsecuredlines"]
print(df.columns)
print(df1.columns)
y_train=df["seriousdlqin2yrs"]
X_train=df.drop(columns="seriousdlqin2yrs")
X_test=df1.drop(columns="seriousdlqin2yrs")
scaler=QuantileTransformer()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
voting=VotingClassifier(
    estimators=[
        ("cat",CatBoostClassifier(random_state=42,auto_class_weights="Balanced"))
        ,("Lgbm",LGBMClassifier(random_state=42,class_weight="balanced"))
        ,("rf",RandomForestClassifier(random_state=42,class_weight="balanced"))
    ]
    ,voting="soft"
)
Grid=GridSearchCV(
    estimator=voting
    ,param_grid={
    "cat__n_estimators":[300,500],
    "cat__learning_rate":[0.05],
    "cat__depth":[6,8],
    "Lgbm__n_estimators":[300],
    "Lgbm__num_leaves":[31,63],
    "rf__n_estimators":[300],
    "rf__max_depth":[10]
    }
    ,scoring="roc_auc"
    ,cv=5
    ,n_jobs=-1
)
Grid.fit(X_train,y_train)
predictions = Grid.predict_proba(X_test)[:,1]
print("Best fit for the model is :",Grid.best_params_)
submission=pd.DataFrame({
    "Id":range(1, len(predictions) + 1)
    ,"Probability":predictions
}
)
submission.to_csv("first_submission_kaggle_give_me_credit.csv",index=False)