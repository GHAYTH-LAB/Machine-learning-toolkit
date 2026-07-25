import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder,QuantileTransformer,LabelEncoder
from sklearn.metrics import f1_score,accuracy_score,precision_score
df=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\train loan approval classification.csv")
df1=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\test data loan approval classification.csv")
df.columns=(df.columns
    .str.lower()
    .str.strip()
)
df=df[df["person_age"]<100]
df["person started working"]=df["person_age"]-df["person_emp_exp"]
df["pourcentage"]=df["loan_percent_income"]*df["loan_int_rate"]
#test
df1.columns=(df1.columns
    .str.lower()
    .str.strip()
)
df1["person started working"]=df1["person_age"]-df1["person_emp_exp"]
df1["pourcentage"]=df1["loan_percent_income"]*df1["loan_int_rate"]
id_extracted=df1["id"]
df1=df1.drop(columns="id")
y_train=df["loan_status"]
X_train=df.drop(columns=["id","loan_status"])
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
voting=VotingClassifier(
    estimators=[
        ("rf",RandomForestClassifier(random_state=42))
        ,("knn",KNeighborsClassifier())
        ,("xg",XGBClassifier(random_state=42))
    ]
)
Grid=GridSearchCV(
    estimator=voting
    ,param_grid={
        "rf__n_estimators":[200,250,300]
        ,"rf__max_depth":[None,5,7]
        ,"rf__min_samples_leaf":[5,10]
        ,"rf__min_samples_split":[5,10,15]
        ,"knn__n_neighbors":[5,7]
        ,"xg__n_estimators":[200,250,300]
        ,"xg__learning_rate":[0.1,0.05]
        ,"voting":["soft","hard"]
    }
    ,cv=5
    ,n_jobs=-1
)
Grid.fit(X_train,y_train)
predictions=Grid.predict(X_test)
print(Grid.best_params_)
submission=pd.DataFrame({
    "ID":id_extracted
    ,"loan_status":predictions
})
submission.to_csv("submission.csv",index=False)