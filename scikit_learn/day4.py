import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder,QuantileTransformer,LabelEncoder
from sklearn.metrics import accuracy_score,recall_score,f1_score
# data handling and cleaning
df=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\train_titanic_classification.csv")
df1=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\test_titanic_classification.csv")
#Data cleaning
df.columns=(df.columns
            .str.lower()
            .str.strip())
extracted_id=df1["PassengerId"]
df["familySize"] = df["sibsp"] + df["parch"] + 1
df["deck"]=df["cabin"].str[0]
df["title"] = df["name"].str.extract(r',\s*([A-Za-z]+)\.')
age_fill=df.groupby(["sex"])["age"].median()
deck_fill=df["deck"].mode()[0]
embarked_fill=df["embarked"].mode()[0]
df=df.fillna({
    "age":df["sex"].map(age_fill)
    ,"deck":deck_fill
    ,"embarked":embarked_fill
    ,
})
df["ticket_number"] = df["ticket"].str.extract(r'(\d+)')
df=df.drop(columns=["cabin","name","ticket"])
df=df.dropna(subset=["title","ticket_number"])
df["ticket_number"]=df["ticket_number"].astype(int)
#data cleaning for test dataset
df1.columns=(df1.columns
            .str.lower()
            .str.strip())
df1["familySize"] = df1["sibsp"] + df1["parch"] + 1
df1["deck"]=df1["cabin"].str[0]
df1["title"] = df1["name"].str.extract(r',\s*([A-Za-z]+)\.')
df1=df1.fillna({
    "age":df1["sex"].map(age_fill)
    ,"deck":deck_fill
    ,"embarked":embarked_fill
})
df1["ticket_number"] = df1["ticket"].str.extract(r'(\d+)')
df1=df1.drop(columns=["cabin","name","ticket"])
df1["ticket_number"]=df1["ticket_number"].astype(int)
y_train=df["survived"]
X_train=df.drop(columns=["survived","passengerid"])
X_test=df1.drop(columns="passengerid")
columns_cat=X_train.select_dtypes(include=["object","str"]).columns
columns_num=X_train.select_dtypes(exclude=["object","str"]).columns
encoder=OneHotEncoder(sparse_output=False,handle_unknown="ignore")
X_train_cat=encoder.fit_transform(X_train[columns_cat])
X_test_cat=encoder.transform(X_test[columns_cat])
scaler=QuantileTransformer(n_quantiles=500)
X_train_num=scaler.fit_transform(X_train[columns_num])
X_test_num=scaler.transform(X_test[columns_num])
X_train=np.hstack([X_train_cat,X_train_num])
X_test=np.hstack([X_test_cat,X_test_num])
grid=GridSearchCV(
    estimator=RandomForestClassifier(random_state=42)
    ,param_grid={
        "n_estimators":[150,200,250,300]
        ,"max_depth":[None,5,7,10,15]
        ,"min_samples_leaf":[5,10,15]
        ,"min_samples_split":[10,15]   
    }
    ,cv=5
    ,n_jobs=-1
)
grid.fit(X_train,y_train)
print(grid.best_params_)
predictions=grid.predict(X_test)
submission=pd.DataFrame({
    "PassengerId":extracted_id
    ,"Survived":predictions
})
submission.to_csv("submission.csv",index=False)