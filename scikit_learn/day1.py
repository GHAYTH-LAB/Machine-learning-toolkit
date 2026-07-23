#first steps into ml
"""from sklearn.datasets import load_diabetes
X,y= load_diabetes(return_X_y=True)
print(X,y)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
model=KNeighborsRegressor()
model.fit(X,y)
print(model.predict(X))
print("---------------")
model=LinearRegression()
model.fit(X,y)
print(model.predict(X))
"""
"""from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X,y=load_iris(return_X_y=True)
X_train,X_test,y_train,y_test=train_test_split(X,y
                                               ,random_state=42
                                               ,test_size=0.2)
model=LogisticRegression()
model.fit(X_train,y_train)
y_predict=model.predict(X_test)
accuracy=accuracy_score(y_test,y_predict)
print(accuracy)"""
"""from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
X,y=load_wine(return_X_y=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,shuffle=True)
model=RandomForestClassifier(n_estimators=100
                             ,max_depth=5
                             ,random_state=42)
model.fit(X_train,y_train)
predictions=model.predict(X_test)
accuracy=accuracy_score(y_test,predictions)
print(accuracy)"""
"""from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler
X,y=load_diabetes(return_X_y=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
model=RandomForestRegressor(random_state=42,
                            n_estimators=300,
                            max_depth=5,
                            min_samples_leaf=15,
                            min_samples_split=10)
model.fit(X_train,y_train)
predictions=model.predict(X_test)
print(mean_squared_error(y_test,predictions))
<<<<<<< HEAD
print(r2_score(y_test,predictions))"""
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.preprocessing import StandardScaler
X,y=load_digits(return_X_y=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
model=KNeighborsClassifier()
model.fit(X_train,y_train)
predictions=model.predict(X_test)
print("accuracy=",accuracy_score(y_test,predictions)
      ,"\n precision=",precision_score(y_test,predictions,average="weighted")
      ,"\n recall=",recall_score(y_test,predictions,average="weighted")
      ,"\n f1=",f1_score(y_test,predictions,average="weighted"))
print(r2_score(y_test,predictions))
