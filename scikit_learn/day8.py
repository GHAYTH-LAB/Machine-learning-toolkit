import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder,QuantileTransformer
from sklearn.metrics import accuracy_score,f1_score,recall_score
df=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\train assurance.csv")
df1=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\test assurance.csv")
print(df.info())
print(df.duplicated().sum())
print(df.isna().sum())