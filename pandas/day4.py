
import pandas as pd
df=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\results.csv")
print(df.to_string())
#select a row
print(df.loc[1])
#select a column
print(df["home_team"])
#Select rows
print(df.loc[[0,10,15]])
print("--------------")
print(df.loc[0:16])
#select by columns
print(df["home_team"])
print(df[["home_team","away_team","result"]].to_string())
#select specific rows+columns
print(df.loc[5:16,["home_team","away_team","result"]])
import pandas as pd;
df=pd.read_json(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\Dataset.json")
print(df)