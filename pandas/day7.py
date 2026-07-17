import pandas as pd
df=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\Airbnb_Open_Data.csv")
print(df.duplicated().sum())
df=df.drop_duplicates()
df=df.drop(columns=['reviews per month', 'review rate number',
       'calculated host listings count', 'availability 365', 'house_rules',
       'license'])
print(df.info())
df.columns=df.columns.str.lower()
print(df.columns)
print(df.isna().sum())
df=df.drop(columns="last review")
print(df.columns)
df=df.dropna()
print(df.isna().sum())
df["instant_bookable"]=df["instant_bookable"].replace({
    True:1
    ,False:0
}) 
print(df["instant_bookable"].to_string())
df[["price","service fee"]]=df[["price","service fee"]].apply(lambda col:col.str.replace("$",""))
df[["price","service fee"]]=df[["price","service fee"]].apply(lambda col:col.str.replace(",",""))
print("\n ",df[["price","service fee"]])
print(df.columns)
df.to_csv("Clean Dataset.csv")