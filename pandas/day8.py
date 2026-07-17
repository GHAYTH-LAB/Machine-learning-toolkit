import pandas as pd
df=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\Bengaluru_House_Data.csv")
df=df.drop_duplicates()
df.columns=df.columns.str.lower()
df.columns=df.columns.str.strip()
df.columns=df.columns.str.replace("_"," ")
print(df.columns)
df=df.fillna({
    "area type":df["area type"].mode()[0]
    ,"availability":df["availability"].mode()[0]
    ,"size":df["size"].mode()[0]
    ,"society":df["society"].mode()[0]
    ,"bath":df["bath"].median()
    ,"balcony":df["balcony"].median()
    ,"price":df["price"].median()
})
print(df.dtypes)
df=df.drop_duplicates()
df=df.dropna(subset="location")
df["area type"] = df["area type"].str.strip()
df["availability"] = df["availability"].str.strip()
df["location"] = df["location"].str.strip()
df["size"] = df["size"].str.strip()
df["society"] = df["society"].str.strip()
df["total sqft"] = df["total sqft"].str.strip()
df["area type"]=df["area type"].str.lower()
df["area type"]=df["area type"].str.replace("-"," ")
df["availability"] = df["availability"].str.lower()
df["availability"]=df["availability"].str.replace("-"," ")
df["location"]=df["location"].str.lower()
df["location"]=df["location"].str.replace("-"," ")
df["society"]=df["society"].str.lower()
df.to_csv("New_ FINAL_cleaned_dataset about bengaluru House Data.CSV",index=False)
print("-----------")