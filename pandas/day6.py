#Revision of day 5 
import pandas as pd
df=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\indian_overseas_migration_dataset.csv")
print(df.to_string())
print("-------------------------")
print(df.loc[0])
print("-------------------------")
print(df.iloc[2])
print("-------------------------")
print(df.loc[[2,3,5]])
print("-------------------------")
print(df.loc[8:12])
print("-------------------------")
print(df.head())
print("--------------------------")
print(df.tail())
print("--------------------------")
print(df["Destination_Country"])
print("--------------------------")
print(df[["Destination_Country","Value","Source_URL"]])
print("---------------------------")
print(df.loc[40:45, ["Destination_Country", "Region", "Year"]])
print("---------------------------")
print(df.mean(numeric_only=True))
print("----------------------------")
print(df.count())
print("----------------------------")
print(df.min(numeric_only=True))
print("---------------------------")
print(df.max(numeric_only=True))
print("----------------------------")
#For a specific column
df=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\quebec_housing_sales_v2.csv")
print(df["bedrooms"].median())
print(df["bedrooms"].sum())
print(df["bedrooms"].min())
print(df["bedrooms"].max())
print(df["bedrooms"].mode())
print(df["bedrooms"].count())
print("--------------------------")
print(df["bedrooms"].describe())
print("--------------------------")
#groupby
New_groups=df.groupby("city")
print(New_groups.sum())
print("---------------------------")
print(New_groups["living_area_sqft"].median())
print("-----------------------------")
print(New_groups.count())
print("----------------------------")
print(New_groups["living_area_sqft"].mean())
df=pd.read_json(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\students.json")
df["uni"]=["insat","fst","fsb","insat","insat","isg","ihec","ihec","insat","fst"]
New_row={
    "id":[11,12]
    ,"name":["Ala","Rayen"]
    ,"age":[20,18]
    ,"city":["Mehdia","KEF"]
    ,"score":[80,85]
    ,"passed":[1,0]
}
New_row=pd.DataFrame(New_row)
df=pd.concat([df,New_row])
print(df.tail())
