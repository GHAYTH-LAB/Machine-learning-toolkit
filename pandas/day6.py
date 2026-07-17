# Day 6: Data cleaning and preprocessing
import pandas as pd

# Load and inspect the migration dataset
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
df["uni"]=["insat","fst","fsb","insat","insat","isg","ihec","ihec","insat","fst","insat","fst","ihec","isg"]
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
print("------------------------")
print(df[(df["age"]>22) & (df["passed"]==1)])
print("---------------------------")
# Pure day 6 learning
# Load the housing sales dataset and remove an unnecessary column
df=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\quebec_housing_sales_v2.csv")
print(df.to_string())
df=df.drop(columns=["property_id"])
print("-----------------------------")

# Remove rows with missing values in specific columns
df=df.dropna(subset=["lot_size_sqft","renovation_year"])
print(df.count())

# Load the migration dataset and fill missing values with the most common entry
df=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\indian_overseas_migration_dataset.csv")
print(df.count())
df=df.fillna({
    "Source_URL":df["Source_URL"].mode()[0]
    ,"Source_Name":df["Source_Name"].mode()[0]
    ,"Notes":df["Notes"].mode()[0]
})
print("------------------")

# Standardize values and convert text to lowercase
df["Region"]=df["Region"].replace({
    "North America":"north america"
    ,"Europe":"europe"
    ,"Asia":"asia"
})
df["Column_Name"]=df["Column_Name"].str.lower()

# Remove duplicate rows from the student dataset
df=pd.read_json(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\students.json")
print(df)
df=df.drop_duplicates()
print("-------------------------")
print(df)