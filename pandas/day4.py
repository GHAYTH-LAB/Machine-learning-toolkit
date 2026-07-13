import pandas as pd
#How to read a  csv file using pandas
df=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\quebec_housing_sales_v2.csv")
print(df.to_string())
print("--------------------")
#How to read a json file using pandas 
df1=pd.read_json(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\students.json")
print(df1)
#selection BY column
print(df1["name"])
print("---------------------")
#Selection by column and printing all the cities
print(df["city"].to_string())
#select by multiple columns
print(df[["bedrooms","bathrooms","garage"]].to_string())
#select per row
print(df.loc[2])
print("---------------------")
print(df1.iloc[2])
print("----------------------")
import pandas as pd
Data=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\assets\WorldCupMatches.csv",index_col="City")
print(Data)
print("----------------------")
print(Data.loc["Sao Paulo "])
print("----------------------")
print(Data["Stadium"])
print("-----------------------")
print(Data[["Home Team Name","Away Team Name","Year"]])