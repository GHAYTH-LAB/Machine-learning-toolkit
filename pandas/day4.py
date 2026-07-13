import pandas as pd
#How to read a  csv file using pandas
df=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\quebec_housing_sales_v2.csv")
print(df.to_string())
print("--------------------")
#How to read a json file using pandas 
df=pd.read_json(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\students.json")
print(df)