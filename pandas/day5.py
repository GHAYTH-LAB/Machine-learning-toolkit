#Revision of day 4
import pandas as pd
df=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\genai_llm_usage_dataset_1000.csv")
print(df)
print(df.loc[1])
print("-------------------")
print(df.iloc[2])
print("--------------------")
print(df["model_name"])
print("--------------------")
print(df.head())
print("---------------------")
print(df.head(8))
print("---------------------")
print(df.tail(10))
print(df.loc[[5,7,10]])
print("----------------------")
print(df.loc[15:20])
print("------------------------")
print(df[["model_name","successful_response","prompt_length"]])
print("------------------------")
print(df.loc[2:15,["model_name","task_type","latency_sec"]])
print("------------------------")
#filtering based on a conditon
# Using the and operator &
print(df[(df["successful_response"] == 1) & (df["application_domain"] == "Coding")])
print("--------------------------")
#using the or operator (|)
print(df[(df["application_domain"]=="coding") | (df["application_domain"]=="Education")])
#How to summerize and analyse data
df=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\weight-height.csv")
#How to get the min value?
print(df.min(numeric_only=True))
#How to get the max value?
print(df.max(numeric_only=True))
#How to get the mean value?
print(df.mean(numeric_only=True))
#How to get the sum value?
print(df.sum(numeric_only=True))
#How to get the median?
print(df.median(numeric_only=True))
#How to get the mode(most frequent value)?
print(df.mode())
#How to get how many non Null values
print(df.count(numeric_only=True))
#The function that groups them all
df=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\dubai_residential_data_2026.csv")
print(df.describe(include="all"))
print("---------------------")
#For a specific colomn
print(df["TRANS_VALUE"].describe())
