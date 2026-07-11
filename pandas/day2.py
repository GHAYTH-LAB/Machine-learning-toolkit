#day2 of learning pandas
import pandas as pd
Data=pd.Series(["A","B","C"],index=["index_1","index_2","index_3"])
#Displaying our Series
print(Data)
#Displaying an item of Series based on their index(label) 
print(Data.loc["index_1"])
 #Displaying an item of series based on their position(first item position is 0)
print(Data.iloc[0])
#this is to modify Data index
Data.index=[1,2,3]
print(Data[Data!="B"])
#--------------------------------------------------------
#Example with a python dictionnary
import pandas as pd
Dict={"first":15,"second":20,"third":40}
print(pd.Series(Dict))
print(pd.Series(Dict).loc["second"])
print(pd.Series(Dict).iloc[2])
print(pd.Series(Dict)[(pd.Series(Dict))<=20])