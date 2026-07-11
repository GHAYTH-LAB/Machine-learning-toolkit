# 1-pd.Series(arr)
import pandas as pd
arr=[10,20.5,30]
new_arr=pd.Series(arr,index=["Appartment 1","Appartment 2","Appartment 3"])
print("The list before applying Series: ")
print(arr)
print("The list after applying Series: ")
print(new_arr)
print(new_arr.loc["Appartment 1"])
#--------------------------------------------------------------------------------------------------------------------