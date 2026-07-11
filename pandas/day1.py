"""
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
"""
"""
import pandas as pd
data=[10,20,"hello",50]
print(data)
data=pd.Series(data,index=["A","B","C"])
print(data)
print("element with index A is = ",data.loc["A"])
"""
"""
import pandas as pd
data=["new","year","new","me",2026]
data=pd.Series(data)
data.index=["8","7","6","5","4"]
print(data)
print(data.loc["6"])
"""
import pandas as pd
Series=pd.Series([1,5,"hello",13,"new"],index=["hello",17,25,"world","Ghayth"])
print(Series)
print(Series.loc["Ghayth"])